"""Subprocesses that are stopped together with every process they start.

`ProcessTree` runs a worker: a subprocess that reads one pickled request from its standard
input and writes its answer to its standard output.  `kill` stops the worker and the
processes it started, such as the worker pools cotengra starts for its search, which
stopping the worker alone would leave running.

On POSIX the worker runs in a session of its own, and `kill` signals its process group.
Windows has no process groups that can be signalled, so there the worker is assigned to a
job object, and `kill` terminates the job.  The job is also set to terminate its processes
when its last handle is closed, which `close` does and which happens when this process exits.
A process the worker starts belongs to the job only if it starts after the assignment, which
`ProcessTree` makes before it returns; a worker that reads its request before it starts any
process is therefore covered whole.

**A worker also stops when its parent has gone.**  On POSIX, `send` leaves the worker's
standard input open, and the worker calls `exit_with_parent` once it has read its request,
so that the end of its input, which comes when the parent closes it or exits, ends the
worker and its process group.  A worker started by a worker is in a session of its own,
which a signal to its parent's process group does not reach; this is what stops it.  On
Windows a job created by a process in a job is nested in that job, so terminating the outer
job, or closing the inner job's handle, which happens when the process holding it exits,
already stops it, and `exit_with_parent` does nothing.
"""
from __future__ import annotations

import os
import pickle
import subprocess
import sys
import threading


class ProcessTree:
    """A worker running `args`, with every process it starts.

    `process` is the `subprocess.Popen` of the worker, with pipes to its standard input and
    output.
    """

    def __init__(self, args: list[str]):
        self.process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        start_new_session=sys.platform != "win32")
        self._job = None
        if sys.platform == "win32":
            try:
                self._job = _job_with(self.process.pid)
            except BaseException:
                self.process.kill()
                self.process.wait()
                raise

    def send(self, request) -> None:
        """Writes `request`, pickled, to the worker's standard input, which stays open.

        Raises what `pickle.dump` raises for a request that cannot be pickled.  A worker
        that has already exited shows as a missing answer, so a closed pipe raises nothing.
        """
        stdin = self.process.stdin
        assert stdin is not None
        try:
            pickle.dump(request, stdin)
            stdin.flush()
        except BrokenPipeError:
            pass

    def kill(self) -> None:
        """Stops the worker and every process it started, and waits for the worker."""
        if sys.platform == "win32":
            if self._job is not None:
                _terminate_job(self._job)
        elif self.process.returncode is None:
            # Until the worker has been waited for, its process ID, which is its process
            # group's, cannot be given to another process.
            import signal

            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.process.wait()

    def close(self) -> None:
        """Closes the worker's standard input and, on Windows, the job object.

        A worker still running then stops, together with the processes it started.
        """
        stdin = self.process.stdin
        if stdin is not None and not stdin.closed:
            try:
                stdin.close()
            except BrokenPipeError:
                pass
        if sys.platform == "win32" and self._job is not None:
            _close_handle(self._job)
            self._job = None

    def __enter__(self) -> ProcessTree:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        """Closes the tree, stopping it first where an exception leaves the block while the
        worker has not been waited for."""
        try:
            if exc_type is not None and self.process.returncode is None:
                self.kill()
        finally:
            self.close()


def exit_with_parent() -> None:
    """Ends this process and its process group once its standard input ends, on POSIX.

    The check runs on a daemon thread, so this returns at once.  A worker calls it after
    reading its request; the group it ends is then the worker and the processes it started,
    `ProcessTree` making the worker the leader of a group of its own.  On Windows the job
    objects stop the worker, so this does nothing; the module docstring says how.
    """
    if sys.platform == "win32":
        return
    stdin = sys.stdin.buffer

    def watch() -> None:
        stdin.read()
        if os.getpgrp() == os.getpid():
            import signal

            os.killpg(os.getpid(), signal.SIGKILL)
        os._exit(1)

    threading.Thread(target=watch, daemon=True).start()


if sys.platform == "win32":
    import ctypes

    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    _PROCESS_SET_QUOTA = 0x0100
    _PROCESS_TERMINATE = 0x0001

    class _BasicLimitInformation(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                    ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", ctypes.c_uint32),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", ctypes.c_uint32),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", ctypes.c_uint32),
                    ("SchedulingClass", ctypes.c_uint32)]

    class _IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class _ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _BasicLimitInformation),
                    ("IoInfo", _IoCounters),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    _kernel32.CreateJobObjectW.restype = ctypes.c_void_p
    _kernel32.SetInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                  ctypes.c_void_p, ctypes.c_uint32]
    _kernel32.SetInformationJobObject.restype = ctypes.c_int
    _kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    _kernel32.OpenProcess.restype = ctypes.c_void_p
    _kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _kernel32.AssignProcessToJobObject.restype = ctypes.c_int
    _kernel32.TerminateJobObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    _kernel32.TerminateJobObject.restype = ctypes.c_int
    _kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    _kernel32.CloseHandle.restype = ctypes.c_int

    def _checked(result):
        if not result:
            raise ctypes.WinError(ctypes.get_last_error())
        return result

    def _job_with(pid: int) -> int:
        """A job object holding process `pid`, which terminates its processes when closed."""
        job = _checked(_kernel32.CreateJobObjectW(None, None))
        try:
            limits = _ExtendedLimitInformation()
            limits.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            _checked(_kernel32.SetInformationJobObject(
                job, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS, ctypes.byref(limits),
                ctypes.sizeof(limits)))
            process = _checked(_kernel32.OpenProcess(
                _PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, pid))
            try:
                _checked(_kernel32.AssignProcessToJobObject(job, process))
            finally:
                _kernel32.CloseHandle(process)
        except BaseException:
            _kernel32.CloseHandle(job)
            raise
        return job

    def _terminate_job(job: int) -> None:
        _checked(_kernel32.TerminateJobObject(job, 1))

    def _close_handle(job: int) -> None:
        _kernel32.CloseHandle(job)
