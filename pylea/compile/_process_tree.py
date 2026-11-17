"""Subprocesses that are stopped together with every process they start.

`ProcessTree` runs a worker: a subprocess that reads one pickled request from its standard
input and writes one pickled answer to its standard output.  A worker's `main` is
``serve(handle)``, which reads the request, computes ``handle(request)`` and writes the
answer, turning any exception into an answer ``{"error": "<type>: <message>"}``.  The
parent reads the answer with `receive`, which returns as soon as the answer is written,
not when the worker ends.  `kill` stops the worker and the processes it started, such as
the worker pools cotengra starts for its search, which stopping the worker alone would
leave running.  Leaving a `with` block stops the tree, so once the answer is in hand
nothing waits for the rest of the worker's shutdown.  `start_worker` starts a module of
this package as a worker, with this process's `sys.path`.

On POSIX the worker runs in a session of its own, and `kill` signals its process group.
The group is signalled only while the worker has not been waited for: until then the
worker's process ID, which is the group's, cannot be given to another process, and only
`kill` waits for the worker.  Windows has no process groups that can be signalled, so
there the worker is assigned to a job object, and `kill` terminates the job.  The job is
also set to terminate its processes when its last handle is closed, which `close` does and
which happens when this process exits.  A process the worker starts belongs to the job
only if it starts after the assignment, which `ProcessTree` makes before it returns; a
worker that reads its request before it starts any process is therefore covered whole.

**A worker also stops when its parent has gone.**  On POSIX, `send` leaves the worker's
standard input open, and `serve` calls `exit_with_parent` once it has read the request,
so that the end of its input, which comes when the parent closes it or exits, ends the
worker and its process group; so does a change of its parent process, for a parent that
has gone while a process it forked still holds the input open.  The watcher reads the input's file descriptor directly,
never `sys.stdin`, so that it holds no lock the interpreter needs at shutdown.  A worker
started by a worker is in a session of its own, which a signal to its parent's process
group does not reach; this is what stops it.  On Windows a job created by a process in a
job is nested in that job, so terminating the outer job, or closing the inner job's
handle, which happens when the process holding it exits, already stops it, and
`exit_with_parent` does nothing.

**A worker stops when its tree is dropped.**  No thread refers to a tree, and collecting
a tree closes the worker's standard input, so a tree that an interrupt took from its
caller before the caller held it ends its worker as well.
"""
from __future__ import annotations

import contextlib
import errno
import math
import os
import pickle
import subprocess
import sys
import threading
from collections.abc import Callable
from typing import Any


def join_timeout(seconds: float | None) -> float | None:
    """`seconds` as a timeout `threading` accepts: `None`, no limit, for `None`, infinity
    and anything at or above `threading.TIMEOUT_MAX`, and 0 for a negative value.

        >>> join_timeout(1e300) is None, join_timeout(-1.0), join_timeout(2.5)
        (True, 0.0, 2.5)
        >>> join_timeout(10**400) is None
        True

    Raises `ValueError` for NaN.
    """
    if seconds is None or seconds >= threading.TIMEOUT_MAX:
        return None
    if seconds <= 0:
        return 0.0
    if math.isnan(seconds):
        raise ValueError("a timeout cannot be NaN")
    return float(seconds)


class DeadlineExceeded(TimeoutError):
    """Raised where a worker has not answered within the time it was given.

    A subclass, so that a `TimeoutError` of the caller's own, such as one raised by a
    signal handler, is never taken for a missed deadline.
    """


# Run by `python -c`: reads this process's `sys.path` from stdin, then runs the module
# named in `sys.argv[1]` as `__main__`.  A worker whose parent dropped it before sending
# anything ends quietly.
_BOOTSTRAP = """
import pickle, runpy, sys
try:
    path = pickle.load(sys.stdin.buffer)
except EOFError:
    sys.exit(0)
sys.path[:] = path
runpy.run_module(sys.argv[1], run_name="__main__", alter_sys=True)
"""


def start_worker(module: str) -> ProcessTree:
    """A `ProcessTree` running `module` as a worker, with this process's `sys.path`.

    The worker then imports what this process imports, from the same places: `pylea`
    found through a change to `sys.path`, and a class defined in a module beside the
    script being run, which an `order_finder` may be.  The path travels on the worker's
    standard input, ahead of the request, so its length is not limited by the command
    line; entries that are not strings, which the import system ignores, are left out.
    """
    # Plain strings: the bootstrap unpickles the path before it can import anything a
    # subclass would need.
    path = [str.__str__(entry) for entry in sys.path if isinstance(entry, str)]
    # `-P`: no folder in front of `sys.path` before the bootstrap replaces it, so that a
    # module in the working directory cannot stand in for one the bootstrap imports.
    return ProcessTree([sys.executable, "-P", "-c", _BOOTSTRAP, module],
                       preamble=pickle.dumps(path))


class _Answer:
    """What the thread reading a worker's output found: the answer, or the exception
    that ended reading, and an event set once reading has ended."""

    def __init__(self) -> None:
        self.value: Any = None
        self.failure: BaseException | None = None
        self.read = threading.Event()


def _read_answer(stdout, answer: _Answer) -> None:
    try:
        answer.value = pickle.load(stdout)
    except BaseException as exc:                # noqa: BLE001, reported by `receive`
        answer.failure = exc
    finally:
        try:
            stdout.close()
        except OSError:
            pass
        answer.read.set()


class _WorkerPopen(subprocess.Popen):
    """A `subprocess.Popen` that closes the worker's standard input when it is collected.

    A `Popen` collected while its process runs is kept by `subprocess` until the
    process ends, its pipes still open.  A worker whose tree was dropped before anything
    held it, as when an interrupt arrives just as the worker starts, would then wait for
    its input until this process exits; with the input closed it ends at once.
    """

    def __del__(self, *args, **kwargs):
        stdin = getattr(self, "stdin", None)
        if stdin is not None:
            try:
                stdin.close()
            except OSError:
                pass
        super().__del__(*args, **kwargs)


class ProcessTree:
    """A worker running `args`, with every process it starts.

    `process` is the `subprocess.Popen` of the worker, with pipes to its standard input and
    output.  A thread reads one pickled answer from the output as soon as the worker
    starts, so that a large answer never fills the pipe while this process waits, and
    closes the output once it has read it.
    """

    def __init__(self, args: list[str], *, preamble: bytes = b""):
        # Nothing here keeps a reference to the tree, not even the thread reading the
        # answer, so that a tree nothing holds is collected and its worker ends.
        self._preamble = preamble
        self.process = _WorkerPopen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    start_new_session=sys.platform != "win32")
        self._job = None
        try:
            if sys.platform == "win32":
                self._job = _job_with(self.process.pid)
            self._answer = _Answer()
            threading.Thread(target=_read_answer, args=(self.process.stdout, self._answer),
                             daemon=True).start()
        except BaseException:
            # The caller never receives this tree, so nothing else would stop it.
            if self._job is None:
                self.process.kill()
            self.kill()
            self.close()
            if self.process.stdout is not None:
                self.process.stdout.close()
            raise

    def __del__(self) -> None:
        if sys.platform == "win32" and getattr(self, "_job", None) is not None:
            _close_handle(self._job)

    def send(self, request) -> None:
        """Writes `request`, pickled, to the worker's standard input, which stays open.

        Returns once the request is pickled: a thread writes it, since a request larger
        than the pipe's buffer is written only as fast as the worker, which first has to
        start, reads it, and the wait belongs to the time `receive` allows.  Raises what
        `pickle.dumps` raises for a request that cannot be pickled.  A worker that has
        exited shows as a missing answer, so a closed pipe raises nothing.
        """
        stdin = self.process.stdin
        assert stdin is not None
        data = self._preamble + pickle.dumps(request)
        # The thread writes through a descriptor of its own, which `close` does not
        # touch, so that closing stdin cannot hand its number to another file while the
        # thread still writes to it, and the thread holds no lock of a buffered object
        # that the interpreter needs at exit.
        fd = os.dup(stdin.fileno())
        # Whichever of this thread and the writer claims the descriptor first closes it;
        # an interrupt while the writer starts must neither leak nor close it twice.
        claim = threading.Lock()
        owner: list[str] = []

        def write() -> None:
            with claim:
                if owner:
                    return
                owner.append("writer")
            _no_broken_pipe_signal(fd)
            view = memoryview(data)
            try:
                while view:
                    view = view[os.write(fd, view):]
            except OSError:
                pass
            finally:
                os.close(fd)

        try:
            threading.Thread(target=write, daemon=True).start()
        except BaseException:
            with claim:
                if not owner:
                    owner.append("caller")
                    os.close(fd)
            raise

    def answered(self) -> bool:
        """Whether reading the answer has ended, with an answer or with the end of the
        worker's output; `receive` then returns or raises at once."""
        return self._answer.read.is_set()

    def receive(self, timeout: float | None = None):
        """The worker's answer, waiting at most `timeout` seconds for it (`join_timeout`
        says which values mean no limit).

        Raises `DeadlineExceeded`, a `TimeoutError`, where no answer has come in time,
        having stopped the tree, and `RuntimeError`, also having stopped the tree, where
        the worker's output ended without a complete answer.
        """
        if not self._answer.read.wait(join_timeout(timeout)):
            self.kill()
            raise DeadlineExceeded(f"the worker did not answer within {timeout} s")
        failure = self._answer.failure
        if failure is not None:
            status = self._exit_status()
            self.kill()
            if status is None and sys.platform != "win32":
                # Read after `kill`: the worker's own status if it had ended.  A status
                # of the kill itself says nothing about the worker, and is left out.
                # Where `SIGCHLD` is ignored, or handled by the program, which may collect
                # the worker itself, `subprocess` reports 0 for a status it never saw.
                import signal

                if (self.process.returncode != -signal.SIGKILL
                        and signal.getsignal(signal.SIGCHLD) == signal.SIG_DFL):
                    status = self.process.returncode
            ended = "" if status is None else f"exit status {status}; "
            raise RuntimeError(
                f"the worker ended without an answer ({ended}"
                f"{type(failure).__name__}: {failure})") from failure
        return self._answer.value

    def _exit_status(self, within: float = 1.0) -> int | None:
        """The worker's exit status, as `subprocess` gives it, if it ends by itself
        within `within` seconds; `None` otherwise, and always `None` on a POSIX system
        without `os.waitid` (macOS with Python 3.12).

        On POSIX the status is read without waiting for the worker, so that `kill` may
        still signal its group; see the module docstring.
        """
        if sys.platform == "win32":
            try:
                return self.process.wait(timeout=within)
            except subprocess.TimeoutExpired:
                return None
        if not hasattr(os, "waitid"):
            return None
        import time

        deadline = time.monotonic() + within
        while True:
            try:
                info = os.waitid(os.P_PID, self.process.pid,
                                 os.WEXITED | os.WNOHANG | os.WNOWAIT)
            except ChildProcessError:           # `SIGCHLD` ignored: no status is kept
                return None
            if info is not None and info.si_pid:
                return (info.si_status if info.si_code == os.CLD_EXITED
                        else -info.si_status)
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.01)

    def kill(self) -> None:
        """Stops the worker and every process it started, and waits for the worker.

        A second call does nothing on POSIX.
        """
        if sys.platform == "win32":
            if self._job is not None:
                _terminate_job(self._job)
        elif self.process.returncode is None:
            # Until the worker has been waited for, its process ID, which is its process
            # group's, cannot be given to another process.  macOS raises `PermissionError`
            # for a group whose only member is the worker, exited but not waited for.
            import signal

            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        self.process.wait()

    def close(self) -> None:
        """Closes the worker's standard input and, on Windows, the job object.

        A worker still running then stops, together with the processes it started.  The
        worker's output is closed by the thread reading the answer, once it has read it.
        """
        stdin = self.process.stdin
        if stdin is not None and not stdin.closed:
            try:
                stdin.close()
            except OSError as exc:
                if not _reader_gone(exc):
                    raise
        if sys.platform == "win32" and self._job is not None:
            _close_handle(self._job)
            self._job = None

    def __enter__(self) -> ProcessTree:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        """Stops the tree and closes it."""
        try:
            self.kill()
        finally:
            self.close()


def _no_broken_pipe_signal(fd: int) -> None:
    """Nothing, a write to `fd` after the worker has gone now failing with `EPIPE`
    rather than delivering `SIGPIPE`, which ends the whole process where the program has
    restored the signal's default action.

    macOS sends the signal to the process, so the descriptor is marked there, with
    `F_SETNOSIGPIPE`, whose value 73 is fixed by macOS's `<sys/fcntl.h>` and which
    Python's `fcntl` module names only from Python 3.13 on; Linux sends it to the writing
    thread, which therefore blocks it.  Called by the thread that writes.
    """
    if sys.platform == "win32":
        return
    import fcntl
    import signal

    if sys.platform == "darwin":
        fcntl.fcntl(fd, getattr(fcntl, "F_SETNOSIGPIPE", 73), 1)
    else:
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGPIPE})


def serve(handle: Callable[[Any], Any]) -> None:
    """Answers the one request on this worker's standard input with ``handle(request)``.

    Any exception, in reading the request or in `handle`, becomes the answer
    ``{"error": "<type>: <message>"}``.  `exit_with_parent` is called once the request
    has been read, so that the worker stops when its parent has gone.  Everything else
    written to standard output, by `handle` or by a process it starts, goes to standard
    error instead, so that only the answer reaches the parent's reader.
    """
    # Unbuffered, so that an answer the parent no longer reads leaves nothing behind to
    # flush when the interpreter ends.
    answer = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    sys.stdout.flush()
    if sys.stderr is not None:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    else:                                       # the parent started it without stderr
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        os.close(devnull)
    try:
        request = pickle.load(sys.stdin.buffer)
    except Exception as exc:                    # noqa: BLE001, reported to the parent
        response: Any = error_response(exc)
    else:
        exit_with_parent()
        try:
            response = handle(request)
        except Exception as exc:                # noqa: BLE001, reported to the parent
            response = error_response(exc)
    try:
        data = pickle.dumps(response)
    except Exception as exc:                    # noqa: BLE001, reported to the parent
        data = pickle.dumps(error_response(exc))
    try:
        view = memoryview(data)
        while view:
            view = view[answer.write(view):]
    except OSError as exc:
        if not _reader_gone(exc):
            raise
    try:
        answer.close()
    except OSError as exc:
        if not _reader_gone(exc):
            raise


def _reader_gone(exc: OSError) -> bool:
    """Whether `exc`, raised by a write to a pipe, says that nothing reads the pipe any
    more: `BrokenPipeError`, or on Windows `EINVAL`, which a write to a pipe whose reader
    has closed raises there."""
    return isinstance(exc, BrokenPipeError) or (
        sys.platform == "win32" and exc.errno == errno.EINVAL)


def error_response(exc: BaseException) -> dict:
    """The answer `serve` writes for `exc`."""
    return {"error": f"{type(exc).__name__}: {exc}"}


def exit_with_parent() -> None:
    """Ends this process and its process group once its standard input ends, or its
    parent process changes, on POSIX.

    The check runs on a daemon thread, so this returns at once.  A worker calls it after
    reading its request; the group it ends is then the worker and the processes it started,
    `ProcessTree` making the worker the leader of a group of its own.  The parent is
    checked as well, once a second, because a process the parent forked keeps the input
    open after the parent has gone.  On Windows the job objects stop the worker, so this
    does nothing; the module docstring says how.
    """
    if sys.platform == "win32":
        return
    import select

    fd = sys.stdin.fileno()
    parent = os.getppid()

    def watch() -> None:
        try:
            while os.getppid() == parent:
                ready, _, _ = select.select([fd], [], [], 1.0)
                if ready and not os.read(fd, 65536):
                    break
        except OSError:
            pass
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
