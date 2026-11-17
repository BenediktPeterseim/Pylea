"""Stopping a `ProcessTree`.

What must hold is that no process a worker started outlives the worker's tree being killed
or closed, including a worker it started in a tree of its own, which a signal to its own
process group does not reach.  Whether a process still runs is read from a file it appends
to, which works the same on every platform.
"""
import sys
import time

import pytest

from pylea.compile._process_tree import ProcessTree

# Appends to the file named in its request until it is stopped.
_INNER = """
import pickle, sys, time
from pylea.compile._process_tree import exit_with_parent
path = pickle.load(sys.stdin.buffer)
exit_with_parent()
while True:
    with open(path, "a") as f:
        f.write(".")
    time.sleep(0.02)
"""

# Starts `_INNER` in a tree of its own, then waits.  Each step, and any exception, is
# appended to a log beside the file `_INNER` writes, which a failing test reports.
_OUTER = """
import pickle, sys, time, traceback
inner, path = pickle.load(sys.stdin.buffer)

def log(text):
    with open(path + ".log", "a") as f:
        f.write(text + "\\n")

try:
    from pylea.compile._process_tree import ProcessTree, exit_with_parent
    exit_with_parent()
    log("request read")
    tree = ProcessTree([sys.executable, "-c", inner])
    log("inner worker started")
    tree.send(path)
    log("request sent")
except BaseException:
    log(traceback.format_exc())
    raise
time.sleep(600)
"""


def _grows(path, within: float) -> bool:
    """Whether the file at `path` grows within `within` seconds."""
    before = path.stat().st_size if path.exists() else 0
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        time.sleep(0.05)
        if path.exists() and path.stat().st_size > before:
            return True
    return False


def _nested(path) -> ProcessTree:
    tree = ProcessTree([sys.executable, "-c", _OUTER])
    tree.send((_INNER, str(path)))
    if not _grows(path, within=60.0):
        tree.kill()
        tree.close()
        log = path.with_name(path.name + ".log")
        steps = log.read_text() if log.exists() else "nothing logged"
        pytest.fail(f"the inner worker never started; the outer worker logged:\n{steps}")
    return tree


def _stops(path) -> bool:
    """Whether the file at `path` stops growing within a few seconds."""
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if not _grows(path, within=1.0):
            return True
    return False


def test_killing_a_tree_stops_a_worker_started_in_a_tree_of_its_own(tmp_path):
    path = tmp_path / "beats"
    tree = _nested(path)
    try:
        tree.kill()
        assert _stops(path)
    finally:
        tree.kill()
        tree.close()


def test_closing_the_input_of_a_running_worker_stops_it_and_the_workers_it_started(
        tmp_path):
    path = tmp_path / "beats"
    tree = _nested(path)
    try:
        tree.close()
        assert _stops(path)
        tree.process.wait(timeout=10.0)
    finally:
        tree.kill()
        tree.close()


def _request_two_tensors():
    return ("adaptive", ("greedy", 1e30, None, False),
            [([(0, 1), (1, 2)], (0, 2), {0: 2, 1: 2, 2: 2})], None)


def _group_is_empty(pid: int) -> bool:
    import os

    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return True
    return False


def test_a_worker_that_answers_ends_with_status_zero_and_reports_no_fatal_error():
    import pickle
    import subprocess

    worker = subprocess.Popen([sys.executable, "-m", "pylea.compile._order_worker"],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
    try:
        assert worker.stdin is not None and worker.stdout is not None
        pickle.dump(_request_two_tensors(), worker.stdin)
        worker.stdin.flush()
        assert "orders" in pickle.load(worker.stdout)
        worker.wait(timeout=60)
        assert worker.returncode == 0
        assert worker.stderr is not None
        assert b"Fatal Python error" not in worker.stderr.read()
    finally:
        worker.kill()
        worker.wait()
        for stream in (worker.stdin, worker.stdout, worker.stderr):
            if stream is not None:
                stream.close()


def test_receive_returns_once_the_answer_is_written_not_when_the_worker_ends():
    code = ("import pickle, sys, time\n"
            "pickle.dump('answer', sys.stdout.buffer)\n"
            "sys.stdout.buffer.flush()\n"
            "time.sleep(30)\n")
    started = time.monotonic()
    with ProcessTree([sys.executable, "-c", code]) as tree:
        assert tree.receive(60) == "answer"
        assert time.monotonic() - started < 2.0 + 5.0   # interpreter start included
    assert time.monotonic() - started < 15.0
    if sys.platform != "win32":
        assert _group_is_empty(tree.process.pid)


def test_kill_after_the_worker_has_ended_by_itself_raises_nothing():
    tree = ProcessTree([sys.executable, "-c", "pass"])
    time.sleep(0.3)
    tree.kill()
    tree.kill()
    tree.close()
    assert tree.process.returncode is not None


@pytest.mark.skipif(sys.platform == "win32", reason="process groups are POSIX")
def test_the_group_is_never_signalled_after_the_worker_has_been_waited_for(monkeypatch):
    import os

    killpg = os.killpg
    recorded = []
    with ProcessTree([sys.executable, "-m", "pylea.compile._order_worker"]) as tree:
        def recording(pid, sig):
            recorded.append(tree.process.returncode)
            killpg(pid, sig)

        monkeypatch.setattr(os, "killpg", recording)
        tree.send(_request_two_tensors())
        assert "orders" in tree.receive(60)
        tree.kill()
        tree.kill()
    assert recorded and all(code is None for code in recorded)


def test_a_partial_answer_raises_runtime_error():
    code = ("import pickle, sys\n"
            "sys.stdout.buffer.write(pickle.dumps(list(range(1000)))[:50])\n"
            "sys.stdout.buffer.flush()\n")
    with ProcessTree([sys.executable, "-c", code]) as tree:
        with pytest.raises(RuntimeError, match="without an answer"):
            tree.receive(60)


def test_an_unreadable_request_is_answered_with_an_error():
    code = ("from pylea.compile._process_tree import serve\n"
            "serve(lambda request: request)\n")
    with ProcessTree([sys.executable, "-c", code]) as tree:
        stdin = tree.process.stdin
        assert stdin is not None
        stdin.write(b"this is not a pickle")
        stdin.flush()
        stdin.close()
        answer = tree.receive(60)
    assert "UnpicklingError" in answer["error"]


def test_a_timeout_past_what_threading_accepts_waits_without_a_limit():
    with ProcessTree([sys.executable, "-c",
                      "import pickle, sys; pickle.dump(1, sys.stdout.buffer)"]) as tree:
        assert tree.receive(1e300) == 1


def test_a_timeout_beyond_the_float_range_means_no_limit():
    from pylea.compile._process_tree import join_timeout
    from pylea.compile.order import AdaptiveOrderFinder

    assert join_timeout(10 ** 400) is None and join_timeout(-(10 ** 400)) == 0.0
    assert AdaptiveOrderFinder(max_time=10 ** 400).max_time is None
    for bad in (float("nan"), -1.0):
        with pytest.raises(ValueError, match="max_time"):
            AdaptiveOrderFinder(max_time=bad)


def test_a_worker_whose_parent_has_gone_reports_nothing():
    import subprocess

    code = ("import os, sys, time\n"
            "from pylea.compile._process_tree import serve\n"
            "time.sleep(0.5)\n"                 # the parent's end of stdout closes meanwhile
            "serve(lambda request: request)\n")
    worker = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert worker.stdout is not None and worker.stdin is not None
    worker.stdout.close()
    worker.stdin.write(b"partial")
    worker.stdin.close()
    worker.wait(timeout=60)
    assert worker.stderr is not None
    err = worker.stderr.read()
    worker.stderr.close()
    assert b"BrokenPipeError" not in err and b"Traceback" not in err


def test_what_a_worker_prints_does_not_reach_its_answer():
    code = ("from pylea.compile._process_tree import serve\n"
            "def handle(request):\n"
            "    print('x' * 100_000)\n"
            "    return request + 1\n"
            "serve(handle)\n")
    with ProcessTree([sys.executable, "-c", code]) as tree:
        tree.send(41)
        assert tree.receive(60) == 42


def test_a_finder_is_inspected_once_per_class(monkeypatch):
    import inspect

    from pylea.compile.order import GreedyOrderFinder, check_finder_and_kappa

    class Counted(GreedyOrderFinder):
        pass

    calls = []
    signature = inspect.signature
    monkeypatch.setattr(inspect, "signature", lambda f: calls.append(f) or signature(f))
    for _ in range(5):
        check_finder_and_kappa(Counted(), 3)
    assert len(calls) == 1
    with pytest.raises(TypeError, match="max_time"):
        from pylea.compile.order import AdaptiveOrderFinder
        AdaptiveOrderFinder(max_time="1")         # type: ignore[arg-type]


@pytest.mark.skipif(sys.platform == "win32", reason="closes a descriptor in a POSIX shell")
def test_a_worker_answers_when_its_parent_has_no_standard_error():
    import subprocess

    code = ("from pylea import DPP, Bool, flip, prob\n"
            "@DPP\n"
            "def coin() -> Bool:\n"
            "    return flip(0.3)\n"
            "assert abs(prob(coin, method='exact', max_time=60) - 0.3) < 1e-12\n")
    subprocess.run(["/bin/sh", "-c", '"$0" -c "$1" 2>&-', sys.executable, code],
                   check=True, timeout=120)


def test_sending_a_large_request_does_not_wait_for_the_worker_to_read_it():
    code = ("import pickle, sys, time\n"
            "time.sleep(3)\n"
            "pickle.dump(len(pickle.load(sys.stdin.buffer)), sys.stdout.buffer)\n")
    with ProcessTree([sys.executable, "-c", code]) as tree:
        started = time.monotonic()
        tree.send(b"x" * 10_000_000)
        assert time.monotonic() - started < 1.0
        assert tree.receive(60) == 10_000_000


def _open_descriptors() -> int:
    import os

    return len(os.listdir("/dev/fd")) if os.path.isdir("/dev/fd") else 0


@pytest.mark.skipif(sys.platform == "win32", reason="counts POSIX descriptors")
def test_an_interrupt_while_a_request_is_handed_to_its_writer_leaks_no_descriptor(
        monkeypatch):
    import pylea.compile._process_tree as process_tree

    with ProcessTree([sys.executable, "-c", "import time; time.sleep(30)"]) as tree:
        before = _open_descriptors()

        class Interrupted:
            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                raise KeyboardInterrupt

        monkeypatch.setattr(process_tree.threading, "Thread", Interrupted)
        with pytest.raises(KeyboardInterrupt):
            tree.send("request")
        monkeypatch.undo()
        assert _open_descriptors() == before


def test_a_lane_interrupted_while_stopping_still_closes_its_pipe(monkeypatch):
    from pylea.sample.exact_lane import ExactLane

    lane = ExactLane(None, {})
    tree = lane._tree
    real_kill = tree.kill

    def interrupted():
        real_kill()
        raise KeyboardInterrupt

    monkeypatch.setattr(tree, "kill", interrupted)
    with pytest.raises(KeyboardInterrupt):
        lane.kill()
    assert tree.process.stdin.closed


def test_a_worker_imports_the_pylea_its_parent_found_through_sys_path(tmp_path):
    import os
    import subprocess

    import pylea

    root = os.path.dirname(os.path.dirname(os.path.abspath(pylea.__file__)))
    code = (f"import sys\nsys.path.insert(0, {root!r})\n"
            "from pylea import DPP, Bool, flip, prob\n"
            "@DPP\n"
            "def coin() -> Bool:\n"
            "    return flip(0.3)\n"
            "assert abs(prob(coin, method='exact', max_time=60) - 0.3) < 1e-12\n")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120, cwd=tmp_path,
                   env=env)


@pytest.mark.skipif(sys.platform == "win32", reason="SIGPIPE is POSIX")
def test_a_request_left_unread_never_ends_the_parent_by_sigpipe():
    import subprocess

    code = ("import signal, sys, time\n"
            "from pylea.compile._process_tree import ProcessTree\n"
            "signal.signal(signal.SIGPIPE, signal.SIG_DFL)\n"
            "for _ in range(5):\n"
            "    tree = ProcessTree([sys.executable, '-c', 'pass'])\n"
            "    tree.send(b'x' * 1_000_000)\n"
            "    tree.kill()\n"
            "    tree.close()\n"
            "    time.sleep(0.2)\n"
            "print('survived')\n")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=120)
    assert done.returncode == 0 and b"survived" in done.stdout


def test_a_worker_imports_a_module_beside_the_script_that_started_it(tmp_path):
    import os
    import subprocess

    (tmp_path / "myfinders.py").write_text(
        "from pylea.compile.order import GreedyOrderFinder\n"
        "class MyFinder(GreedyOrderFinder):\n"
        "    pass\n")
    (tmp_path / "main.py").write_text(
        "from myfinders import MyFinder\n"
        "from pylea import DPP, Bool, flip, prob\n"
        "@DPP\n"
        "def coin() -> Bool:\n"
        "    return flip(0.3)\n"
        "p = prob(coin, method='exact', max_time=60, order_finder=MyFinder())\n"
        "assert abs(p - 0.3) < 1e-12, p\n")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    subprocess.run([sys.executable, str(tmp_path / "main.py")], check=True, timeout=120,
                   cwd=elsewhere, env=dict(os.environ))


def test_the_exit_status_of_a_worker_that_ended_without_an_answer_is_its_own():
    with ProcessTree([sys.executable, "-c", "import sys; sys.exit(3)"]) as tree:
        with pytest.raises(RuntimeError, match="exit status 3"):
            tree.receive(60)


def test_an_interrupt_while_a_request_is_sent_leaves_no_worker(monkeypatch):
    import os

    import pylea.sample.exact_lane as exact_lane

    trees = []
    start = exact_lane.start_worker

    def recorded(module):
        trees.append(start(module))
        return trees[-1]

    def interrupted(self, request):
        raise KeyboardInterrupt                 # as if Ctrl-C came while sending

    monkeypatch.setattr(exact_lane, "start_worker", recorded)
    monkeypatch.setattr(ProcessTree, "send", interrupted)
    with pytest.raises(KeyboardInterrupt):
        exact_lane.ExactLane(object(), {})
    (tree,) = trees
    assert tree.process.returncode is not None
    if sys.platform != "win32":
        with pytest.raises(ProcessLookupError):
            os.killpg(tree.process.pid, 0)


@pytest.mark.skipif(sys.platform == "win32", reason="counts POSIX descriptors")
def test_an_interrupt_while_a_tree_starts_stops_the_worker_and_closes_its_pipes(
        monkeypatch):
    import os

    import pylea.compile._process_tree as process_tree

    started = []

    class Recorded(process_tree._WorkerPopen):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            started.append(self)

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    before = _open_descriptors()
    monkeypatch.setattr(process_tree, "_WorkerPopen", Recorded)
    monkeypatch.setattr(process_tree.threading, "Event", interrupted)
    with pytest.raises(KeyboardInterrupt):
        ProcessTree([sys.executable, "-c", "import time; time.sleep(30)"])
    monkeypatch.undo()
    (worker,) = started
    assert worker.returncode is not None
    with pytest.raises(ProcessLookupError):
        os.killpg(worker.pid, 0)
    assert _open_descriptors() == before


def test_a_tree_that_nothing_holds_stops_its_worker():
    """What happens to a tree an interrupt took from its caller before the caller held
    it: the worker waits for its request, and must end once the tree is collected."""
    import gc
    import os

    from pylea.compile._process_tree import start_worker

    tree = start_worker("pylea.sample._exact_worker")
    worker = tree.process.pid
    del tree
    gc.collect()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            done = os.waitpid(worker, os.WNOHANG)[0] == worker
        except ChildProcessError:
            done = True
        if done:
            break
        time.sleep(0.05)
    else:
        pytest.fail("the worker of a dropped tree is still running")


def test_entries_of_sys_path_that_are_not_plain_strings_do_not_break_the_worker(
        monkeypatch):
    import pathlib

    from pylea import DPP, Bool, flip, prob

    class Entry(str):
        pass

    monkeypatch.setattr(sys, "path", sys.path + [pathlib.Path("/nonexistent/dir"),
                                                 Entry("/nonexistent/subclass"),
                                                 "/nonexistent/" + "x" * 200_000])

    @DPP
    def coin() -> Bool:
        return flip(0.3)

    assert abs(prob(coin, method="exact", max_time=60) - 0.3) < 1e-12



def test_a_module_in_the_working_directory_cannot_stand_in_for_one_the_worker_needs(
        tmp_path, monkeypatch):
    from pylea import DPP, Bool, flip, prob

    (tmp_path / "struct.py").write_text("raise ImportError('the wrong struct')\n")
    monkeypatch.chdir(tmp_path)

    @DPP
    def coin() -> Bool:
        return flip(0.3)

    assert abs(prob(coin, method="exact", max_time=60) - 0.3) < 1e-12


@pytest.mark.skipif(sys.platform == "win32", reason="process groups are POSIX")
@pytest.mark.parametrize("waitid", [True, False])
def test_a_worker_that_crashes_takes_the_processes_it_started_with_it(
        tmp_path, monkeypatch, waitid):
    """Without `os.waitid`, as on macOS with Python 3.12, the status is read after the
    group has been signalled, never by waiting first."""
    import os

    if not waitid:
        monkeypatch.delattr(os, "waitid", raising=False)
    record = tmp_path / "pid"
    code = ("import os, subprocess, sys, time\n"
            "child = subprocess.Popen(['sleep', '30'], stdout=subprocess.DEVNULL)\n"
            f"open({str(record)!r}, 'w').write(str(child.pid))\n"
            "os._exit(3)\n")
    started = time.monotonic()
    with ProcessTree([sys.executable, "-c", code]) as tree:
        with pytest.raises(RuntimeError, match="exit status 3"):
            tree.receive(60)
    assert time.monotonic() - started < 15.0
    child = int(record.read_text())
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail("a process the crashed worker started is still running")


def test_an_answer_that_cannot_be_pickled_is_reported_as_an_error():
    code = ("from pylea.compile._process_tree import serve\n"
            "serve(lambda request: lambda: request)\n")
    with ProcessTree([sys.executable, "-c", code]) as tree:
        tree.send(1)
        answer = tree.receive(60)
    assert "local object" in answer["error"]


@pytest.mark.skipif(sys.platform == "win32", reason="os.waitid is POSIX")
def test_without_waitid_a_status_that_the_kill_caused_is_not_reported(monkeypatch):
    import os

    monkeypatch.delattr(os, "waitid", raising=False)
    code = "import os, time; os.close(1); time.sleep(30)"     # no answer, still running
    started = time.monotonic()
    with ProcessTree([sys.executable, "-c", code]) as tree:
        with pytest.raises(RuntimeError) as raised:
            tree.receive(60)
    assert "exit status" not in str(raised.value)
    assert time.monotonic() - started < 15.0


@pytest.mark.skipif(sys.platform == "win32", reason="SIGCHLD is POSIX")
def test_an_exit_status_the_system_did_not_keep_is_not_reported(monkeypatch):
    import signal

    monkeypatch.setattr(signal, "SIGCHLD", signal.SIGCHLD)
    previous = signal.signal(signal.SIGCHLD, signal.SIG_IGN)
    try:
        with ProcessTree([sys.executable, "-c", "import os; os._exit(3)"]) as tree:
            with pytest.raises(RuntimeError) as raised:
                tree.receive(60)
    finally:
        signal.signal(signal.SIGCHLD, previous)
    assert "exit status" not in str(raised.value)


@pytest.mark.skipif(sys.platform == "win32", reason="fork is POSIX")
def test_a_worker_stops_when_its_parent_dies_while_a_forked_child_holds_its_input(
        tmp_path):
    import os
    import subprocess

    record = tmp_path / "worker"
    code = ("import os, sys, time\n"
            "from pylea.compile._process_tree import ProcessTree\n"
            "worker = ('from pylea.compile._process_tree import serve\\n'\n"
            "          'import time\\n'\n"
            "          'serve(lambda request: time.sleep(60))\\n')\n"
            "tree = ProcessTree([sys.executable, '-c', worker])\n"
            "tree.send('request')\n"
            "time.sleep(1.0)\n"
            f"open({str(record)!r}, 'w').write(str(tree.process.pid))\n"
            "if os.fork() == 0:\n"
            "    time.sleep(20)\n"
            "    os._exit(0)\n"
            "os._exit(0)\n")                     # the parent dies without closing anything
    subprocess.run([sys.executable, "-c", code], timeout=60)
    worker = int(record.read_text())
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(worker, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    pytest.fail("the worker outlived its parent")


@pytest.mark.skipif(sys.platform == "win32", reason="SIGCHLD is POSIX")
def test_no_false_exit_status_is_reported_where_the_program_collects_children():
    import os
    import signal

    def collect(signum, frame):
        try:
            while os.waitpid(-1, os.WNOHANG)[0]:
                pass
        except ChildProcessError:
            pass

    previous = signal.signal(signal.SIGCHLD, collect)
    try:
        with ProcessTree([sys.executable, "-c", "import os, time; os._exit(3)"]) as tree:
            with pytest.raises(RuntimeError) as raised:
                tree.receive(60)
    finally:
        signal.signal(signal.SIGCHLD, previous)
    message = str(raised.value)
    assert "exit status" not in message or "exit status 3" in message
