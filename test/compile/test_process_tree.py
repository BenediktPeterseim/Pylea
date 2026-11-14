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
