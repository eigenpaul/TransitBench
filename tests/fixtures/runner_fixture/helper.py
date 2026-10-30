"""Helpers for the runner fixture: the behaviour per star and the sandbox probes, one bit each in FLAGS order."""

import hashlib
import os
import socket
from pathlib import Path

FLAGS = ("tmpdir_writable", "tmp_writable", "shm_writable", "solution_writable", "data_visible", "grade_visible",
         "network")
POOL_SUM = sum(i * i for i in range(10))


def mode(data: bytes) -> str:
    k = int.from_bytes(hashlib.sha256(data).digest()[:4], "big") % 16
    return "sleep" if k == 0 else "raise" if k == 1 else "probe"


def writable(d) -> bool:
    try:
        path = os.path.join(d, ".tb-fixture")
        with open(path, "w") as f:
            f.write("x")
        os.unlink(path)
        return True
    except (OSError, TypeError):
        return False


def network() -> bool:
    try:
        with socket.create_connection(("1.1.1.1", 53), timeout=2):
            return True
    except OSError:
        return False


def probe() -> int:
    seen = (writable(os.environ.get("TMPDIR")), writable("/tmp"), writable("/dev/shm"),
            writable(Path(__file__).resolve().parent), os.path.exists("/app/data"), os.path.exists("/opt/tb-grade"),
            network())
    return sum(int(b) << i for i, b in enumerate(seen))


def decode(period: float) -> dict[str, bool]:
    flags = int(period) - 1
    return {name: bool(flags >> i & 1) for i, name in enumerate(FLAGS)}


def square(x: int) -> int:
    return x * x


def uid() -> int:
    return os.getuid()


def cpus() -> int:
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
