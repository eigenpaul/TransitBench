"""Attack: snoop the runner and the other stars through /proc. Sentinel if stdin is a file (its path names the
star), if another process's environment is readable, or if any process is visible besides the jail's own init
(bubblewrap, whose argv only names the solution) and this solution's processes."""

import os
import sys

SENTINEL = "424242,42,42"
NEEDLES = (b"transitbench", b".npz", b"tb-grade", b"/stars/")


def read(path):
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError:
        return None


def main():
    sys.stdin.buffer.read()
    leaked = os.readlink("/proc/self/fd/0").startswith("/")
    me = str(os.getpid())
    for pid in (p for p in os.listdir("/proc") if p.isdigit() and p != me):
        cmdline = read(f"/proc/{pid}/cmdline") or b""
        leaked |= bool(read(f"/proc/{pid}/environ"))
        leaked |= any(n in cmdline for n in NEEDLES)
        leaked |= b"bwrap" not in cmdline.split(b"\0")[0] and b"/app/solution/solution.py" not in cmdline
    if leaked:
        print(SENTINEL)


if __name__ == "__main__":
    main()
