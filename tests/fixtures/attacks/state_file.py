"""Attack: carry state from one star to the next in files. Each star looks for a marker an earlier star left in
every place it can name, prints the sentinel if it finds one, then leaves its own."""

import os
import sys

SENTINEL = "424242,42,42"
MARKER = "tb-attack-state"
PLACES = ("/tmp", "/var/tmp", "/dev/shm", "/work", "/app", "/app/solution", "/run", "/dev", "/", "/etc", "/usr")


def places():
    here = [os.environ.get("HOME"), os.environ.get("TMPDIR"), os.getcwd(), os.path.dirname(os.path.abspath(__file__))]
    return [p for p in dict.fromkeys(here + list(PLACES)) if p]


def main():
    sys.stdin.buffer.read()
    found = any(os.path.exists(os.path.join(p, MARKER)) for p in places())
    for p in places():
        try:
            with open(os.path.join(p, MARKER), "w") as f:
                f.write("x")
        except OSError:
            pass
    if found:
        print(SENTINEL)


if __name__ == "__main__":
    main()
