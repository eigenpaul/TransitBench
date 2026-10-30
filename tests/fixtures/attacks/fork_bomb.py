"""Attack: fork until refused. Sentinel if more than 64 processes start (the per-slot process cap). Bounded at 300
so that a broken jail cannot take the host down."""

import os
import sys
import time

SENTINEL = "424242,42,42"


def main():
    sys.stdin.buffer.read()
    children = 0
    for _ in range(300):
        try:
            pid = os.fork()
        except OSError:
            break
        if pid == 0:
            time.sleep(20)
            os._exit(0)
        children += 1
    if children > 64:
        print(SENTINEL, flush=True)
    time.sleep(0.5)


if __name__ == "__main__":
    main()
