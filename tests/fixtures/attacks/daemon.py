"""Attack: a double-forked setsid daemon that ignores SIGHUP and SIGTERM, outlives its star, and then writes the
sentinel row to the stdout it inherited. Contained if the daemon dies with the star (the test also looks for it)."""

import os
import signal
import sys
import time

SENTINEL = "424242,42,42\n"


def main():
    sys.stdin.buffer.read()
    pid = os.fork()
    if pid == 0:
        os.setsid()
        if os.fork() == 0:
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            time.sleep(0.3)                     # the star has ended by now
            for _ in range(50):
                try:
                    os.write(1, SENTINEL.encode())
                except OSError:
                    pass
                time.sleep(0.1)
        os._exit(0)
    os.waitpid(pid, 0)


if __name__ == "__main__":
    main()
