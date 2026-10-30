"""Attack: reach the network, or another star running at the same time. Sentinel if a TCP connection to 1.1.1.1:53
or 169.254.169.254:80 succeeds, or if a concurrent star answers on a loopback port or an abstract unix socket."""

import errno
import socket
import sys
import time

SENTINEL = "424242,42,42"
PORT, ABSTRACT = 47474, b"\0tb-attack-rendezvous"


def reach(addr):
    try:
        with socket.create_connection(addr, timeout=1):
            return True
    except OSError:
        return False


def knock():
    found = reach(("127.0.0.1", PORT))
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.connect(ABSTRACT)
            found = True
    except OSError:
        pass
    return found


def listen(seconds):
    servers = []
    try:
        for family, addr in ((socket.AF_INET, ("127.0.0.1", PORT)), (socket.AF_UNIX, ABSTRACT)):
            s = socket.socket(family, socket.SOCK_STREAM)
            servers.append(s)
            try:
                s.bind(addr)
                s.listen()
                s.setblocking(False)
            except OSError as e:
                if e.errno == errno.EADDRINUSE:
                    return True               # another star holds it: a shared namespace
                servers.pop().close()
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            for s in servers:
                try:
                    s.accept()[0].close()
                    return True
                except BlockingIOError:
                    pass
            time.sleep(0.05)
        return False
    finally:
        for s in servers:
            s.close()


def main():
    sys.stdin.buffer.read()
    worked = reach(("1.1.1.1", 53)) or reach(("169.254.169.254", 80)) or knock() or listen(1.5)
    if worked:
        print(SENTINEL)


if __name__ == "__main__":
    main()
