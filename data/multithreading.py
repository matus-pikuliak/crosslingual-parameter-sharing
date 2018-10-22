import queue
import threading


class IteratorThread(threading.Thread):
    def __init__(self, ite, q):
        threading.Thread.__init__(self)
        self.ite = ite
        self.stop = threading.Event()
        self.q = q

    def run(self):
        while not self.stop.is_set():
            try:
                result = next(self.ite)
            except Exception as e:
                result = e
            self.q.put(result)

    def join(self, timeout=None):
        self.stop.set()
        self.q.get()
        threading.Thread.join(self)


def multithreading(iterator):

    def wrapped(self, *args, **kwargs):
        ite = iterator(self, *args, **kwargs)
        q = queue.Queue(1)
        thr = IteratorThread(ite, q)
        thr.start()

        while True:
            to_sent = q.get()
            if isinstance(to_sent, Exception):
                thr.join()
                raise to_sent
            yield to_sent

    return wrapped