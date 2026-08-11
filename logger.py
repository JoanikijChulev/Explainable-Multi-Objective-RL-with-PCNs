import h5py
from pathlib import Path
import numpy as np
import torch
import cv2
import time
import threading
import atexit
from artifacts import output_artifacts_dir


def resize_image(frame, max_width=100):
    w, h = frame.shape[:2]
    r = h/w
    w = np.minimum(w, max_width)
    h = int(r*w)
    frame = cv2.resize(frame, (h, w))
    return frame


class Logger(object):

    def __init__(self, logdir='runs/', append=False):
        self.types = {}
        self.to_log = {}
        self._closed = False
        self._stop_event = threading.Event()
        if logdir is not None:
            self.logdir = Path(logdir)
            self.logdir.mkdir(parents=True, exist_ok=True)
            h5py.File(self.logdir / 'log.h5', 'a' if append else 'w').close()
        else:
            self.logdir = None
        # writing to file will be done periodically, in a separate thread
        self._write_thread = threading.Thread(target=self.wait_and_write, daemon=True)
        # lock to ensure data won't be tampered with when writing
        self._lock = threading.Lock()
        # at exist, force leftover data to be written
        atexit.register(self.close)
        self._write_thread.start()

    def wait_and_write(self, wait=30):
        while not self._stop_event.wait(wait):
            with self._lock:
                if self._closed:
                    return
                self.flush()

    def flush(self):
        if self._closed:
            return
        if self.logdir is None:
            for tag in self.types:
                self.to_log[tag] = []
            return
        with h5py.File(self.logdir / 'log.h5', 'r+') as f:
            for tag, type_ in self.types.items():
                pending = self.to_log.get(tag, [])
                if not pending:
                    self.to_log.setdefault(tag, [])
                    continue
                if type_ == 'scalar':
                    self.log_scalar(tag, f)
                else:
                    self.log_ndarray(tag, f)

    def put(self, tag, value, step, type_):
        if type_ == 'image':
            value = resize_image(value)
        with self._lock:
            if not tag in self.to_log:
                self.types[tag] = type_
                self.to_log[tag] = []
            self.to_log[tag].append((step, value))

    def log_scalar(self, tag, log_file):
        toadd = self.to_log.pop(tag, [])
        if not toadd:
            self.to_log[tag] = []
            return
        toadd = np.array(toadd)
        self.to_log[tag] = []

        if not tag in log_file:
            log_file.create_dataset(tag, toadd.shape, maxshape=(None, 2), dtype=np.float32)
            log_file[tag].attrs['type'] = self.types[tag]
        else:
            log_file[tag].resize(log_file[tag].len()+len(toadd), 0)

        log_file[tag][-len(toadd):] = np.array(toadd)

    def log_ndarray(self, tag, log_file):
        toadd = self.to_log.pop(tag, [])
        if not toadd:
            self.to_log[tag] = []
            return
        steps, ndarray = zip(*toadd)
        steps, ndarray = np.array(steps), np.stack(ndarray, axis=0)
        self.to_log[tag] = []

        if not (tag+'/ndarray') in log_file:
            log_file.create_dataset(tag + '/step', steps.shape, maxshape=(None,), dtype=np.int32)
            log_file.create_dataset(tag + '/ndarray', ndarray.shape, maxshape=(None,)+tuple(ndarray.shape[1:]), dtype=ndarray.dtype)
            log_file[tag].attrs['type'] = self.types[tag]
        else:
            for t in [tag+'/ndarray', tag+'/step']:
                log_file[t].resize(log_file[t].len()+len(steps), 0)
        log_file[tag+'/step'][-len(steps):] = steps
        log_file[tag+'/ndarray'][-len(ndarray):] = ndarray

    def close(self):
        if self._closed:
            return
        self._stop_event.set()
        if self._write_thread.is_alive() and threading.current_thread() is not self._write_thread:
            self._write_thread.join()
        with self._lock:
            if self._closed:
                return
            self.flush()
            self._closed = True



if __name__ == '__main__':
    import gymnasium as gym
    import matplotlib.pyplot as plt

    l = Logger(output_artifacts_dir() / 'logger_demo')

    env = gym.make('CartPole-v1', render_mode='rgb_array')
    env.reset(); rew = 0; step = 0; done = False
    while step < 100:
        _, r, terminated, truncated, _ = env.step(env.action_space.sample())
        done = terminated or truncated
        rew += r
        l.put('reward', rew, step, 'scalar')
        l.put('frame', env.render(), step, 'image')
        # l.log_ndarray('frame')
        step += 1

    # l.log_scalar('reward')
    # l.log_image('frame')

    time.sleep(30)

    log = h5py.File(l.logdir / 'log.h5', 'r')

    plt.figure()
    plt.plot(log['reward'][:,0], log['reward'][:,1])
    plt.show()

    for frame in log['frame/ndarray'][-5:]:
        plt.figure()
        plt.imshow(frame)
        plt.show()

    log.close()
