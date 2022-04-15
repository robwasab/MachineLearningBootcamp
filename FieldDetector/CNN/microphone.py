import pyaudio
import sys

from time import sleep
from queue import Queue
import numpy as np

from scipy.signal import decimate



def list_microphones(pa):
    count = pa.get_device_count()
    print('device count: ', count)
    for k in range(count):
        info = pa.get_device_info_by_index(k)
        if info['hostApi'] == 0 and info['maxInputChannels'] == 1:
            device_name = info['name']
            device_index = info['index']
            print(f'[{device_index}]: "{device_name}"')


def get_info_by_name(pa, name):
    count = pa.get_device_count()
    for k in range(count):
        info = pa.get_device_info_by_index(k)
        if info['hostApi'] == 0 \
            and info['maxInputChannels'] == 1 \
            and info['name'] == name:
            return info
    return None


def init_stream_by_name(
    pa, name, chunk=1024, rate=44100, stream_callback=None, start=True):

    info = get_info_by_name(pa, name)

    if None == info:
        print(f'Could not init {name}')
        print('List of available devices:')
        list_devices(pa)
        sys.exit(-1)

    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=rate,
        input=True,
        input_device_index=info['index'],
        frames_per_buffer=chunk,
        stream_callback=stream_callback,
        start=start
    )

    return stream


callback_flags2str = {
    1: 'InputUnderflow',
    2: 'InputOverflow',
    4: 'OutputUnderflow',
    8: 'OutputOverflow',
    16: 'PrimingOutput',
}


class Runloop(object):
    def __init__(self, debug=False):
        self.cmd_queue = Queue()
        self.debug = debug

    def print(self, *args):
        print(f'[{self.position}]', *args)

    def enqueue_task(self, task, *args):
        self.cmd_queue.put((task, args))


    def main_loop_task(self):
        if not self.cmd_queue.empty():
            if self.debug:
                print('Runloop queue size: ', self.cmd_queue.qsize())
            cmd, arg = self.cmd_queue.get()
            cmd(*arg)


class ContinuousMicrophone(Runloop):
    def __init__(self, pa, name, fs, downsample_factor, label, done_callback, chunk=2*4096):
        super(ContinuousMicrophone, self).__init__()

        self.stream = init_stream_by_name(
            pa,
            name,
            chunk=chunk,
            stream_callback=self.__stream_callback,
            start=False,
            rate=fs
        )

        self.state_record = False
        self.label = label
        self.done_callback = done_callback
        self.downsample_factor = downsample_factor


    def start_recording(self):
        self.enqueue_task(self.__task_start_recording)


    def print(self, *args):
        print(f'[{self.label}]', *args)


    def __task_start_recording(self, *args):
        if False == self.state_record:
            self.state_record = True
            self.print('starting to record: ', self.label)
            self.stream.start_stream()


    def __task_downsample(self, audio_data):
        segmented_data = np.frombuffer(audio_data, dtype=np.int16) / 32768.0

        if 1 < self.downsample_factor:
            decimated_data = decimate(segmented_data, self.downsample_factor)
            self.done_callback(self.label, decimated_data)
        else:
            self.done_callback(self.label, segmented_data)


    def __stream_callback(self, in_data, frame_count, time_info, status_flags):
        if status_flags > 0:
            for k in range(0, 4):
                if (1 << k) & status_flags:
                    print(callback_flags2str[(1 << k)])

        self.enqueue_task(self.__task_downsample, in_data)

        return (None, pyaudio.paContinue)
