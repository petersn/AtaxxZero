#!/usr/bin/python

import numpy as np
import ctypes

dll = ctypes.CDLL("./cpp/self_play_client.so")

launch_threads = dll.launch_threads
launch_threads.restype = None
launch_threads.argtypes = [
	ctypes.c_char_p, # char* output_path
	ctypes.c_int,    # int visits
	ctypes.c_void_p, # float* fill_buffer1
	ctypes.c_void_p, # float* fill_buffer2
	ctypes.c_int,    # int buffer_entries
	ctypes.c_int,    # int thread_count
]

get_workload = dll.get_workload
get_workload.restype = ctypes.c_int
get_workload.argtypes = []

complete_workload = dll.complete_workload
complete_workload.restype = None
complete_workload.argtypes = [
	ctypes.c_void_p, # float* posteriors
	ctypes.c_void_p, # float* values
]

shutdown = dll.shutdown
shutdown.restype = None
shutdown.argtypes = []

if __name__ == "__main__":
	buffer_entries = 32
	arrays = [
		np.zeros((buffer_entries, 7, 7, 4), dtype=np.float32)
		for _ in (0, 1)
	]
	launch_threads(
		"/dev/null",
		ctypes.c_void_p(arrays[0].ctypes.data),
		ctypes.c_void_p(arrays[1].ctypes.data),
		buffer_entries,
		buffer_entries * 2,
	)
	while True:
		i = get_workload()
#		print "Workload in buffer:", i
		posteriors = np.zeros((buffer_entries, 7, 7, 17), dtype=np.float32)
		values = np.zeros((buffer_entries, 1), dtype=np.float32)
		complete_workload(i, ctypes.c_void_p(posteriors.ctypes.data), ctypes.c_void_p(values.ctypes.data))

