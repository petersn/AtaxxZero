
CXXFLAGS=-std=c++17 -g -Wall -Wextra -pthread -fPIC -I/usr/local/cuda/include -L/usr/local/cuda/lib -L/usr/local/cuda/lib64
CXXFLAGS+=-O3 -ffast-math -flto
LIBS=-lcublas -lcudnn -lcudart

all: self_play_client.so engine_main #fast_server self_play_client

self_play_client: self_play_client.o fast_eval.o bitboards.o move.o movegen.o other.o makemove.o invalid.o ataxx.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

self_play_client.so: self_play_client.o fast_eval.o bitboards.o move.o movegen.o other.o makemove.o invalid.o ataxx.o
	$(CXX) $(CXXFLAGS) -shared -Wl,-soname,$@ -o $@ $^ $(LIBS)

fast_server: fast_server.o
	$(CXX) $(CXXFLAGS) -o $@ fast_server.o $(LIBS)

engine_main: engine_main.o fast_eval.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

.PHONY: clean
clean:
	rm -f *.o self_play_client self_play_client.so fast_server engine_main

