
CXXFLAGS=-std=c++14 -g -Wall -Wextra -pthread -fPIC
CXXFLAGS+=-O3 -ffast-math -flto
LIBS=#-lrpc

all: self_play_client.so #fast_server self_play_client

self_play_client: self_play_client.o bitboards.o move.o movegen.o other.o makemove.o invalid.o ataxx.o Makefile
	$(CXX) $(CXXFLAGS) -o $@ self_play_client.o bitboards.o move.o movegen.o other.o makemove.o invalid.o ataxx.o $(LIBS)

self_play_client.so: self_play_client.o bitboards.o move.o movegen.o other.o makemove.o invalid.o ataxx.o Makefile
	$(CXX) $(CXXFLAGS) -shared -Wl,-soname,$@ -o $@ self_play_client.o bitboards.o move.o movegen.o other.o makemove.o invalid.o ataxx.o $(LIBS)

fast_server: fast_server.o Makefile
	$(CXX) $(CXXFLAGS) -o $@ fast_server.o $(LIBS)

.PHONY: clean
clean:
	rm -f *.o self_play_client self_play_client.so fast_server

