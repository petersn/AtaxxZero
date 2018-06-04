
CXXFLAGS=-std=c++14 -g -Wall -Wextra -pthread
#CXXFLAGS+=-O3 -ffast-math -flto
LIBS=-lboost_system -lrpc

all: client

client: client.o bitboards.o move.o movegen.o other.o makemove.o invalid.o ataxx.o Makefile
	$(CXX) $(CXXFLAGS) -o $@ client.o bitboards.o move.o movegen.o other.o makemove.o invalid.o ataxx.o $(LIBS)

.PHONY: clean
clean:
	rm -f *.o client

