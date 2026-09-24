# C++ lock-handover + sharding demos (phase 3)
CXX ?= g++
FLAGS := -O2 -std=c++17 -pthread -Wall -Wextra
.PHONY: all run clean
all: locks sharded
locks: tas_vs_ticket_vs_mcs.cpp
	$(CXX) $(FLAGS) $< -o $@
sharded: sharded_counter.cpp
	$(CXX) $(FLAGS) $< -o $@
run: all
	./locks 8 200000
	./sharded 8 2000000
clean:
	rm -f locks sharded
