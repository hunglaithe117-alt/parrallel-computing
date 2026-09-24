# C++ sync-learn demos (Gate 1: learn-through-examples)
CXX ?= g++
FLAGS := -O2 -std=c++17 -Wall -Wextra
DEMOS := storebuffer-visibility invqueue-loadload compiler-barrier-merged
.PHONY: all clean $(DEMOS)
all: $(DEMOS)
$(DEMOS):
	$(CXX) $(FLAGS) $@/demo.cpp -o /tmp/cpp-$@-fixed
	$(CXX) $(FLAGS) -DBROKEN $@/demo.cpp -o /tmp/cpp-$@-broken
	@echo "== $@ fixed =="; /tmp/cpp-$@-fixed
	@echo "== $@ broken (negative control) =="; /tmp/cpp-$@-broken
clean:
	rm -f /tmp/cpp-*-fixed /tmp/cpp-*-broken
