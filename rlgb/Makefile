# rlgb — CC BY-NC-ND 4.0 license. Built by BuiltNotTaught.
CC      ?= cc
ARCH    ?= -march=native
CFLAGS  ?= -O3 $(ARCH) -flto -fomit-frame-pointer -Wall -Wextra -std=c11 -fPIC
LDFLAGS ?= -shared -flto

SRC = src/gb.c src/ppu.c
LIB = rlgb/libgb.so

all: $(LIB)

$(LIB): $(SRC) src/gb.h src/cycles.inc
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(SRC)

clean:
	rm -f $(LIB)

.PHONY: all clean
