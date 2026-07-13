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

# Profile-guided build: ~20% faster than plain -O3 (profiles bench.py's
# workload). Result is machine- and profile-specific, like -march=native.
PGODIR ?= /tmp/rlgb-pgo
pgo:
	$(CC) $(CFLAGS) -fprofile-generate -fprofile-dir=$(PGODIR) $(LDFLAGS) -o $(LIB) $(SRC)
	.venv/bin/python bench.py >/dev/null
	$(CC) $(CFLAGS) -fprofile-use -fprofile-dir=$(PGODIR) -fprofile-correction $(LDFLAGS) -o $(LIB) $(SRC)

clean:
	rm -f $(LIB)

smoke: $(LIB)
	.venv/bin/python tests/smoke.py

test: $(LIB)
	.venv/bin/python -m pytest tests/ -q

.PHONY: all clean smoke test pgo
