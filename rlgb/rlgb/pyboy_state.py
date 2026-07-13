"""PyBoy save-state interop for rlgb — reads v0..v17, writes v13.

Implemented from the reverse-engineered format spec in
``docs/pyboy_state_format.md``. No PyBoy code was copied; this module only
speaks the byte format so RL pipelines can move between emulators.

DMG states only (rlgb is an original-Game-Boy core); CGB states are rejected.

CC BY-NC-ND 4.0 license. Built by BuiltNotTaught.
"""
from __future__ import annotations

import ctypes
import struct
import time

import numpy as np

LATEST_VERSION = 17
WRITE_VERSION = 13
FRAME_CYCLES = 70224

# header 0x149 -> number of 8 KiB cart-RAM banks PyBoy serializes.
# Note the minimum is one full bank: PyBoy dumps cart RAM unconditionally,
# even for carts that declare no RAM (incl. ROM-only and MBC2).
RAM_BANKS = {0x00: 1, 0x01: 1, 0x02: 1, 0x03: 4, 0x04: 16, 0x05: 8}
RTC_CARTS = {0x0F, 0x10}
MBC1_CARTS = {0x01, 0x02, 0x03}

TAC_PERIODS = (1024, 16, 64, 256)

# fixed sound-channel block sizes (v13+): sweep, tone, wave, noise
SOUND_CHANNELS = 77 + 57 + 63 + 65


class _Reader:
    def __init__(self, data: bytes):
        self.d = data
        self.o = 0

    def u8(self):
        v = self.d[self.o]
        self.o += 1
        return v

    def u16(self):
        v = int.from_bytes(self.d[self.o:self.o + 2], "little")
        self.o += 2
        return v

    def u32(self):
        v = int.from_bytes(self.d[self.o:self.o + 4], "little")
        self.o += 4
        return v

    def u64(self):
        v = int.from_bytes(self.d[self.o:self.o + 8], "little")
        self.o += 8
        return v

    def f32(self):
        v = struct.unpack_from("<f", self.d, self.o)[0]
        self.o += 4
        return v

    def f64(self):
        v = struct.unpack_from("<d", self.d, self.o)[0]
        self.o += 8
        return v

    def blob(self, n):
        v = self.d[self.o:self.o + n]
        self.o += n
        return v

    def skip(self, n):
        self.o += n

    def remaining(self):
        return len(self.d) - self.o


class _Writer:
    def __init__(self):
        self.b = bytearray()

    def u8(self, v):
        self.b.append(v & 0xFF)

    def u16(self, v):
        self.b += int(v & 0xFFFF).to_bytes(2, "little")

    def u32(self, v):
        self.b += int(v & 0xFFFFFFFF).to_bytes(4, "little")

    def u64(self, v):
        self.b += int(v & (2**64 - 1)).to_bytes(8, "little")

    def f64(self, v):
        self.b += struct.pack("<d", v)

    def blob(self, data):
        self.b += bytes(data)


def _skip_sound(r: _Reader, ver: int):
    if ver < 13:
        return
    if ver == 13:
        r.skip(16 + SOUND_CHANNELS)
        return
    r.skip(8)                                # audiobuffer_head
    samples_per_frame = r.u64()
    r.skip(8)                                # cycles_per_sample (f64)
    r.skip((samples_per_frame + 1) * 2)      # audio buffer
    r.skip(1 + 8 + 8 + 8 + 8 + 8 + 8 + 8 + 1 + 1 + 8)
    if ver >= 15:
        # NR50 byte: absent in v14, added during v15's lifetime (PyBoy
        # 2.7.0); states from PyBoy 2.6.x also lack it (spec §15.4).
        r.skip(1)
    r.skip(SOUND_CHANNELS)


def _rtc_regs_from_timezero(timezero: float, halt: int, day_carry: int):
    """PyBoy anchors its RTC to wall time; convert once to register values."""
    t = max(0.0, time.time() - timezero)
    days = int(t // 86400)
    dh = (days >> 8) & 1
    if halt:
        dh |= 0x40
    if day_carry:
        dh |= 0x80
    return bytes([int(t % 60), int(t // 60 % 60), int(t // 3600 % 24),
                  days & 0xFF, dh])


def load_pyboy_state(gb, data: bytes) -> int:
    """Restore a PyBoy .state file into a GameBoy. Returns the state version."""
    rom = open(gb.rom_path, "rb").read(0x150)
    cart_type, ram_code = rom[0x147], rom[0x149]

    r = _Reader(data)
    first = r.u8()
    if first >= 2:
        ver = first
        r.skip(1)                            # bootrom_enabled
    else:
        ver = 1                              # v0/1: that byte was the flag
    if ver > LATEST_VERSION:
        raise ValueError(f"PyBoy state version {ver} is newer than supported ({LATEST_VERSION})")
    if ver >= 16:
        r.skip(1)                            # key0
    if ver >= 8:
        r.skip(2)                            # key1, double_speed
        if r.u8():
            raise ValueError("CGB-mode PyBoy state; rlgb is a DMG (original Game Boy) core")

    # ---- CPU ----
    a, f_, b, c, d, e = (r.u8() for _ in range(6))
    hl, sp, pc = r.u16(), r.u16(), r.u16()
    ime, halted = r.u8(), r.u8()
    r.skip(1)                                # stopped
    ie = r.u8() if ver >= 5 else None
    iflag = None
    if ver >= 8:
        r.skip(1)                            # interrupt_queued
        iflag = r.u8()
    cycles = r.u64() if ver >= 12 else 0

    # ---- LCD ----
    vram = r.blob(0x2000)
    oam = r.blob(0xA0)
    lcdc, bgp, obp0, obp1 = (r.u8() for _ in range(4))
    stat = ly = lyc = 0
    if ver >= 5:
        stat, ly, lyc = r.u8(), r.u8(), r.u8()
    scy, scx, wy, wx = (r.u8() for _ in range(4))
    if ver >= 16:
        r.skip(1)                            # object_priority_mode
    if ver >= 11:
        r.skip(144 * 5)                      # scanline parameters
    lcd_clock = None
    if ver >= 8:
        r.skip(1)                            # cgb flag (already validated)
        if ver >= 17:
            r.skip(1)                        # downgraded_to_dmg
        r.skip(1)                            # speed_shift
        if ver >= 13:
            r.skip(3)                        # frame_done, first_frame, reset
        if ver >= 12:
            r.skip(8)                        # last_cycles
        lcd_clock = r.u64()
        r.skip(8 + 1)                        # clock_target, next_stat_mode

    _skip_sound(r, ver)

    # ---- Renderer ----
    if 2 <= ver < 11:
        r.skip(144 * (5 if ver > 3 else 4))  # old scanline parameters
    if ver >= 6:
        r.skip(144 * 160 * (5 if ver >= 10 else 4))

    # ---- RAM ----
    wram = r.blob(0x2000)
    r.skip(0x60)                             # FEA0-FEFF junk region
    io_ports = r.blob(0x4C)
    hram = r.blob(0x7F)

    if ver <= 15:
        r.skip(0x20 + 1 + 3 + 1 + 0x0F)      # mb tail incl. wram_select
    else:
        r.skip(1)                            # wram_select
    if ver < 5:
        ie = r.u8()

    # ---- Timer ----
    div = divc_low = tima = tma = tac = 0
    if ver >= 5:
        div, tima = r.u8(), r.u8()
        divc_low = r.u16() & 0xFF
        r.skip(2)                            # TIMA_counter
        tma, tac = r.u8(), r.u8()
        if ver >= 12:
            r.skip(8)                        # last_cycles
        if ver >= 13:
            r.skip(8)                        # _cycles_to_interrupt

    # ---- Cartridge ----
    rom_bank, ram_bank, ram_enable, memorymodel = (r.u8() for _ in range(4))
    cartram = r.blob(RAM_BANKS.get(ram_code, 1) * 0x2000)
    rtc_regs = None
    if cart_type in RTC_CARTS:
        timezero = r.f32() if ver <= 12 else r.f64()
        halt, day_carry = r.u8(), r.u8()
        rtc_regs = _rtc_regs_from_timezero(timezero, halt, day_carry)
    if cart_type in MBC1_CARTS and ver >= 3:
        rom_bank = r.u8()                    # bank_select_register1
        ram_bank = r.u8()                    # bank_select_register2

    # ---- Interaction / Serial ----
    buttons = 0
    if ver >= 7:
        directional, standard = r.u8(), r.u8()
        buttons = ((~standard & 0xF) | ((~directional & 0xF) << 4)) & 0xFF
    if ver >= 15:
        r.skip(36)                           # serial: 4 x u8 + 4 x u64

    if r.remaining():
        raise ValueError(
            f"{r.remaining()} bytes left after parsing a v{ver} state — "
            "state was probably saved with a different ROM, a CGB machine, "
            "or a v6-v12 build with sound enabled")
    if r.o > len(r.d):
        raise ValueError(f"v{ver} state is truncated ({len(r.d)} bytes)")

    # ================= apply to the machine =================
    lib, g = gb._lib, gb._g
    gb.reset()
    gb.vram[:] = np.frombuffer(vram, dtype=np.uint8)
    gb.oam[:] = np.frombuffer(oam, dtype=np.uint8)
    gb.wram[:] = np.frombuffer(wram, dtype=np.uint8)
    gb.hram[:] = np.frombuffer(hram, dtype=np.uint8)
    gb.io[:0x4C] = np.frombuffer(io_ports, dtype=np.uint8)
    gb.io[0x4C:] = 0

    io = gb.io           # raw writes: no io_write side effects, on purpose
    io[0x40], io[0x41] = lcdc, stat & 0x78
    io[0x42], io[0x43] = scy, scx
    io[0x44], io[0x45] = ly, lyc
    io[0x47], io[0x48], io[0x49] = bgp, obp0, obp1
    io[0x4A], io[0x4B] = wy, wx
    io[0x05], io[0x06], io[0x07] = tima, tma, tac

    regs = gb.registers
    regs.a, regs.f, regs.b, regs.c, regs.d, regs.e = a, f_, b, c, d, e
    regs.hl, regs.sp, regs.pc = hl, sp, pc
    regs.ime, regs.halted = ime, halted
    if ie is not None:
        gb.memory[0xFFFF] = ie

    if len(cartram):
        n = min(len(cartram), len(gb.cartram))
        gb.cartram[:n] = np.frombuffer(cartram[:n], dtype=np.uint8)

    lib.gb_set_timing(g, ((div << 8) | divc_low) & 0xFFFF)
    lib.gb_set_mbc_state(g, rom_bank, ram_bank, 1 if ram_enable else 0,
                         1 if memorymodel else 0)
    if rtc_regs is not None:
        lib.gb_set_rtc_state(g, rtc_regs, rtc_regs)
    lib.gb_set_buttons(g, buttons)
    if iflag is not None:
        # last: gb_set_buttons edge-triggers the joypad interrupt (IF bit 4),
        # but a state restore must reproduce IF exactly as saved
        io[0x0F] = iflag & 0x1F

    mode = stat & 3
    line_dot = 0
    if lcd_clock is not None and 0 <= lcd_clock - ly * 456 < 456:
        line_dot = int(lcd_clock - ly * 456)
    stat_line = bool(((ly == lyc) and (stat & 0x40)) or
                     (mode == 0 and stat & 0x08) or
                     (mode == 1 and stat & 0x10) or
                     (mode == 2 and stat & 0x20))
    lib.gb_set_ppu_state(g, mode, line_dot, 0, int(stat_line))
    if not cycles and lcd_clock:
        cycles = int(lcd_clock)
    lib.gb_set_counters(g, cycles, cycles // FRAME_CYCLES)
    return ver


def save_pyboy_state(gb) -> bytes:
    """Serialize a GameBoy as a PyBoy v13 state file."""
    rom = open(gb.rom_path, "rb").read(0x150)
    cart_type, ram_code = rom[0x147], rom[0x149]

    lib, g = gb._lib, gb._g
    io = gb.io
    regs = gb.registers
    cycles = gb.cycles
    ppu = lib.gb_get_ppu_state(g)
    mode, line_dot = ppu & 3, (ppu >> 8) & 0xFFFF
    mbc = lib.gb_get_mbc_state(g)
    rom_bank, ram_bank = mbc & 0xFFFF, (mbc >> 16) & 0xFF
    ram_enable, mbc1_mode = (mbc >> 24) & 1, (mbc >> 25) & 1
    divc = lib.gb_get_timing(g)
    ly, lyc = int(io[0x44]), int(io[0x45])
    stat_value = 0x80 | (int(io[0x41]) & 0x78) | (0x04 if ly == lyc else 0) | mode

    w = _Writer()
    w.u8(WRITE_VERSION)
    w.u8(0)                                  # bootrom_enabled
    w.u8(0)                                  # key1
    w.u8(0)                                  # double_speed
    w.u8(0)                                  # cgb

    # ---- CPU ----
    for v in (regs.a, regs.f, regs.b, regs.c, regs.d, regs.e):
        w.u8(v)
    w.u16(regs.hl)
    w.u16(regs.sp)
    w.u16(regs.pc)
    w.u8(regs.ime)
    w.u8(regs.halted)
    w.u8(0)                                  # stopped
    w.u8(gb.memory[0xFFFF])                  # IE
    w.u8(0)                                  # interrupt_queued
    w.u8(int(io[0x0F]) & 0x1F)               # IF
    w.u64(cycles)

    # ---- LCD ----
    w.blob(gb.vram.tobytes())
    w.blob(gb.oam.tobytes())
    for reg in (0x40, 0x47, 0x48, 0x49):
        w.u8(int(io[reg]))                   # LCDC, BGP, OBP0, OBP1
    w.u8(stat_value)
    w.u8(ly)
    w.u8(lyc)
    for reg in (0x42, 0x43, 0x4A, 0x4B):
        w.u8(int(io[reg]))                   # SCY, SCX, WY, WX
    tiledata = int(io[0x40]) & 0x10          # stored as LCDC bit 4 in place
    for _ in range(144):                     # scanline parameters
        w.u8(int(io[0x43]))                  # SCX
        w.u8(int(io[0x42]))                  # SCY
        w.u8((int(io[0x4B])) & 0xFF)         # stored as WX + 7 (raw WX reg)
        w.u8(int(io[0x4A]))                  # WY
        w.u8(tiledata)
    w.u8(0)                                  # cgb
    w.u8(0)                                  # speed_shift
    w.u8(1)                                  # frame_done
    w.u8(0)                                  # first_frame
    w.u8(0)                                  # reset
    w.u64(cycles)                            # last_cycles
    # PyBoy's LCD state machine: next_stat_mode is the mode entered when
    # clock reaches clock_target; LY increments on entering mode 2/1, and
    # the frame resets when the target is hit with LY == 153.
    clock = ly * 456 + line_dot
    if mode == 2:
        target, nxt = ly * 456 + 80, 3
    elif mode == 3:
        target, nxt = ly * 456 + 250, 0          # mode 3 spans 170 dots
    elif mode == 0:
        target, nxt = (ly + 1) * 456, (1 if ly >= 143 else 2)
    else:                                        # vblank: one line at a time
        target, nxt = (ly + 1) * 456, 1
    w.u64(clock)
    w.u64(target)
    w.u8(nxt)                                # next_stat_mode

    # ---- Sound (v13 fixed layout: real registers, idle timers) ----
    w.u64(cycles)
    w.u64(cycles)
    _write_sound_channels(w, io)

    # ---- Renderer: 144x160 x (u32 RGBA + attr u8) ----
    gray = gb.screen_gray
    px = np.zeros((144, 160, 5), dtype=np.uint8)
    px[..., 0] = px[..., 1] = px[..., 2] = gray
    px[..., 3] = 0xFF
    w.blob(px.tobytes())

    # ---- RAM ----
    w.blob(gb.wram.tobytes())
    w.blob(bytes(0x60))
    w.blob(gb.io[:0x4C].tobytes())
    w.blob(gb.hram.tobytes())
    w.blob(bytes(0x20))                      # mb tail (v<=15)
    w.u8(0)                                  # object_priority_mode
    w.blob(bytes(3))
    w.u8(0)                                  # wram_select
    w.blob(bytes(0x0F))

    # ---- Timer ----
    period = TAC_PERIODS[int(io[0x07]) & 3]
    w.u8(divc >> 8)                          # DIV
    w.u8(int(io[0x05]))                      # TIMA
    w.u16(divc & 0xFF)                       # DIV_counter
    w.u16(divc % period)                     # TIMA_counter
    w.u8(int(io[0x06]))                      # TMA
    w.u8(int(io[0x07]))                      # TAC
    w.u64(cycles)                            # last_cycles
    w.u64(period)                            # _cycles_to_interrupt

    # ---- Cartridge ----
    if rom_bank > 0xFF:
        raise ValueError("ROM bank > 255 cannot be represented in a PyBoy state")
    if cart_type in MBC1_CARTS:
        # PyBoy uses this byte directly as the active bank: store the
        # resolved bank (bank2<<5 | bank1, low 5 bits of 0 read as 1).
        resolved = ((ram_bank & 0x03) << 5) | (rom_bank & 0x1F)
        if resolved & 0x1F == 0:
            resolved |= 1
        w.u8(resolved)
    else:
        w.u8(rom_bank)
    w.u8(ram_bank)
    w.u8(ram_enable)
    w.u8(mbc1_mode)
    ram_banks = RAM_BANKS.get(ram_code, 1)   # PyBoy always dumps >= 1 bank
    buf = bytes(gb.cartram.tobytes()[:ram_banks * 0x2000])
    w.blob(buf + bytes(ram_banks * 0x2000 - len(buf)))
    if cart_type in RTC_CARTS:
        tmp = ctypes.create_string_buffer(10)
        lib.gb_get_rtc_state(g, tmp)
        s_, m_, h_, dl_, dh_ = tmp.raw[:5]
        elapsed = (((dh_ & 1) << 8 | dl_) * 86400 + h_ * 3600 + m_ * 60 + s_)
        w.f64(time.time() - elapsed)
        w.u8(1 if (dh_ & 0x40) else 0)       # halt
        w.u8(1 if (dh_ & 0x80) else 0)       # day_carry
    if cart_type in MBC1_CARTS:
        w.u8(rom_bank & 0x1F)                # bank_select_register1
        w.u8(ram_bank & 0x03)                # bank_select_register2

    # ---- Interaction ----
    buttons = lib.gb_get_buttons(g)
    w.u8((~(buttons >> 4)) & 0xF)            # directional
    w.u8((~buttons) & 0xF)                   # standard
    return bytes(w.b)


NOISE_DIVISORS = (8, 16, 32, 48, 64, 80, 96, 112)


def _write_sound_channels(w: _Writer, io):
    """APU channel blocks: register fields decomposed from the NRxx backing
    bytes, run-time timers idle, channels muted until the game retriggers.

    The period fields must be the hardware-consistent non-zero values —
    PyBoy advances channel phase with ``timer += period`` catch-up loops,
    so a zero period would spin its tick forever.
    """
    def tone(nrx1, nrx2, nrx3, nrx4):
        sound_period = ((nrx4 & 7) << 8) | nrx3
        period = 4 * (0x800 - sound_period)
        w.u8(nrx1 >> 6)                      # wave duty
        w.u8(nrx1 & 0x3F)                    # initial length timer
        w.u8(nrx2 >> 4)                      # envelope volume
        w.u8((nrx2 >> 3) & 1)                # envelope direction
        w.u8(nrx2 & 7)                       # envelope pace
        w.u16(sound_period)
        w.u8((nrx4 >> 6) & 1)                # length enable
        w.u8(0)                              # enable (idle until retrigger)
        w.u64(0)                             # length timer
        w.u64(0)                             # envelope timer
        w.u64(period)                        # period timer
        w.u64(period)
        w.u64(0)                             # wave frame
        w.u64(nrx2 >> 4)                     # volume
        return sound_period

    # CH1 (sweep) = tone block + sweep tail
    shadow = tone(int(io[0x11]), int(io[0x12]), int(io[0x13]), int(io[0x14]))
    nr10 = int(io[0x10])
    w.u8((nr10 >> 4) & 7)                    # sweep pace
    w.u8((nr10 >> 3) & 1)                    # sweep direction
    w.u8(nr10 & 7)                           # sweep magnitude
    w.u64(0)                                 # sweep timer
    w.u8(0)                                  # sweep enable
    w.u64(shadow)                            # shadow period

    # CH2 (tone)
    tone(int(io[0x16]), int(io[0x17]), int(io[0x18]), int(io[0x19]))

    # CH3 (wave)
    for a in range(0x30, 0x40):
        w.u8(int(io[a]))                     # wave table FF30-FF3F
    sound_period = ((int(io[0x1E]) & 7) << 8) | int(io[0x1D])
    period = 2 * (0x800 - sound_period)
    w.u8((int(io[0x1A]) >> 7) & 1)           # DAC power
    w.u8(int(io[0x1B]))                      # initial length timer
    w.u8((int(io[0x1C]) >> 5) & 3)           # output level
    w.u16(sound_period)
    w.u8((int(io[0x1E]) >> 6) & 1)           # length enable
    w.u8(0)                                  # enable
    w.u64(0)                                 # length timer
    w.u64(period)                            # period timer
    w.u64(period)
    w.u64(0)                                 # wave frame
    w.u64(0)                                 # volume shift

    # CH4 (noise)
    nr42, nr43 = int(io[0x21]), int(io[0x22])
    period = NOISE_DIVISORS[nr43 & 7] << (nr43 >> 4)
    w.u8(int(io[0x20]) & 0x3F)               # initial length timer
    w.u8(nr42 >> 4)                          # envelope volume
    w.u8((nr42 >> 3) & 1)                    # envelope direction
    w.u8(nr42 & 7)                           # envelope pace
    w.u8(nr43 >> 4)                          # clock shift
    w.u8((nr43 >> 3) & 1)                    # LFSR width
    w.u8(nr43 & 7)                           # clock divider
    w.u8((int(io[0x23]) >> 6) & 1)           # length enable
    w.u8(0)                                  # enable
    w.u64(0)                                 # length timer
    w.u64(period)                            # period timer
    w.u64(0)                                 # envelope timer
    w.u64(period)
    w.u64(0x7FFF)                            # LFSR shift register
    w.u64(0)                                 # LFSR feed
    w.u64(nr42 >> 4)                         # volume
