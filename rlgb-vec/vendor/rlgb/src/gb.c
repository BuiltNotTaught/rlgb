/*
 * rlgb core: bus, cartridge mappers, timer, SM83 CPU, save states, C API.
 * Written from public DMG hardware documentation (Pan Docs). CC BY-NC-ND 4.0 license.
 * Built by BuiltNotTaught.
 */
#include <stdlib.h>
#include <string.h>
#include "gb.h"

#include "cycles.inc"

/* flag bits in F */
enum { FZ = 0x80, FN = 0x40, FH = 0x20, FC = 0x10 };

/* IF/IE interrupt bits */
enum { I_VBLANK = 1, I_STAT = 2, I_TIMER = 4, I_SERIAL = 8, I_JOYPAD = 16 };

/* io[] indices we touch often */
enum {
    R_P1 = 0x00, R_SB = 0x01, R_SC = 0x02, R_DIV = 0x04, R_TIMA = 0x05,
    R_TMA = 0x06, R_TAC = 0x07, R_IF = 0x0F,
    R_LCDC = 0x40, R_STAT = 0x41, R_SCY = 0x42, R_SCX = 0x43, R_LY = 0x44,
    R_LYC = 0x45, R_DMA = 0x46, R_BGP = 0x47, R_OBP0 = 0x48, R_OBP1 = 0x49,
    R_WY = 0x4A, R_WX = 0x4B,
};

GB_INTERNAL void ppu_tick(GB *g, uint32_t cycles);       /* ppu.c */
GB_INTERNAL void ppu_lcd_write(GB *g, uint8_t reg, uint8_t v);
GB_INTERNAL uint8_t ppu_stat_read(GB *g);

/* ------------------------------------------------------------------ */
/* Cartridge mappers                                                    */
/* ------------------------------------------------------------------ */

/* Rebuild the bus page table from the resolved offsets. Directly-mapped
 * regions (ROM banks, VRAM, WRAM, echo RAM) get a base pointer such that a
 * byte is base[addr & 0xFFF]; regions needing logic (cart RAM, MMIO/OAM/HRAM
 * at 0xF000) stay NULL so rd8/wr8 fall through to the slow path. */
static void rebuild_maps(GB *g)
{
    GBState *s = &g->s;
    const uint8_t *rom = g->rom;

    for (int p = 0; p < 4; p++)               /* 0000-3FFF: ROM bank 0 window */
        g->read_map[p] = rom ? rom + s->rom_off0 + (uint32_t)p * 0x1000 : NULL;
    for (int p = 0; p < 4; p++)               /* 4000-7FFF: ROM bank 1 window */
        g->read_map[4 + p] = rom ? rom + s->rom_off1 + (uint32_t)p * 0x1000 : NULL;
    g->read_map[0x8] = s->vram;               /* 8000-8FFF */
    g->read_map[0x9] = s->vram + 0x1000;      /* 9000-9FFF */
    g->read_map[0xA] = NULL;                  /* A000-BFFF cart RAM: slow */
    g->read_map[0xB] = NULL;
    g->read_map[0xC] = s->wram;               /* C000-CFFF */
    g->read_map[0xD] = s->wram + 0x1000;      /* D000-DFFF */
    g->read_map[0xE] = s->wram;               /* E000-EFFF echo */
    g->read_map[0xF] = NULL;                  /* F000-FFFF echo/OAM/MMIO: slow */

    for (int p = 0; p < 8; p++) g->write_map[p] = NULL;   /* ROM: MBC registers */
    g->write_map[0x8] = s->vram;
    g->write_map[0x9] = s->vram + 0x1000;
    g->write_map[0xA] = NULL;
    g->write_map[0xB] = NULL;
    g->write_map[0xC] = s->wram;
    g->write_map[0xD] = s->wram + 0x1000;
    g->write_map[0xE] = s->wram;
    g->write_map[0xF] = NULL;
}

static void mbc_remap(GB *g)
{
    GBState *s = &g->s;
    uint32_t bank1 = s->rom_bank, bank0 = 0;

    switch (s->mbc_type) {
    case 1: {
        uint32_t lo = s->rom_bank & 0x1F;
        if (lo == 0) lo = 1;
        uint32_t hi = s->ram_bank & 0x03;
        bank1 = (hi << 5) | lo;
        if (s->mbc1_mode) {
            bank0 = (hi << 5);
            s->ram_off = (s->ram_size > 0x2000) ? (hi * 0x2000u) : 0;
        } else {
            bank0 = 0;
            s->ram_off = 0;
        }
        break;
    }
    case 2:
        bank1 = s->rom_bank & 0x0F;
        if (bank1 == 0) bank1 = 1;
        s->ram_off = 0;
        break;
    case 3:
        bank1 = s->rom_bank & 0x7F;
        if (bank1 == 0) bank1 = 1;
        s->ram_off = (s->ram_bank & 0x03) * 0x2000u;
        break;
    case 5:
        bank1 = s->rom_bank & 0x1FF;   /* bank 0 is legal on MBC5 */
        s->ram_off = (s->ram_bank & 0x0F) * 0x2000u;
        break;
    default:
        bank1 = 1;
        s->ram_off = 0;
        break;
    }
    /* Bank offsets are clamped against the ACTUAL ROM buffer size, never a
     * state-supplied bank count: a crafted save state could otherwise set
     * rom_banks=0 (skipping the clamp) and drive rom_off past the buffer,
     * making every 0x4000-0x7FFF read out-of-bounds. Same for cart RAM. */
    uint32_t banks = g->rom_size / 0x4000u;
    if (banks == 0) banks = 1;
    s->rom_banks = banks;
    bank0 %= banks;
    bank1 %= banks;
    s->rom_off0 = bank0 * 0x4000u;
    s->rom_off1 = bank1 * 0x4000u;

    if (s->ram_size > GB_CARTRAM_MAX) s->ram_size = GB_CARTRAM_MAX;
    if (s->ram_size && s->ram_off >= s->ram_size)
        s->ram_off %= s->ram_size;
    if (s->ram_off + 0x2000u > GB_CARTRAM_MAX)
        s->ram_off = 0;

    rebuild_maps(g);
}

static void mbc_write(GB *g, uint16_t a, uint8_t v)
{
    GBState *s = &g->s;
    switch (s->mbc_type) {
    case 0:
        return;
    case 1:
        switch (a >> 13) {
        case 0: s->ram_enable = ((v & 0x0F) == 0x0A); break;
        case 1: s->rom_bank = v & 0x1F; break;
        case 2: s->ram_bank = v & 0x03; break;
        case 3: s->mbc1_mode = v & 1; break;
        }
        break;
    case 2:
        if (a < 0x4000) {
            if (a & 0x0100) s->rom_bank = v & 0x0F;
            else            s->ram_enable = ((v & 0x0F) == 0x0A);
        }
        break;
    case 3:
        switch (a >> 13) {
        case 0: s->ram_enable = ((v & 0x0F) == 0x0A); break;
        case 1: s->rom_bank = v & 0x7F; break;
        case 2: s->ram_bank = v; break;             /* 0-3 RAM, 8-C RTC */
        case 3:                                     /* RTC latch on 0->1 */
            if (s->rtc_latch_prev == 0 && v == 1)
                memcpy(s->rtc_latched, s->rtc, 5);
            s->rtc_latch_prev = v;
            break;
        }
        break;
    case 5:
        switch (a >> 12) {
        case 0: case 1: s->ram_enable = ((v & 0x0F) == 0x0A); break;
        case 2: s->rom_bank = (s->rom_bank & 0x100) | v; break;
        case 3: s->rom_bank = (s->rom_bank & 0x0FF) | ((v & 1) << 8); break;
        case 4: case 5: s->ram_bank = v & 0x0F; break;
        }
        break;
    }
    mbc_remap(g);
}

static uint8_t cartram_read(GB *g, uint16_t a)
{
    GBState *s = &g->s;
    if (!s->ram_enable) return 0xFF;
    if (s->mbc_type == 3 && s->ram_bank >= 0x08) {
        uint8_t r = s->ram_bank - 0x08;
        return (r < 5) ? s->rtc_latched[r] : 0xFF;
    }
    if (s->mbc_type == 2)
        return 0xF0 | s->cartram[(a - 0xA000) & 0x1FF];
    if (!s->ram_size) return 0xFF;
    return s->cartram[(s->ram_off + (a - 0xA000)) % s->ram_size];
}

static void cartram_write(GB *g, uint16_t a, uint8_t v)
{
    GBState *s = &g->s;
    if (!s->ram_enable) return;
    if (s->mbc_type == 3 && s->ram_bank >= 0x08) {
        uint8_t r = s->ram_bank - 0x08;
        if (r < 5) { s->rtc[r] = v; s->rtc_latched[r] = v; }
        return;
    }
    if (s->mbc_type == 2) {
        s->cartram[(a - 0xA000) & 0x1FF] = v & 0x0F;
        return;
    }
    if (!s->ram_size) return;
    s->cartram[(s->ram_off + (a - 0xA000)) % s->ram_size] = v;
}

/* ------------------------------------------------------------------ */
/* Timer                                                                */
/* ------------------------------------------------------------------ */

/* TAC input clock select -> which divc bit's falling edge clocks TIMA.
 * Shift = bit index + 1, so edges = multiples of 2^shift crossed. */
static const uint8_t TAC_SHIFT[4] = { 10, 4, 6, 8 };

static void timer_tick(GB *g, uint32_t cycles)
{
    GBState *s = &g->s;
    uint32_t old = s->divc;
    s->divc = (uint16_t)(s->divc + cycles);

    if (!(s->io[R_TAC] & 0x04)) return;
    uint32_t sh = TAC_SHIFT[s->io[R_TAC] & 3];
    uint32_t edges = ((old + cycles) >> sh) - (old >> sh);
    while (edges--) {
        if (++s->io[R_TIMA] == 0) {
            s->io[R_TIMA] = s->io[R_TMA];
            s->io[R_IF] |= I_TIMER;
        }
    }
}

static void div_reset(GB *g)
{
    GBState *s = &g->s;
    /* Resetting DIV drops the selected timer bit; if it was high, that is a
     * falling edge and TIMA ticks once (documented DMG quirk). */
    if (s->io[R_TAC] & 0x04) {
        uint32_t bit = 1u << (TAC_SHIFT[s->io[R_TAC] & 3] - 1);
        if (s->divc & bit) {
            if (++s->io[R_TIMA] == 0) {
                s->io[R_TIMA] = s->io[R_TMA];
                s->io[R_IF] |= I_TIMER;
            }
        }
    }
    s->divc = 0;
}

/* MBC3 RTC advances with emulated time (deterministic for RL). */
static void rtc_tick(GB *g, uint32_t cycles)
{
    GBState *s = &g->s;
    if (!s->has_rtc || (s->rtc[4] & 0x40)) return;   /* halt bit */
    s->rtc_sub += cycles;
    while (s->rtc_sub >= 4194304) {
        s->rtc_sub -= 4194304;
        if (++s->rtc[0] >= 60) {
            s->rtc[0] = 0;
            if (++s->rtc[1] >= 60) {
                s->rtc[1] = 0;
                if (++s->rtc[2] >= 24) {
                    s->rtc[2] = 0;
                    if (++s->rtc[3] == 0) {          /* day low wrapped */
                        if (s->rtc[4] & 1) {
                            s->rtc[4] &= ~1;
                            s->rtc[4] |= 0x80;       /* day counter carry */
                        } else {
                            s->rtc[4] |= 1;
                        }
                    }
                }
            }
        }
    }
}

/* Advance all time-driven peripherals by the cycles an instruction took.
 * One call site keeps the three subsystems in lockstep and lets the compiler
 * schedule them together instead of across three separate call boundaries. */
static inline void advance(GB *g, uint32_t cyc)
{
    timer_tick(g, cyc);
    ppu_tick(g, cyc);
    rtc_tick(g, cyc);
}

/* ------------------------------------------------------------------ */
/* Bus                                                                  */
/* ------------------------------------------------------------------ */

static uint8_t joypad_read(GB *g)
{
    GBState *s = &g->s;
    uint8_t sel = s->io[R_P1];
    uint8_t v = 0xC0 | (sel & 0x30) | 0x0F;
    if (!(sel & 0x10)) v &= ~((s->buttons >> 4) & 0x0F);   /* d-pad */
    if (!(sel & 0x20)) v &= ~(s->buttons & 0x0F);          /* buttons */
    return v;
}

static uint8_t io_read(GB *g, uint8_t x)
{
    GBState *s = &g->s;
    switch (x) {
    case R_P1:   return joypad_read(g);
    case R_DIV:  return s->divc >> 8;
    case R_STAT: return ppu_stat_read(g);
    case R_IF:   return 0xE0 | (s->io[R_IF] & 0x1F);
    default:     return s->io[x];
    }
}

static uint8_t rd8_slow(GB *g, uint16_t a);

/* Fast bus read: mapped page -> one indexed load; else full decode. */
static inline uint8_t rd8(GB *g, uint16_t a)
{
    const uint8_t *p = g->read_map[a >> 12];
    return p ? p[a & 0x0FFF] : rd8_slow(g, a);
}

static void oam_dma(GB *g, uint8_t page)
{
    /* Instant OAM DMA: exact 160-cycle bus contention is irrelevant to the
     * games RL targets, and everything stays deterministic. */
    uint16_t src = (uint16_t)page << 8;
    for (int i = 0; i < 0xA0; i++)
        g->s.oam[i] = rd8(g, src + i);
}

static void io_write(GB *g, uint8_t x, uint8_t v)
{
    GBState *s = &g->s;
    switch (x) {
    case R_P1:
        s->io[R_P1] = (s->io[R_P1] & 0xCF) | (v & 0x30);
        return;
    case R_SC:
        s->io[R_SC] = v;
        if (v & 0x80) {
            /* No link partner: capture the byte for the host, complete the
             * transfer immediately, shift in 0xFF. */
            if (g->serial_len < GB_SERIAL_BUF)
                g->serial[g->serial_len++] = s->io[R_SB];
            if (v & 0x01) {                   /* internal clock drives it */
                s->io[R_SB] = 0xFF;
                s->io[R_SC] = v & 0x7F;
                s->io[R_IF] |= I_SERIAL;
            }
        }
        return;
    case R_DIV:  div_reset(g); return;
    case R_DMA:  s->io[R_DMA] = v; oam_dma(g, v); return;
    case R_LCDC: case R_STAT: case R_LY: case R_LYC:
        ppu_lcd_write(g, x, v);
        return;
    case R_IF:   s->io[R_IF] = v & 0x1F; return;
    default:
        s->io[x] = v;
        return;
    }
}

static uint8_t rd8_slow(GB *g, uint16_t a)
{
    GBState *s = &g->s;
    switch (a >> 12) {
    case 0x0: case 0x1: case 0x2: case 0x3:
        return g->rom[s->rom_off0 + a];
    case 0x4: case 0x5: case 0x6: case 0x7:
        return g->rom[s->rom_off1 + (a & 0x3FFF)];
    case 0x8: case 0x9:
        return s->vram[a & 0x1FFF];
    case 0xA: case 0xB:
        return cartram_read(g, a);
    case 0xC: case 0xD:
        return s->wram[a & 0x1FFF];
    case 0xE:
        return s->wram[a & 0x1FFF];
    default:                                   /* 0xF000-0xFFFF */
        if (a < 0xFE00) return s->wram[a & 0x1FFF];
        if (a < 0xFEA0) return s->oam[a - 0xFE00];
        if (a < 0xFF00) return 0xFF;
        if (a < 0xFF80) return io_read(g, a & 0x7F);
        if (a < 0xFFFF) return s->hram[a - 0xFF80];
        return s->ie;
    }
}

static void wr8_slow(GB *g, uint16_t a, uint8_t v);

/* Fast bus write: mapped RAM page -> one indexed store; else full decode. */
static inline void wr8(GB *g, uint16_t a, uint8_t v)
{
    uint8_t *p = g->write_map[a >> 12];
    if (p) p[a & 0x0FFF] = v;
    else   wr8_slow(g, a, v);
}

static void wr8_slow(GB *g, uint16_t a, uint8_t v)
{
    GBState *s = &g->s;
    switch (a >> 12) {
    case 0x0: case 0x1: case 0x2: case 0x3:
    case 0x4: case 0x5: case 0x6: case 0x7:
        mbc_write(g, a, v);
        return;
    case 0x8: case 0x9:
        s->vram[a & 0x1FFF] = v;
        return;
    case 0xA: case 0xB:
        cartram_write(g, a, v);
        return;
    case 0xC: case 0xD:
        s->wram[a & 0x1FFF] = v;
        return;
    case 0xE:
        s->wram[a & 0x1FFF] = v;
        return;
    default:
        if (a < 0xFE00)      s->wram[a & 0x1FFF] = v;
        else if (a < 0xFEA0) s->oam[a - 0xFE00] = v;
        else if (a < 0xFF00) return;
        else if (a < 0xFF80) io_write(g, a & 0x7F, v);
        else if (a < 0xFFFF) s->hram[a - 0xFF80] = v;
        else                 s->ie = v;
    }
}

/* ------------------------------------------------------------------ */
/* SM83 CPU                                                             */
/* ------------------------------------------------------------------ */

#define RP_HL(g) (uint16_t)(((g)->s.h << 8) | (g)->s.l)
#define RP_BC(g) (uint16_t)(((g)->s.b << 8) | (g)->s.c)
#define RP_DE(g) (uint16_t)(((g)->s.d << 8) | (g)->s.e)

static inline uint8_t get_r8(GB *g, int i)
{
    i &= 7;
    return i == 6 ? rd8(g, RP_HL(g)) : *g->r8[i];
}

static inline void set_r8(GB *g, int i, uint8_t v)
{
    i &= 7;
    if (i == 6) wr8(g, RP_HL(g), v);
    else        *g->r8[i] = v;
}

static uint16_t get_r16(GB *g, int i)   /* BC DE HL SP */
{
    switch (i & 3) {
    case 0: return RP_BC(g);
    case 1: return RP_DE(g);
    case 2: return RP_HL(g);
    default: return g->s.sp;
    }
}

static void set_r16(GB *g, int i, uint16_t v)
{
    switch (i & 3) {
    case 0: g->s.b = v >> 8; g->s.c = (uint8_t)v; break;
    case 1: g->s.d = v >> 8; g->s.e = (uint8_t)v; break;
    case 2: g->s.h = v >> 8; g->s.l = (uint8_t)v; break;
    default: g->s.sp = v; break;
    }
}

static uint8_t fetch8(GB *g) { return rd8(g, g->s.pc++); }

static uint16_t fetch16(GB *g)
{
    uint8_t lo = fetch8(g), hi = fetch8(g);
    return (uint16_t)(hi << 8) | lo;
}

static void push16(GB *g, uint16_t v)
{
    wr8(g, --g->s.sp, v >> 8);
    wr8(g, --g->s.sp, (uint8_t)v);
}

static uint16_t pop16(GB *g)
{
    uint8_t lo = rd8(g, g->s.sp++);
    uint8_t hi = rd8(g, g->s.sp++);
    return (uint16_t)(hi << 8) | lo;
}

static int cond(GB *g, int cc)
{
    switch (cc & 3) {
    case 0: return !(g->s.f & FZ);
    case 1: return  (g->s.f & FZ) != 0;
    case 2: return !(g->s.f & FC);
    default: return (g->s.f & FC) != 0;
    }
}

static void alu(GB *g, int op, uint8_t v)
{
    GBState *s = &g->s;
    int a = s->a, cy = (s->f & FC) ? 1 : 0, r;
    switch (op & 7) {
    case 0:                                     /* ADD */
        r = a + v;
        s->f = (uint8_t)((((r & 0xFF) == 0) ? FZ : 0) |
                         (((a ^ v ^ r) & 0x10) ? FH : 0) | ((r > 0xFF) ? FC : 0));
        s->a = (uint8_t)r;
        break;
    case 1:                                     /* ADC */
        r = a + v + cy;
        s->f = (uint8_t)((((r & 0xFF) == 0) ? FZ : 0) |
                         (((a ^ v ^ r) & 0x10) ? FH : 0) | ((r > 0xFF) ? FC : 0));
        s->a = (uint8_t)r;
        break;
    case 2:                                     /* SUB */
        r = a - v;
        s->f = (uint8_t)(FN | (((r & 0xFF) == 0) ? FZ : 0) |
                         (((a ^ v ^ r) & 0x10) ? FH : 0) | ((r < 0) ? FC : 0));
        s->a = (uint8_t)r;
        break;
    case 3:                                     /* SBC */
        r = a - v - cy;
        s->f = (uint8_t)(FN | (((r & 0xFF) == 0) ? FZ : 0) |
                         (((a ^ v ^ r) & 0x10) ? FH : 0) | ((r < 0) ? FC : 0));
        s->a = (uint8_t)r;
        break;
    case 4:                                     /* AND */
        s->a &= v;
        s->f = (uint8_t)((s->a ? 0 : FZ) | FH);
        break;
    case 5:                                     /* XOR */
        s->a ^= v;
        s->f = s->a ? 0 : FZ;
        break;
    case 6:                                     /* OR */
        s->a |= v;
        s->f = s->a ? 0 : FZ;
        break;
    default:                                    /* CP */
        r = a - v;
        s->f = (uint8_t)(FN | (((r & 0xFF) == 0) ? FZ : 0) |
                         (((a ^ v ^ r) & 0x10) ? FH : 0) | ((r < 0) ? FC : 0));
        break;
    }
}

static uint8_t inc8(GB *g, uint8_t v)
{
    uint8_t r = v + 1;
    g->s.f = (uint8_t)((g->s.f & FC) | (r ? 0 : FZ) | (((v & 0x0F) == 0x0F) ? FH : 0));
    return r;
}

static uint8_t dec8(GB *g, uint8_t v)
{
    uint8_t r = v - 1;
    g->s.f = (uint8_t)((g->s.f & FC) | FN | (r ? 0 : FZ) | (((v & 0x0F) == 0) ? FH : 0));
    return r;
}

static void add_hl(GB *g, uint16_t v)
{
    uint32_t hl = RP_HL(g), r = hl + v;
    g->s.f = (uint8_t)((g->s.f & FZ) |
                       (((hl ^ v ^ r) & 0x1000) ? FH : 0) | ((r > 0xFFFF) ? FC : 0));
    g->s.h = (uint8_t)(r >> 8);
    g->s.l = (uint8_t)r;
}

static uint16_t add_sp_e8(GB *g)     /* shared by ADD SP,e8 / LD HL,SP+e8 */
{
    int8_t e = (int8_t)fetch8(g);
    uint16_t sp = g->s.sp;
    uint16_t r = (uint16_t)(sp + e);
    g->s.f = (uint8_t)((((sp ^ e ^ r) & 0x10) ? FH : 0) |
                       (((sp ^ e ^ r) & 0x100) ? FC : 0));
    return r;
}

static uint8_t cb_op(GB *g, int op, uint8_t v)   /* rot/shift group */
{
    GBState *s = &g->s;
    uint8_t c = (s->f & FC) ? 1 : 0, r;
    switch (op & 7) {
    case 0: r = (uint8_t)((v << 1) | (v >> 7)); s->f = (v & 0x80) ? FC : 0; break; /* RLC */
    case 1: r = (uint8_t)((v >> 1) | (v << 7)); s->f = (v & 1) ? FC : 0; break;    /* RRC */
    case 2: r = (uint8_t)((v << 1) | c);        s->f = (v & 0x80) ? FC : 0; break; /* RL  */
    case 3: r = (uint8_t)((v >> 1) | (c << 7)); s->f = (v & 1) ? FC : 0; break;    /* RR  */
    case 4: r = (uint8_t)(v << 1);              s->f = (v & 0x80) ? FC : 0; break; /* SLA */
    case 5: r = (uint8_t)((v >> 1) | (v & 0x80)); s->f = (v & 1) ? FC : 0; break;  /* SRA */
    case 6: r = (uint8_t)((v << 4) | (v >> 4)); s->f = 0; break;                   /* SWAP */
    default: r = v >> 1;                        s->f = (v & 1) ? FC : 0; break;    /* SRL */
    }
    if (r == 0) s->f |= FZ;
    return r;
}

static int cpu_cb(GB *g)
{
    uint8_t op = fetch8(g);
    int r = op & 7, bit = (op >> 3) & 7;
    switch (op >> 6) {
    case 0: set_r8(g, r, cb_op(g, bit, get_r8(g, r))); break;
    case 1: {                                          /* BIT */
        uint8_t v = get_r8(g, r);
        g->s.f = (uint8_t)((g->s.f & FC) | FH | ((v & (1 << bit)) ? 0 : FZ));
        break;
    }
    case 2: set_r8(g, r, get_r8(g, r) & ~(1 << bit)); break;   /* RES */
    default: set_r8(g, r, get_r8(g, r) | (1 << bit)); break;   /* SET */
    }
    return CYC_CB[op];
}

static const uint16_t IRQ_VEC[5] = { 0x40, 0x48, 0x50, 0x58, 0x60 };

/* While halted with no pending interrupt, nothing can raise IF before the
 * next PPU mode/line boundary or the TIMA overflow edge (serial and joypad
 * IRQs are event-driven, not time-driven, in this core). Skip there in one
 * burst instead of 4-cycle steps; timer_tick/ppu_tick/rtc_tick all apply
 * arbitrary cycle counts exactly, so the result is bit-identical. */
static uint32_t halt_skip_cycles(GB *g)
{
    GBState *s = &g->s;
    uint32_t until;

    if (!(s->io[R_LCDC] & 0x80)) {
        /* LCD off: no PPU IRQs at all; burst to the frame-pacing wrap. */
        until = (154 - (uint32_t)s->io[R_LY]) * 456 - s->line_dot;
    } else if (s->io[R_LY] >= 144) {
        /* In vblank only inc_ly can raise IF, and only via LYC match; with
         * the LYC IRQ disabled, burst to the end of line 153 (the wrap to
         * line 0 enters mode 2, which can fire STAT). */
        if (!(s->io[R_STAT] & 0x40))
            until = (154 - (uint32_t)s->io[R_LY]) * 456 - s->line_dot;
        else
            until = 456 - (uint32_t)s->line_dot;
    } else if (!(s->io[R_STAT] & 0x68)) {
        /* No LYC/mode-2/mode-0 STAT sources enabled: nothing can raise IF
         * before vblank entry at LY=144 dot 0 (mode-1 STAT, bit 4, fires
         * there too). Burst the whole remaining visible region in one go —
         * ppu_tick still walks every mode boundary internally, so rendering
         * and STAT mode bits stay exact. */
        until = (144 - (uint32_t)s->io[R_LY]) * 456 - s->line_dot;
    } else if (s->line_dot < 252) {
        /* Mode-3 entry at dot 80 raises no IF (ppu_tick still renders the
         * scanline there); next IF-capable boundary is hblank at dot 252. */
        until = 252 - (uint32_t)s->line_dot;
    } else {
        until = 456 - (uint32_t)s->line_dot;
    }

    if (s->io[R_TAC] & 0x04) {
        uint32_t step = 1u << TAC_SHIFT[s->io[R_TAC] & 3];
        uint32_t first = step - (s->divc & (step - 1));
        uint32_t t = first + (uint32_t)(255 - s->io[R_TIMA]) * step;
        if (t < until) until = t;
    }

    until &= ~3u;                                 /* 4-cycle machine granularity */
    return until < 4 ? 4 : until;
}

/* The interpreter core. Internal linkage so the frame loop below calls it
 * directly (and LTO can inline it) instead of routing every instruction
 * through the public gb_step's PLT entry. */
__attribute__((noinline)) static int step_one(GB *g)
{
    GBState *s = &g->s;
    int cyc;

    uint8_t pending = s->io[R_IF] & s->ie & 0x1F;

    if (s->ime && pending) {
        s->halted = 0;
        s->ime = 0;
        s->ime_delay = 0;
        int i = __builtin_ctz(pending);
        s->io[R_IF] &= ~(1 << i);
        push16(g, s->pc);
        s->pc = IRQ_VEC[i];
        cyc = 20;
        advance(g, cyc);
        s->cycles += cyc;
        return cyc;
    }

    if (s->halted) {
        if (pending) {
            s->halted = 0;
        } else {
            cyc = (int)halt_skip_cycles(g);
            advance(g, cyc);
            s->cycles += cyc;
            return cyc;
        }
    }

    if (s->ime_delay) { s->ime_delay = 0; s->ime = 1; }

    uint8_t op = rd8(g, s->pc);
    if (s->halt_bug) s->halt_bug = 0;      /* PC not incremented: byte re-read */
    else s->pc++;

    cyc = CYC_BASE[op];

    if (op >= 0x40 && op < 0x80) {
        if (op == 0x76) {                   /* HALT */
            if (!s->ime && (s->io[R_IF] & s->ie & 0x1F))
                s->halt_bug = 1;
            else
                s->halted = 1;
        } else {
            set_r8(g, (op >> 3) & 7, get_r8(g, op & 7));
        }
    } else if (op >= 0x80 && op < 0xC0) {
        alu(g, (op >> 3) & 7, get_r8(g, op & 7));
    } else switch (op) {
    /* --- 0x00-0x3F --- */
    case 0x00: break;                                        /* NOP */
    case 0x10: fetch8(g); break;                             /* STOP (treated as 2-byte NOP) */
    case 0x01: case 0x11: case 0x21: case 0x31:
        set_r16(g, op >> 4, fetch16(g)); break;              /* LD rr,d16 */
    case 0x02: wr8(g, RP_BC(g), s->a); break;
    case 0x12: wr8(g, RP_DE(g), s->a); break;
    case 0x22: { uint16_t hl = RP_HL(g); wr8(g, hl, s->a); hl++; s->h = hl >> 8; s->l = (uint8_t)hl; break; }
    case 0x32: { uint16_t hl = RP_HL(g); wr8(g, hl, s->a); hl--; s->h = hl >> 8; s->l = (uint8_t)hl; break; }
    case 0x0A: s->a = rd8(g, RP_BC(g)); break;
    case 0x1A: s->a = rd8(g, RP_DE(g)); break;
    case 0x2A: { uint16_t hl = RP_HL(g); s->a = rd8(g, hl); hl++; s->h = hl >> 8; s->l = (uint8_t)hl; break; }
    case 0x3A: { uint16_t hl = RP_HL(g); s->a = rd8(g, hl); hl--; s->h = hl >> 8; s->l = (uint8_t)hl; break; }
    case 0x03: case 0x13: case 0x23: case 0x33:
        set_r16(g, op >> 4, get_r16(g, op >> 4) + 1); break; /* INC rr */
    case 0x0B: case 0x1B: case 0x2B: case 0x3B:
        set_r16(g, op >> 4, get_r16(g, op >> 4) - 1); break; /* DEC rr */
    case 0x04: case 0x0C: case 0x14: case 0x1C:
    case 0x24: case 0x2C: case 0x34: case 0x3C:
        set_r8(g, (op >> 3) & 7, inc8(g, get_r8(g, (op >> 3) & 7))); break;
    case 0x05: case 0x0D: case 0x15: case 0x1D:
    case 0x25: case 0x2D: case 0x35: case 0x3D:
        set_r8(g, (op >> 3) & 7, dec8(g, get_r8(g, (op >> 3) & 7))); break;
    case 0x06: case 0x0E: case 0x16: case 0x1E:
    case 0x26: case 0x2E: case 0x36: case 0x3E:
        set_r8(g, (op >> 3) & 7, fetch8(g)); break;          /* LD r,d8 */
    case 0x07:                                               /* RLCA */
        s->f = (s->a & 0x80) ? FC : 0;
        s->a = (uint8_t)((s->a << 1) | (s->a >> 7));
        break;
    case 0x0F:                                               /* RRCA */
        s->f = (s->a & 1) ? FC : 0;
        s->a = (uint8_t)((s->a >> 1) | (s->a << 7));
        break;
    case 0x17: {                                             /* RLA */
        uint8_t c = (s->f & FC) ? 1 : 0;
        s->f = (s->a & 0x80) ? FC : 0;
        s->a = (uint8_t)((s->a << 1) | c);
        break;
    }
    case 0x1F: {                                             /* RRA */
        uint8_t c = (s->f & FC) ? 1 : 0;
        s->f = (s->a & 1) ? FC : 0;
        s->a = (uint8_t)((s->a >> 1) | (c << 7));
        break;
    }
    case 0x27: {                                             /* DAA */
        int a = s->a;
        if (!(s->f & FN)) {
            if ((s->f & FC) || a > 0x99) { a += 0x60; s->f |= FC; }
            if ((s->f & FH) || (a & 0x0F) > 0x09) a += 0x06;
        } else {
            if (s->f & FC) a -= 0x60;
            if (s->f & FH) a -= 0x06;
        }
        s->f &= ~(FH | FZ);
        if ((a & 0xFF) == 0) s->f |= FZ;
        s->a = (uint8_t)a;
        break;
    }
    case 0x2F: s->a = ~s->a; s->f |= FN | FH; break;         /* CPL */
    case 0x37: s->f = (uint8_t)((s->f & FZ) | FC); break;    /* SCF */
    case 0x3F: s->f = (uint8_t)((s->f & FZ) | ((s->f & FC) ^ FC)); break; /* CCF */
    case 0x08: {                                             /* LD (a16),SP */
        uint16_t a = fetch16(g);
        wr8(g, a, (uint8_t)s->sp);
        wr8(g, a + 1, s->sp >> 8);
        break;
    }
    case 0x09: case 0x19: case 0x29: case 0x39:
        add_hl(g, get_r16(g, op >> 4)); break;
    case 0x18: {                                             /* JR e8 */
        int8_t e = (int8_t)fetch8(g);
        s->pc += e;
        break;
    }
    case 0x20: case 0x28: case 0x30: case 0x38: {            /* JR cc,e8 */
        int8_t e = (int8_t)fetch8(g);
        if (cond(g, (op >> 3) & 3)) { s->pc += e; cyc += CYC_EXTRA[op]; }
        break;
    }
    /* --- 0xC0-0xFF --- */
    case 0xC0: case 0xC8: case 0xD0: case 0xD8:              /* RET cc */
        if (cond(g, (op >> 3) & 3)) { s->pc = pop16(g); cyc += CYC_EXTRA[op]; }
        break;
    case 0xC9: s->pc = pop16(g); break;                      /* RET */
    case 0xD9: s->pc = pop16(g); s->ime = 1; break;          /* RETI */
    case 0xC1: case 0xD1: case 0xE1: {                       /* POP rr */
        uint16_t v = pop16(g);
        set_r16(g, (op >> 4) & 3, v);
        break;
    }
    case 0xF1: {                                             /* POP AF */
        uint16_t v = pop16(g);
        s->a = v >> 8;
        s->f = v & 0xF0;
        break;
    }
    case 0xC5: push16(g, RP_BC(g)); break;
    case 0xD5: push16(g, RP_DE(g)); break;
    case 0xE5: push16(g, RP_HL(g)); break;
    case 0xF5: push16(g, (uint16_t)((s->a << 8) | (s->f & 0xF0))); break;
    case 0xC2: case 0xCA: case 0xD2: case 0xDA: {            /* JP cc,a16 */
        uint16_t a = fetch16(g);
        if (cond(g, (op >> 3) & 3)) { s->pc = a; cyc += CYC_EXTRA[op]; }
        break;
    }
    case 0xC3: s->pc = fetch16(g); break;                    /* JP a16 */
    case 0xE9: s->pc = RP_HL(g); break;                      /* JP HL */
    case 0xC4: case 0xCC: case 0xD4: case 0xDC: {            /* CALL cc,a16 */
        uint16_t a = fetch16(g);
        if (cond(g, (op >> 3) & 3)) {
            push16(g, s->pc);
            s->pc = a;
            cyc += CYC_EXTRA[op];
        }
        break;
    }
    case 0xCD: {                                             /* CALL a16 */
        uint16_t a = fetch16(g);
        push16(g, s->pc);
        s->pc = a;
        break;
    }
    case 0xC7: case 0xCF: case 0xD7: case 0xDF:
    case 0xE7: case 0xEF: case 0xF7: case 0xFF:              /* RST */
        push16(g, s->pc);
        s->pc = (uint16_t)(op & 0x38);
        break;
    case 0xC6: case 0xCE: case 0xD6: case 0xDE:
    case 0xE6: case 0xEE: case 0xF6: case 0xFE:              /* ALU a,d8 */
        alu(g, (op >> 3) & 7, fetch8(g));
        break;
    case 0xCB: cyc = cpu_cb(g); break;
    case 0xE0: wr8(g, 0xFF00 + fetch8(g), s->a); break;      /* LDH (a8),A */
    case 0xF0: s->a = rd8(g, 0xFF00 + fetch8(g)); break;     /* LDH A,(a8) */
    case 0xE2: wr8(g, 0xFF00 + s->c, s->a); break;
    case 0xF2: s->a = rd8(g, 0xFF00 + s->c); break;
    case 0xEA: wr8(g, fetch16(g), s->a); break;              /* LD (a16),A */
    case 0xFA: s->a = rd8(g, fetch16(g)); break;             /* LD A,(a16) */
    case 0xE8: s->sp = add_sp_e8(g); break;                  /* ADD SP,e8 */
    case 0xF8: {                                             /* LD HL,SP+e8 */
        uint16_t r = add_sp_e8(g);
        s->h = r >> 8;
        s->l = (uint8_t)r;
        break;
    }
    case 0xF9: s->sp = RP_HL(g); break;                      /* LD SP,HL */
    case 0xF3: s->ime = 0; s->ime_delay = 0; break;          /* DI */
    case 0xFB: if (!s->ime) s->ime_delay = 1; break;         /* EI */
    default:                                                 /* illegal opcode */
        cyc = 4;
        break;
    }

    advance(g, cyc);
    s->cycles += cyc;
    return cyc;
}

int gb_step(GB *g) { return step_one(g); }

/* ------------------------------------------------------------------ */
/* Public API                                                           */
/* ------------------------------------------------------------------ */

GB *gb_new(void)
{
    GB *g = (GB *)calloc(1, sizeof(GB));
    if (g) g->render = 1;
    return g;
}

void gb_free(GB *g)
{
    if (!g) return;
    free((void *)g->rom);
    free(g);
}

static const uint32_t RAM_SIZES[6] = { 0, 0x800, 0x2000, 0x8000, 0x20000, 0x10000 };

int gb_load_rom(GB *g, const uint8_t *data, uint32_t size)
{
    if (!g || !data || size < 0x150) return -1;

    /* Accept any .gb/.gbc image: pad to a 16 KiB bank boundary (min 32 KiB)
     * with 0xFF, like unmapped cartridge bus reads. GBC-only games run in
     * DMG mode, exactly as on original hardware. */
    uint32_t padded = size < 0x8000 ? 0x8000 : ((size + 0x3FFF) & ~0x3FFFu);
    uint8_t *copy = (uint8_t *)malloc(padded);
    if (!copy) return -2;
    memset(copy, 0xFF, padded);
    memcpy(copy, data, size);
    size = padded;
    free((void *)g->rom);
    g->rom = copy;
    g->rom_size = size;

    GBState *s = &g->s;
    uint8_t cart = data[0x147];
    switch (cart) {
    case 0x00: case 0x08: case 0x09:
        s->mbc_type = 0; break;
    case 0x01: case 0x02: case 0x03:
        s->mbc_type = 1; break;
    case 0x05: case 0x06:
        s->mbc_type = 2; break;
    case 0x0F: case 0x10:
        s->mbc_type = 3; s->has_rtc = 1; break;
    case 0x11: case 0x12: case 0x13:
        s->mbc_type = 3; break;
    case 0x19: case 0x1A: case 0x1B: case 0x1C: case 0x1D: case 0x1E:
        s->mbc_type = 5; break;
    default:
        s->mbc_type = 1; break;                /* best effort */
    }
    s->rom_banks = size / 0x4000;
    uint8_t rs = data[0x149];
    s->ram_size = (rs < 6) ? RAM_SIZES[rs] : 0;
    if (s->mbc_type == 2) s->ram_size = 0x200;

    gb_reset(g);
    return 0;
}

void gb_reset(GB *g)
{
    GBState *s = &g->s;

    /* Bind the register file (opcode order B C D E H L _ A; 6=(HL) unused). */
    g->r8[0] = &s->b; g->r8[1] = &s->c; g->r8[2] = &s->d; g->r8[3] = &s->e;
    g->r8[4] = &s->h; g->r8[5] = &s->l; g->r8[6] = NULL;   g->r8[7] = &s->a;

    uint8_t mbc = s->mbc_type, rtc = s->has_rtc;
    uint32_t banks = s->rom_banks, rsz = s->ram_size;
    uint8_t cartram_keep[GB_CARTRAM_MAX];
    memcpy(cartram_keep, s->cartram, sizeof(cartram_keep));   /* battery RAM survives reset */

    memset(s, 0, sizeof(*s));
    s->mbc_type = mbc; s->has_rtc = rtc;
    s->rom_banks = banks; s->ram_size = rsz;
    memcpy(s->cartram, cartram_keep, sizeof(cartram_keep));

    /* DMG post-boot-ROM register state (documented in Pan Docs) */
    s->a = 0x01; s->f = 0xB0;
    s->b = 0x00; s->c = 0x13;
    s->d = 0x00; s->e = 0xD8;
    s->h = 0x01; s->l = 0x4D;
    s->sp = 0xFFFE; s->pc = 0x0100;
    s->rom_bank = 1;

    s->io[R_P1]   = 0x30;
    s->io[R_TAC]  = 0xF8;
    s->io[R_IF]   = 0x01;
    s->io[R_LCDC] = 0x91;
    s->io[R_STAT] = 0x85 & 0x78;
    s->io[R_BGP]  = 0xFC;
    s->io[R_OBP0] = 0xFF;
    s->io[R_OBP1] = 0xFF;
    s->io[R_WY]   = 0x00;
    s->io[R_WX]   = 0x00;
    s->io[0x10] = 0x80; s->io[0x11] = 0xBF; s->io[0x12] = 0xF3;
    s->io[0x14] = 0xBF; s->io[0x16] = 0x3F; s->io[0x19] = 0xBF;
    s->io[0x1A] = 0x7F; s->io[0x1B] = 0xFF; s->io[0x1C] = 0x9F;
    s->io[0x1E] = 0xBF; s->io[0x20] = 0xFF; s->io[0x23] = 0xBF;
    s->io[0x24] = 0x77; s->io[0x25] = 0xF3; s->io[0x26] = 0xF1;
    s->divc = 0xABCC;
    s->ppu_mode = 1;                        /* boot hands off inside vblank */
    s->io[R_LY] = 0;

    mbc_remap(g);
    g->serial_len = 0;
}

uint32_t gb_run_frame(GB *g)
{
    GBState *s = &g->s;
    uint64_t start = s->cycles;
    uint64_t cap = start + GB_FRAME_CYCLES + 456; /* LCD-off fallback */
    s->frame_done = 0;
    while (!s->frame_done && s->cycles < cap)
        step_one(g);
    return (uint32_t)(s->cycles - start);
}

uint32_t gb_run_frames(GB *g, uint32_t n)
{
    uint32_t t = 0;
    while (n--) t += gb_run_frame(g);
    return t;
}

uint64_t gb_cycles(GB *g) { return g->s.cycles; }
uint32_t gb_frames(GB *g) { return g->s.frames; }

void gb_set_buttons(GB *g, uint8_t mask)
{
    uint8_t newly = mask & ~g->s.buttons;
    g->s.buttons = mask;
    if (newly) g->s.io[R_IF] |= I_JOYPAD;
}

uint8_t gb_get_buttons(GB *g) { return g->s.buttons; }

uint8_t gb_read(GB *g, uint16_t a)          { return rd8(g, a); }
void    gb_write(GB *g, uint16_t a, uint8_t v) { wr8(g, a, v); }

uint8_t *gb_ptr_vram(GB *g)        { return g->s.vram; }
uint8_t *gb_ptr_wram(GB *g)        { return g->s.wram; }
uint8_t *gb_ptr_oam(GB *g)         { return g->s.oam; }
uint8_t *gb_ptr_hram(GB *g)        { return g->s.hram; }
uint8_t *gb_ptr_io(GB *g)          { return g->s.io; }
uint8_t *gb_ptr_cartram(GB *g)     { return g->s.cartram; }
uint8_t *gb_ptr_framebuffer(GB *g) { return g->s.framebuf; }
uint32_t gb_cartram_size(GB *g)    { return g->s.ram_size; }

uint32_t gb_get_reg(GB *g, int r)
{
    GBState *s = &g->s;
    switch (r) {
    case GB_REG_A: return s->a;   case GB_REG_F: return s->f;
    case GB_REG_B: return s->b;   case GB_REG_C: return s->c;
    case GB_REG_D: return s->d;   case GB_REG_E: return s->e;
    case GB_REG_H: return s->h;   case GB_REG_L: return s->l;
    case GB_REG_SP: return s->sp; case GB_REG_PC: return s->pc;
    case GB_REG_AF: return (uint32_t)((s->a << 8) | s->f);
    case GB_REG_BC: return RP_BC(g);
    case GB_REG_DE: return RP_DE(g);
    case GB_REG_HL: return RP_HL(g);
    case GB_REG_IME: return s->ime;
    case GB_REG_HALTED: return s->halted;
    default: return 0;
    }
}

void gb_set_reg(GB *g, int r, uint32_t v)
{
    GBState *s = &g->s;
    switch (r) {
    case GB_REG_A: s->a = (uint8_t)v; break;
    case GB_REG_F: s->f = (uint8_t)(v & 0xF0); break;
    case GB_REG_B: s->b = (uint8_t)v; break;
    case GB_REG_C: s->c = (uint8_t)v; break;
    case GB_REG_D: s->d = (uint8_t)v; break;
    case GB_REG_E: s->e = (uint8_t)v; break;
    case GB_REG_H: s->h = (uint8_t)v; break;
    case GB_REG_L: s->l = (uint8_t)v; break;
    case GB_REG_SP: s->sp = (uint16_t)v; break;
    case GB_REG_PC: s->pc = (uint16_t)v; break;
    case GB_REG_AF: s->a = (uint8_t)(v >> 8); s->f = (uint8_t)(v & 0xF0); break;
    case GB_REG_BC: s->b = (uint8_t)(v >> 8); s->c = (uint8_t)v; break;
    case GB_REG_DE: s->d = (uint8_t)(v >> 8); s->e = (uint8_t)v; break;
    case GB_REG_HL: s->h = (uint8_t)(v >> 8); s->l = (uint8_t)v; break;
    case GB_REG_IME: s->ime = v ? 1 : 0; break;
    case GB_REG_HALTED: s->halted = v ? 1 : 0; break;
    }
}

void gb_set_render(GB *g, int on) { g->render = on ? 1 : 0; }

uint32_t gb_state_size(void) { return (uint32_t)sizeof(GBState); }

void gb_save_state(GB *g, uint8_t *out)
{
    memcpy(out, &g->s, sizeof(GBState));
}

int gb_load_state(GB *g, const uint8_t *in, uint32_t size)
{
    if (size != sizeof(GBState)) return -1;
    memcpy(&g->s, in, sizeof(GBState));
    /* Treat the blob as untrusted: keep only fields the loaded ROM defines,
     * so a hostile state can't point the mapper outside its buffers. */
    if (g->s.mbc_type > 5) g->s.mbc_type = 1;
    g->s.rom_banks = g->rom_size / 0x4000u;
    if (g->s.ram_size > GB_CARTRAM_MAX) g->s.ram_size = GB_CARTRAM_MAX;
    mbc_remap(g);                          /* re-derives + clamps all offsets */
    return 0;
}

void gb_set_timing(GB *g, uint16_t divc) { g->s.divc = divc; }
uint16_t gb_get_timing(GB *g) { return g->s.divc; }

void gb_set_ppu_state(GB *g, uint8_t mode, uint16_t line_dot,
                      uint8_t win_line, uint8_t stat_line)
{
    g->s.ppu_mode = mode & 3;
    g->s.line_dot = line_dot % 456;
    g->s.win_line = win_line;
    g->s.stat_line = stat_line ? 1 : 0;
}

uint32_t gb_get_ppu_state(GB *g)
{
    return (uint32_t)g->s.ppu_mode | ((uint32_t)g->s.line_dot << 8) |
           ((uint32_t)g->s.win_line << 24);
}

void gb_set_counters(GB *g, uint64_t cycles, uint32_t frames)
{
    g->s.cycles = cycles;
    g->s.frames = frames;
}

void gb_set_mbc_state(GB *g, uint16_t rom_bank, uint8_t ram_bank,
                      uint8_t ram_enable, uint8_t mbc1_mode)
{
    g->s.rom_bank = rom_bank;
    g->s.ram_bank = ram_bank;
    g->s.ram_enable = ram_enable ? 1 : 0;
    g->s.mbc1_mode = mbc1_mode ? 1 : 0;
    mbc_remap(g);
}

uint32_t gb_get_mbc_state(GB *g)
{
    return (uint32_t)g->s.rom_bank | ((uint32_t)g->s.ram_bank << 16) |
           ((uint32_t)(g->s.ram_enable & 1) << 24) |
           ((uint32_t)(g->s.mbc1_mode & 1) << 25);
}

void gb_set_rtc_state(GB *g, const uint8_t rtc[5], const uint8_t latched[5])
{
    memcpy(g->s.rtc, rtc, 5);
    memcpy(g->s.rtc_latched, latched, 5);
}

void gb_get_rtc_state(GB *g, uint8_t out[10])
{
    memcpy(out, g->s.rtc, 5);
    memcpy(out + 5, g->s.rtc_latched, 5);
}

uint32_t gb_serial_read(GB *g, uint8_t *out, uint32_t max)
{
    uint32_t n = g->serial_len < max ? g->serial_len : max;
    memcpy(out, g->serial, n);
    memmove(g->serial, g->serial + n, g->serial_len - n);
    g->serial_len -= n;
    return n;
}

uint8_t  gb_cart_type(GB *g) { return g->rom ? g->rom[0x147] : 0; }
uint32_t gb_rom_banks(GB *g) { return g->s.rom_banks; }
