/*
 * rlgb — a from-scratch Game Boy (DMG) core built for headless RL training.
 *
 * Implemented from public hardware documentation (Pan Docs, gbdev community
 * opcode tables). No emulator source code was copied.
 *
 * Design rules:
 *   - All machine state lives in one flat, pointer-free struct (GBState) so a
 *     save state is a single memcpy. ROM stays outside (read-only, shared).
 *   - Headless-first: no audio synthesis, no window, optional pixel rendering.
 *   - Everything is exposed: raw pointers into VRAM/WRAM/OAM/HRAM/IO/cart RAM,
 *     CPU registers, and a full bus peek/poke that bypasses nothing.
 *
 * License: CC BY-NC-ND 4.0. Built by BuiltNotTaught.
 */
#ifndef RLGB_GB_H
#define RLGB_GB_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Internal linkage marker for cross-file core functions (gb.c <-> ppu.c).
 * Hidden visibility keeps them non-interposable, so with -flto the linker can
 * inline them and, failing that, still call them directly instead of routing
 * every per-instruction ppu_tick through the shared-object PLT. */
#if defined(__GNUC__) || defined(__clang__)
#define GB_INTERNAL __attribute__((visibility("hidden")))
#else
#define GB_INTERNAL
#endif

#define GB_W 160
#define GB_H 144
#define GB_FRAME_CYCLES 70224          /* t-cycles per frame, 59.7275 Hz */
#define GB_CARTRAM_MAX 0x20000         /* 128 KiB, largest licensed cart RAM */
#define GB_SERIAL_BUF 4096

/* Joypad button bit mask (gb_set_buttons) */
enum {
    GB_BTN_A      = 0x01,
    GB_BTN_B      = 0x02,
    GB_BTN_SELECT = 0x04,
    GB_BTN_START  = 0x08,
    GB_BTN_RIGHT  = 0x10,
    GB_BTN_LEFT   = 0x20,
    GB_BTN_UP     = 0x40,
    GB_BTN_DOWN   = 0x80,
};

/* CPU register ids for gb_get_reg / gb_set_reg */
enum {
    GB_REG_A, GB_REG_F, GB_REG_B, GB_REG_C, GB_REG_D, GB_REG_E,
    GB_REG_H, GB_REG_L, GB_REG_SP, GB_REG_PC,
    GB_REG_AF, GB_REG_BC, GB_REG_DE, GB_REG_HL,
    GB_REG_IME, GB_REG_HALTED,
};

/* Everything below is snapshotted verbatim by gb_save_state(). Keep it flat:
 * fixed-width integers and arrays only — no pointers, no host handles. */
typedef struct {
    /* --- SM83 CPU --- */
    uint8_t  a, f, b, c, d, e, h, l;
    uint16_t sp, pc;
    uint8_t  ime, ime_delay, halted, halt_bug, stopped;
    uint64_t cycles;                   /* total t-cycles since reset */
    uint32_t frames;                   /* completed frames since reset */

    /* --- memories --- */
    uint8_t vram[0x2000];
    uint8_t wram[0x2000];
    uint8_t oam[0xA0];
    uint8_t hram[0x7F];
    uint8_t io[0x80];                  /* FF00-FF7F backing store */
    uint8_t ie;                        /* FFFF */
    uint8_t cartram[GB_CARTRAM_MAX];

    /* --- cartridge / MBC --- */
    uint8_t  mbc_type;                 /* 0 none, 1 MBC1, 2 MBC2, 3 MBC3, 5 MBC5 */
    uint8_t  has_rtc;
    uint8_t  ram_enable;
    uint8_t  mbc1_mode;
    uint16_t rom_bank;                 /* raw register value */
    uint8_t  ram_bank;                 /* raw register value (or RTC select) */
    uint32_t rom_off0, rom_off1;       /* resolved byte offsets into ROM */
    uint32_t ram_off;                  /* resolved byte offset into cartram */
    uint32_t rom_banks;                /* number of 16 KiB banks */
    uint32_t ram_size;                 /* bytes of cart RAM present */

    /* --- MBC3 RTC --- */
    uint8_t  rtc[5];                   /* S, M, H, DL, DH (live) */
    uint8_t  rtc_latched[5];
    uint8_t  rtc_latch_prev;
    uint64_t rtc_sub;                  /* t-cycle accumulator toward 1 second */

    /* --- timer --- */
    uint16_t divc;                     /* 16-bit divider; DIV = divc >> 8 */

    /* --- PPU --- */
    uint8_t  ppu_mode;                 /* 0 hblank, 1 vblank, 2 oam, 3 draw */
    uint16_t line_dot;                 /* dot within current line, 0..455 */
    uint8_t  win_line;                 /* window internal line counter */
    uint8_t  stat_line;                /* STAT interrupt line (edge detect) */
    uint8_t  frame_done;               /* set at vblank entry, cleared by run_frame */
    uint8_t  framebuf[GB_W * GB_H];    /* post-palette shade 0..3 per pixel */

    /* --- joypad --- */
    uint8_t buttons;                   /* GB_BTN_* mask, 1 = pressed */
} GBState;

typedef struct GB {
    /* host-side, NOT part of save states */
    const uint8_t *rom;
    uint32_t rom_size;
    int render;                        /* 0 = skip pixel work (fastest) */
    uint8_t serial[GB_SERIAL_BUF];     /* bytes games pushed out the link port */
    uint32_t serial_len;

    /* Bus page table: one base pointer per 4 KiB page so a hot read/write is
     * an indexed load instead of an address-range switch. A NULL entry means
     * "needs logic" (cart RAM banking, MMIO) and falls to the slow path.
     * Derived state — rebuilt from GBState by mbc_remap(), never saved. */
    const uint8_t *read_map[16];
    uint8_t       *write_map[16];

    /* SM83 8-bit register file in opcode-encoding order (B C D E H L _ A);
     * index 6 = (HL) is handled specially. Lets get_r8/set_r8 be a single
     * indexed load instead of a jump-table branch. Points into s (stable for
     * the GB's lifetime), so it survives load_state's memcpy. */
    uint8_t *r8[8];

    GBState s;
} GB;

/* lifecycle */
GB  *gb_new(void);
void gb_free(GB *g);
/* Load a ROM image. The buffer is copied internally. Returns 0 on success. */
int  gb_load_rom(GB *g, const uint8_t *data, uint32_t size);
/* Reset to post-boot-ROM state (no boot ROM required or used). */
void gb_reset(GB *g);

/* execution */
int      gb_step(GB *g);               /* one instruction; returns t-cycles */
uint32_t gb_run_frame(GB *g);          /* run to next frame boundary; returns t-cycles */
uint32_t gb_run_frames(GB *g, uint32_t n);
uint64_t gb_cycles(GB *g);
uint32_t gb_frames(GB *g);

/* input */
void gb_set_buttons(GB *g, uint8_t mask);
uint8_t gb_get_buttons(GB *g);

/* unrestricted bus access (exactly what the CPU would see/do) */
uint8_t gb_read(GB *g, uint16_t addr);
void    gb_write(GB *g, uint16_t addr, uint8_t v);

/* zero-copy pointers into live machine memory (read AND write) */
uint8_t *gb_ptr_vram(GB *g);
uint8_t *gb_ptr_wram(GB *g);
uint8_t *gb_ptr_oam(GB *g);
uint8_t *gb_ptr_hram(GB *g);
uint8_t *gb_ptr_io(GB *g);
uint8_t *gb_ptr_cartram(GB *g);
uint8_t *gb_ptr_framebuffer(GB *g);    /* GB_W*GB_H bytes, values 0..3 */
uint32_t gb_cartram_size(GB *g);

/* CPU register access */
uint32_t gb_get_reg(GB *g, int reg);
void     gb_set_reg(GB *g, int reg, uint32_t v);

/* rendering control: 0 = headless-fast (PPU timing still exact, no pixels) */
void gb_set_render(GB *g, int on);

/* save states: flat snapshot of GBState */
uint32_t gb_state_size(void);
void     gb_save_state(GB *g, uint8_t *out);       /* out: gb_state_size() bytes */
int      gb_load_state(GB *g, const uint8_t *in, uint32_t size);

/* Internal-state accessors for foreign save-state interop (PyBoy import
 * needs to reconstruct timing/mapper state that isn't bus-visible). */
void     gb_set_timing(GB *g, uint16_t divc);
uint16_t gb_get_timing(GB *g);
void     gb_set_ppu_state(GB *g, uint8_t mode, uint16_t line_dot,
                          uint8_t win_line, uint8_t stat_line);
uint32_t gb_get_ppu_state(GB *g);   /* mode | line_dot<<8 | win_line<<24 */
void     gb_set_counters(GB *g, uint64_t cycles, uint32_t frames);
void     gb_set_mbc_state(GB *g, uint16_t rom_bank, uint8_t ram_bank,
                          uint8_t ram_enable, uint8_t mbc1_mode);
uint32_t gb_get_mbc_state(GB *g);   /* rom_bank | ram_bank<<16 | enable<<24 | mode<<25 */
void     gb_set_rtc_state(GB *g, const uint8_t rtc[5], const uint8_t latched[5]);
void     gb_get_rtc_state(GB *g, uint8_t out[10]);

/* serial/link output capture (Blargg test ROMs print here) */
uint32_t gb_serial_read(GB *g, uint8_t *out, uint32_t max);

/* cartridge header info */
uint8_t  gb_cart_type(GB *g);
uint32_t gb_rom_banks(GB *g);

#ifdef __cplusplus
}
#endif
#endif
