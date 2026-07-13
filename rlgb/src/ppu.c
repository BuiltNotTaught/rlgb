/*
 * rlgb PPU: scanline renderer + exact mode timing (456 dots/line, 154 lines).
 * Written from public DMG hardware documentation (Pan Docs). CC BY-NC-ND 4.0 license.
 * Built by BuiltNotTaught.
 */
#include <string.h>
#include "gb.h"

enum { FZ = 0x80 };
enum { I_VBLANK = 1, I_STAT = 2 };
enum {
    R_IF = 0x0F, R_LCDC = 0x40, R_STAT = 0x41, R_SCY = 0x42, R_SCX = 0x43,
    R_LY = 0x44, R_LYC = 0x45, R_BGP = 0x47, R_OBP0 = 0x48, R_OBP1 = 0x49,
    R_WY = 0x4A, R_WX = 0x4B,
};

/* LCDC bits */
enum {
    LCDC_BG_EN = 0x01, LCDC_OBJ_EN = 0x02, LCDC_OBJ16 = 0x04,
    LCDC_BG_MAP = 0x08, LCDC_TILEDATA = 0x10, LCDC_WIN_EN = 0x20,
    LCDC_WIN_MAP = 0x40, LCDC_ON = 0x80,
};

GB_INTERNAL uint8_t ppu_stat_read(GB *g)
{
    GBState *s = &g->s;
    uint8_t lyc_eq = (s->io[R_LY] == s->io[R_LYC]) ? 0x04 : 0x00;
    uint8_t mode = (s->io[R_LCDC] & LCDC_ON) ? s->ppu_mode : 0;
    return (uint8_t)(0x80 | (s->io[R_STAT] & 0x78) | lyc_eq | mode);
}

/* Recompute the STAT interrupt line; IRQ fires on its rising edge only
 * (interrupt "blocking" behaviour on real DMG). */
static void stat_update(GB *g)
{
    GBState *s = &g->s;
    if (!(s->io[R_LCDC] & LCDC_ON)) { s->stat_line = 0; return; }
    uint8_t en = s->io[R_STAT];
    uint8_t line =
        ((s->io[R_LY] == s->io[R_LYC]) && (en & 0x40)) ||
        (s->ppu_mode == 0 && (en & 0x08)) ||
        (s->ppu_mode == 1 && (en & 0x10)) ||
        (s->ppu_mode == 2 && (en & 0x20));
    if (line && !s->stat_line)
        s->io[R_IF] |= I_STAT;
    s->stat_line = line;
}

/* ------------------------------------------------------------------ */
/* Scanline renderer                                                    */
/* ------------------------------------------------------------------ */

/* EXPAND[b] spreads the 8 bits of b into 8 bytes, MSB first (bit 7 ->
 * byte 0), so a tile row's colors are EXPAND[lo] | EXPAND[hi] << 1:
 * one u64 holding 8 pixel values 0..3, leftmost pixel in byte 0.
 * Filled once at library load (deterministic, state-independent), so the
 * per-pixel-row hot path carries no init check. */
static uint64_t EXPAND[256];

__attribute__((constructor)) static void ppu_build_tables(void)
{
    for (int b = 0; b < 256; b++) {
        uint64_t v = 0;
        for (int k = 0; k < 8; k++)
            v |= (uint64_t)((b >> (7 - k)) & 1) << (k * 8);
        EXPAND[b] = v;
    }
}

static inline uint64_t tile_row_colors(const GBState *s, uint16_t addr)
{
    return EXPAND[s->vram[addr]] | (EXPAND[s->vram[addr + 1]] << 1);
}

static void render_scanline(GB *g)
{
    GBState *s = &g->s;
    uint8_t ly = s->io[R_LY];
    uint8_t lcdc = s->io[R_LCDC];
    uint8_t *row = s->framebuf + (size_t)ly * GB_W;
    uint8_t bgidx[GB_W];                   /* pre-palette BG/window color 0-3 */

    uint8_t bgp = s->io[R_BGP];

    uint8_t bgpal[4] = {
        (uint8_t)(bgp & 3), (uint8_t)((bgp >> 2) & 3),
        (uint8_t)((bgp >> 4) & 3), (uint8_t)((bgp >> 6) & 3),
    };

    /* --- background + window --- */
    if (lcdc & LCDC_BG_EN) {
        uint8_t scy = s->io[R_SCY], scx = s->io[R_SCX];
        uint8_t wy = s->io[R_WY], wx = s->io[R_WX];
        int win_active = (lcdc & LCDC_WIN_EN) && ly >= wy && wx <= 166;
        int win_start = win_active ? (wx < 7 ? 0 : wx - 7) : GB_W;
        if (win_start > GB_W) win_start = GB_W;
        int win_used = 0;

        /* Tile-batched: one map/tile-data fetch per run of up to 8 pixels.
         * A run never crosses a tile boundary, the window start, or the end
         * of the line, so pixel output matches the per-pixel definition. */
        int x = 0;
        while (x < GB_W) {
            uint16_t map;
            uint8_t px, py;
            int limit;
            if (win_active && x >= win_start) {
                map = (lcdc & LCDC_WIN_MAP) ? 0x1C00 : 0x1800;
                px = (uint8_t)(x - win_start);
                py = s->win_line;
                win_used = 1;
                limit = GB_W - x;
            } else {
                map = (lcdc & LCDC_BG_MAP) ? 0x1C00 : 0x1800;
                px = (uint8_t)(x + scx);
                py = (uint8_t)(ly + scy);
                limit = (win_start < GB_W ? win_start : GB_W) - x;
            }
            uint8_t tile = s->vram[map + (py >> 3) * 32 + (px >> 3)];
            uint16_t addr;
            if (lcdc & LCDC_TILEDATA)
                addr = (uint16_t)(tile * 16);
            else
                addr = (uint16_t)(0x1000 + (int8_t)tile * 16);
            addr += (py & 7) * 2;
            uint64_t colors = tile_row_colors(s, addr) >> ((px & 7) * 8);
            int run = 8 - (px & 7);
            if (run > limit) run = limit;
            if (run == 8) {
                /* Full, tile-aligned 8-pixel run (px&7==0): store the eight
                 * pre-palette colors in one write instead of eight. When BGP
                 * is the identity map (0xE4, the common in-game case) the
                 * shade equals the color, so the row copies too. */
                uint64_t cc = colors;
                memcpy(&bgidx[x], &cc, 8);
                if (bgp == 0xE4) {
                    memcpy(&row[x], &cc, 8);
                } else {
                    for (int k = 0; k < 8; k++, cc >>= 8)
                        row[x + k] = bgpal[cc & 3];
                }
            } else {
                for (int k = 0; k < run; k++, colors >>= 8) {
                    uint8_t c = (uint8_t)(colors & 3);
                    bgidx[x + k] = c;
                    row[x + k] = bgpal[c];
                }
            }
            x += run;
        }
        if (win_used) s->win_line++;
    } else {
        memset(bgidx, 0, sizeof(bgidx));
        memset(row, 0, GB_W);              /* BG disabled draws white */
    }

    /* --- sprites --- */
    if (!(lcdc & LCDC_OBJ_EN)) return;

    int h = (lcdc & LCDC_OBJ16) ? 16 : 8;
    uint8_t idxs[10];
    int n = 0;
    for (int i = 0; i < 40 && n < 10; i++) {  /* OAM scan: first 10 on line */
        int sy = s->oam[i * 4] - 16;
        if (ly >= sy && ly < sy + h) idxs[n++] = (uint8_t)i;
    }
    /* DMG priority: smaller X wins; ties broken by OAM order. Stable-sort by X. */
    for (int i = 1; i < n; i++) {
        uint8_t k = idxs[i];
        int j = i - 1;
        while (j >= 0 && s->oam[idxs[j] * 4 + 1] > s->oam[k * 4 + 1]) {
            idxs[j + 1] = idxs[j];
            j--;
        }
        idxs[j + 1] = k;
    }
    if (n == 0) return;                        /* no sprites on this line */

    /* Walk sprites in priority order; the first opaque pixel claims an x,
     * transparent pixels fall through to lower-priority sprites. Equivalent
     * to the per-pixel first-opaque-wins scan, one pass per sprite. */
    uint8_t claimed[GB_W];
    memset(claimed, 0, sizeof(claimed));
    for (int i = 0; i < n; i++) {
        const uint8_t *sp = s->oam + idxs[i] * 4;
        int sx = sp[1] - 8;
        uint8_t attr = sp[3];
        int line = ly - (sp[0] - 16);
        if (attr & 0x40) line = h - 1 - line;          /* Y flip */
        uint8_t tile = sp[2];
        if (h == 16) tile = (uint8_t)((tile & 0xFE) | (line >> 3));
        uint16_t addr = (uint16_t)(tile * 16 + (line & 7) * 2);
        uint64_t colors = tile_row_colors(s, addr);
        uint8_t pal = (attr & 0x10) ? s->io[R_OBP1] : s->io[R_OBP0];
        int x0 = sx < 0 ? 0 : sx;
        int x1 = sx + 8 > GB_W ? GB_W : sx + 8;
        for (int x = x0; x < x1; x++) {
            if (claimed[x]) continue;
            int k = (attr & 0x20) ? (7 - (x - sx)) : (x - sx);    /* X flip */
            uint8_t c = (uint8_t)((colors >> (k * 8)) & 3);
            if (c == 0) continue;                       /* transparent */
            claimed[x] = 1;
            if (!((attr & 0x80) && bgidx[x] != 0))
                row[x] = (pal >> (c * 2)) & 3;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Timing                                                               */
/* ------------------------------------------------------------------ */

static void set_mode(GB *g, uint8_t mode)
{
    g->s.ppu_mode = mode;
    stat_update(g);
}

static void inc_ly(GB *g)
{
    GBState *s = &g->s;
    uint8_t ly = s->io[R_LY] + 1;
    if (ly == 154) {
        ly = 0;
        s->win_line = 0;
    }
    s->io[R_LY] = ly;
    if (ly == 144) {
        s->io[R_IF] |= I_VBLANK;
        s->frames++;
        s->frame_done = 1;
        set_mode(g, 1);
    } else if (ly < 144) {
        set_mode(g, 2);
    } else {
        stat_update(g);                    /* LYC can match inside vblank */
    }
}

/* Dots remaining in the current mode segment (from line_dot to the next
 * boundary that runs logic). Kept out of ppu_tick's hot path as a helper so
 * both paths stay in sync. */
static inline uint32_t seg_until(const GBState *s)
{
    if (s->io[R_LY] >= 144)     return 456 - s->line_dot;
    if (s->line_dot < 80)       return 80 - s->line_dot;
    if (s->line_dot < 252)      return 252 - s->line_dot;
    return 456 - s->line_dot;
}

/* LCD off: PPU idle. Still produce frame pacing so run_frame() and host FPS
 * accounting keep working. Cold — games rarely run with the LCD disabled. */
__attribute__((noinline, cold))
static void ppu_tick_lcdoff(GB *g, uint32_t cycles)
{
    GBState *s = &g->s;
    s->line_dot += cycles;
    while (s->line_dot >= 456) {
        s->line_dot -= 456;
        if (++s->io[R_LY] >= 154) {
            s->io[R_LY] = 0;
            s->frames++;
            s->frame_done = 1;
        }
    }
}

/* Slow path: the step crosses at least one mode/line boundary. Walks every
 * boundary exactly (rendering, STAT edges, LY increments), so behaviour is
 * identical to a per-dot loop. */
__attribute__((noinline))
static void ppu_tick_cross(GB *g, uint32_t cycles)
{
    GBState *s = &g->s;
    for (;;) {
        uint8_t ly = s->io[R_LY];
        uint32_t until = seg_until(s);
        if (cycles < until) {
            s->line_dot += cycles;
            return;
        }
        s->line_dot += until;      /* land exactly on the next boundary */
        cycles -= until;

        if (s->line_dot >= 456) {
            s->line_dot = 0;
            if (ly < 144 && s->ppu_mode != 0)
                set_mode(g, 0);
            inc_ly(g);
        } else if (ly < 144 && s->line_dot == 80 && s->ppu_mode == 2) {
            set_mode(g, 3);
            if (g->render) render_scanline(g);
        } else if (ly < 144 && s->line_dot == 252 && s->ppu_mode == 3) {
            set_mode(g, 0);
        }
        if (cycles == 0) return;
    }
}

/* Hot path: tiny enough to inline into the CPU's per-instruction advance().
 * The overwhelmingly common case is a step that stays inside the current mode
 * segment — one compare and one add, no call. */
GB_INTERNAL void ppu_tick(GB *g, uint32_t cycles)
{
    GBState *s = &g->s;
    if (__builtin_expect(!(s->io[R_LCDC] & LCDC_ON), 0)) {
        ppu_tick_lcdoff(g, cycles);
        return;
    }
    if (__builtin_expect(cycles < seg_until(s), 1)) {
        s->line_dot += cycles;
        return;
    }
    ppu_tick_cross(g, cycles);
}

GB_INTERNAL void ppu_lcd_write(GB *g, uint8_t reg, uint8_t v)
{
    GBState *s = &g->s;
    switch (reg) {
    case R_LCDC: {
        uint8_t was_on = s->io[R_LCDC] & LCDC_ON;
        s->io[R_LCDC] = v;
        if (was_on && !(v & LCDC_ON)) {
            s->io[R_LY] = 0;
            s->line_dot = 0;
            s->ppu_mode = 0;
            s->stat_line = 0;
        } else if (!was_on && (v & LCDC_ON)) {
            s->io[R_LY] = 0;
            s->line_dot = 0;
            s->win_line = 0;
            set_mode(g, 2);
        }
        break;
    }
    case R_STAT:
        s->io[R_STAT] = v & 0x78;
        stat_update(g);
        break;
    case R_LY:
        break;                              /* read-only */
    case R_LYC:
        s->io[R_LYC] = v;
        stat_update(g);
        break;
    }
}
