//! Run-length connected-region core shared by both extraction paths.
//!
//! A single raster sweep turns a packed detection bit mask
//! ([`sweep_runs_mask`]) — where work scales with the number of runs rather
//! than pixels — into horizontal runs, merging 8-connected runs across rows
//! with union-find. The pixel-by-pixel predicate form ([`sweep_runs`]) is
//! kept as the test reference; both produce identical [`RunRegions`] for
//! the same lit set.
//! Region payloads live with the callers: the fast path computes moments
//! from the run lists in a post-pass (lit pixels are ≪1% of the image, so
//! the second touch is nearly free), while the quality path runs its
//! annulus/moment/deblend stages per region. This replaces the generic
//! mask → labels connected-component labeling, which materialized a u8 mask
//! and a u32 labels buffer (~10 MB at 2 Mpix) and required a per-pixel label
//! test in every downstream stage.
//!
//! The row packers ([`pack_lit_row`], [`pack_above_row`]) build that bit mask
//! one image row at a time with a vectorizable compare and a byte→bit fold.

use std::ops::Range;

use super::par;

/// A horizontal run of lit pixels: columns `c0..=c1` of `row`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Run {
    pub row: u32,
    pub c0: u32,
    pub c1: u32,
}

impl Run {
    #[inline]
    pub(super) fn len(&self) -> usize {
        (self.c1 - self.c0 + 1) as usize
    }
}

/// Result of [`sweep_runs`] / [`sweep_runs_mask`]: the runs in creation (row-major) order, each
/// run's dense region id, and the region count. Region ids are assigned in
/// order of first appearance, so iterating regions is deterministic.
pub(super) struct RunRegions {
    pub runs: Vec<Run>,
    pub region_of_run: Vec<u32>,
    pub n_regions: usize,
}

/// Pixel count and bounding box of one region's run list.
#[derive(Clone, Copy, Debug)]
pub(super) struct RegionExtent {
    pub npix: usize,
    pub min_row: usize,
    pub max_row: usize,
    pub min_col: usize,
    pub max_col: usize,
}

impl RegionExtent {
    /// Border gate shared by both extraction paths: `false` when the bounding
    /// box comes within `margin` pixels of any edge of a `w`×`h` image (a
    /// star cut by the frame edge has a truncated PSF and a CoM biased
    /// toward the interior — a plausible but wrong position). `margin == 0`
    /// disables the gate. The `min_*` tests come first so a margin larger
    /// than the image cannot underflow `h - margin`.
    #[inline]
    pub(super) fn clear_of_border(&self, margin: usize, w: usize, h: usize) -> bool {
        margin == 0
            || !(self.min_row < margin
                || self.min_col < margin
                || self.max_row >= h - margin
                || self.max_col >= w - margin)
    }
}

impl RunRegions {
    /// Pixel count and bounding box of the region whose (row-major) run
    /// indices are `region_runs` (a slice of the `order` returned by
    /// [`Self::group_by_region`]).
    pub(super) fn extent(&self, region_runs: &[u32]) -> RegionExtent {
        let mut e = RegionExtent {
            npix: 0,
            min_row: usize::MAX,
            max_row: 0,
            min_col: usize::MAX,
            max_col: 0,
        };
        for &i in region_runs {
            let run = self.runs[i as usize];
            e.npix += run.len();
            e.min_row = e.min_row.min(run.row as usize);
            e.max_row = e.max_row.max(run.row as usize);
            e.min_col = e.min_col.min(run.c0 as usize);
            e.max_col = e.max_col.max(run.c1 as usize);
        }
        e
    }

    /// Group runs by region: `order[offsets[k] as usize..offsets[k + 1] as
    /// usize]` are the (ascending, hence row-major) run indices of region
    /// `k`. Counting sort — O(runs + regions).
    pub(super) fn group_by_region(&self) -> (Vec<u32>, Vec<u32>) {
        let mut counts = vec![0u32; self.n_regions + 1];
        for &reg in &self.region_of_run {
            counts[reg as usize + 1] += 1;
        }
        for k in 1..counts.len() {
            counts[k] += counts[k - 1];
        }
        let offsets = counts.clone();
        let mut cursor = counts;
        let mut order = vec![0u32; self.runs.len()];
        for (i, &reg) in self.region_of_run.iter().enumerate() {
            order[cursor[reg as usize] as usize] = i as u32;
            cursor[reg as usize] += 1;
        }
        (offsets, order)
    }
}

/// Run-length 8-connected labeling over `lit(row, col)` — the pixel-by-pixel
/// reference that [`sweep_runs_mask`] (and its banded variant) is tested
/// against; both extraction paths use the mask form.
///
/// The predicate is called exactly once per pixel in raster order, so callers
/// may keep per-row state inside the closure (e.g. a row-blended background).
#[cfg(test)]
pub(super) fn sweep_runs(
    w: usize,
    h: usize,
    mut lit: impl FnMut(usize, usize) -> bool,
) -> RunRegions {
    let mut sweep = Sweep::new();
    for r in 0..h {
        let mut start: Option<usize> = None;
        for c in 0..w {
            if lit(r, c) {
                if start.is_none() {
                    start = Some(c);
                }
            } else if let Some(s) = start.take() {
                sweep.push_run(r, s, c - 1);
            }
        }
        if let Some(s) = start.take() {
            sweep.push_run(r, s, w - 1);
        }
        sweep.end_row();
    }
    sweep.finish()
}

/// Run-length 8-connected labeling over a packed bit mask.
///
/// `mask` holds `words_per_row` `u64` words per image row (`h` rows, row
/// `r` at `mask[r * words_per_row..]`); column `c` is lit when bit `c % 64`
/// of word `c / 64` is set. Bits at or beyond `w` in the last word of a row
/// are ignored, so `w` need not be a multiple of 64. Runs are read off the
/// words with `trailing_zeros` / `trailing_ones` (a run that reaches the top
/// bit of a word is carried into the next), so the cost is proportional to
/// the number of runs, not pixels. Yields exactly the [`RunRegions`] that
/// [`sweep_runs`] gives for the same lit set — runs in row-major order,
/// same connectivity, same first-appearance region ids.
pub(super) fn sweep_runs_mask(
    w: usize,
    h: usize,
    words_per_row: usize,
    mask: &[u64],
) -> RunRegions {
    let geom = MaskGeometry::new(w, h, words_per_row, mask.len());
    let mut sweep = Sweep::new();
    for (r, row) in mask.chunks_exact(words_per_row).take(h).enumerate() {
        geom.push_row(&mut sweep, r, row);
        sweep.end_row();
    }
    sweep.finish()
}

/// [`sweep_runs_mask`] with the rows labeled in independent bands of
/// `band_rows` rows (one task per band — multi-threaded under the `parallel`
/// feature), then stitched: the bands' run lists are concatenated in row
/// order with their provisional labels offset, each band's first row is
/// merged with its predecessor's last row exactly as consecutive rows are
/// inside a band, and the shared root-compaction pass assigns region ids.
/// The runs are the same row-major list, the union-find partition is the
/// same set of connected components, and ids are assigned in first-
/// appearance order over that list — so the result is identical to
/// [`sweep_runs_mask`] regardless of `band_rows` or thread count.
///
/// The CCL path uses it under `parallel`; the sequential build keeps the
/// plain sweep (the function stays compiled so its tests run in both).
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
pub(super) fn sweep_runs_mask_banded(
    w: usize,
    h: usize,
    words_per_row: usize,
    mask: &[u64],
    band_rows: usize,
) -> RunRegions {
    let geom = MaskGeometry::new(w, h, words_per_row, mask.len());
    let band_rows = band_rows.max(1);
    let n_bands = h.div_ceil(band_rows);

    /// One band's labeling: its `Sweep` (runs, local parents, and — once
    /// every row is in — `prev_start`, where its last row's runs begin)
    /// plus the number of runs in its first row.
    struct Band {
        sweep: Sweep,
        first_len: usize,
    }
    let bands: Vec<Band> = par::map_indices(n_bands, |b| {
        let y0 = b * band_rows;
        let y1 = (y0 + band_rows).min(h);
        let mut sweep = Sweep::new();
        let mut first_len = 0;
        for r in y0..y1 {
            geom.push_row(
                &mut sweep,
                r,
                &mask[r * words_per_row..(r + 1) * words_per_row],
            );
            if r == y0 {
                first_len = sweep.runs.len();
            }
            sweep.end_row();
        }
        Band { sweep, first_len }
    });

    // Concatenate (run order stays row-major) with labels offset by each
    // band's first global run index, stitching every band boundary: a
    // band's first row against its predecessor's last row, both addressed
    // as index ranges of the concatenated run list.
    let total: usize = bands.iter().map(|b| b.sweep.runs.len()).sum();
    let mut all = Sweep::new();
    all.runs.reserve_exact(total);
    all.parents.reserve_exact(total);
    let mut prev_last = 0..0;
    for band in bands {
        let off = all.runs.len();
        all.runs.extend_from_slice(&band.sweep.runs);
        all.parents
            .extend(band.sweep.parents.iter().map(|&p| p + off as u32));
        merge_rows(
            &mut all.parents,
            &all.runs,
            off..off + band.first_len,
            prev_last,
        );
        prev_last = off + band.sweep.prev_start..all.runs.len();
    }
    all.finish()
}

/// Per-image constants of a packed mask: the words that hold real columns
/// and the mask that clears a row's padding bits (see [`sweep_runs_mask`]).
#[derive(Clone, Copy)]
struct MaskGeometry {
    w: usize,
    n_words: usize,
    tail_mask: u64,
}

impl MaskGeometry {
    fn new(w: usize, h: usize, words_per_row: usize, mask_len: usize) -> Self {
        let n_words = w.div_ceil(64);
        assert!(
            words_per_row >= n_words && mask_len >= words_per_row * h,
            "bit mask too small for a {w}x{h} image ({words_per_row} words/row, {mask_len} words)"
        );
        // Clears the padding bits of a row's last word (no-op when 64 | w).
        let tail_mask = if w.is_multiple_of(64) {
            u64::MAX
        } else {
            (1u64 << (w % 64)) - 1
        };
        Self {
            w,
            n_words,
            tail_mask,
        }
    }

    /// Push the runs of mask row `r` (its `words_per_row` words, of which
    /// the first `n_words` are read) onto `sweep`, left to right.
    #[inline]
    fn push_row(&self, sweep: &mut Sweep, r: usize, row: &[u64]) {
        let n_words = self.n_words;
        // Start column of a run that reached the top bit of the previous word.
        let mut open: Option<usize> = None;
        for (wi, &word) in row[..n_words].iter().enumerate() {
            let base = wi * 64;
            let mut bits = if wi + 1 == n_words {
                word & self.tail_mask
            } else {
                word
            };
            // Bits of this word already consumed (shifted out of `bits`).
            let mut pos = 0u32;
            if let Some(s) = open {
                let ones = bits.trailing_ones();
                if ones == 64 {
                    continue; // whole word lit: the run stays open
                }
                sweep.push_run(r, s, base + ones as usize - 1);
                open = None;
                bits >>= ones;
                pos = ones;
            }
            while bits != 0 {
                let tz = bits.trailing_zeros();
                bits >>= tz;
                pos += tz;
                let start = base + pos as usize;
                let ones = bits.trailing_ones();
                if pos + ones >= 64 {
                    open = Some(start); // reaches the top bit: may continue
                    break;
                }
                sweep.push_run(r, start, start + ones as usize - 1);
                bits >>= ones;
                pos += ones;
            }
        }
        if let Some(s) = open {
            sweep.push_run(r, s, self.w - 1);
        }
    }
}

/// Pack one image row's detection mask: bit `c % 64` of `words[c / 64]` is
/// set when `row[c]` is finite and exceeds `thr[c]`. `(p > t) & (p < +∞)` is
/// exactly `p.is_finite() && p > t` (NaN fails both compares, −∞ fails the
/// first, +∞ the second) but branch-free: the compares fill a 64-byte
/// scratch (a vectorizable loop), then each 8 bytes fold to 8 bits with the
/// multiply-shift trick. Padding bits at or beyond `row.len()` are zero.
#[inline]
pub(super) fn pack_lit_row(row: &[f32], thr: &[f32], words: &mut [u64]) {
    debug_assert_eq!(row.len(), thr.len());
    debug_assert_eq!(words.len(), row.len().div_ceil(64));
    for (wi, word) in words.iter_mut().enumerate() {
        let c0 = wi * 64;
        let c1 = (c0 + 64).min(row.len());
        let mut bytes = [0u8; 64];
        for ((m, &p), &t) in bytes.iter_mut().zip(&row[c0..c1]).zip(&thr[c0..c1]) {
            *m = ((p > t) & (p < f32::INFINITY)) as u8;
        }
        *word = pack_bytes(&bytes);
    }
}

/// Pack one image row against a single threshold: bit `c % 64` of
/// `words[c / 64]` is set exactly when `row[c] > thr` (the plain `>`
/// predicate — a NaN is never lit, `+∞` always is). Same packing as
/// [`pack_lit_row`]; padding bits are zero.
#[inline]
pub(super) fn pack_above_row(row: &[f32], thr: f32, words: &mut [u64]) {
    debug_assert_eq!(words.len(), row.len().div_ceil(64));
    for (wi, word) in words.iter_mut().enumerate() {
        let c0 = wi * 64;
        let c1 = (c0 + 64).min(row.len());
        let mut bytes = [0u8; 64];
        for (m, &p) in bytes.iter_mut().zip(&row[c0..c1]) {
            *m = (p > thr) as u8;
        }
        *word = pack_bytes(&bytes);
    }
}

/// Fold 64 bytes, each 0 or 1, into a u64 with byte `i` → bit `i`.
#[inline]
fn pack_bytes(bytes: &[u8; 64]) -> u64 {
    let mut acc = 0u64;
    for (k, chunk) in bytes.as_chunks::<8>().0.iter().enumerate() {
        let x = u64::from_le_bytes(*chunk);
        // Each byte is 0 or 1; the multiply gathers the low bit of each
        // byte into the top byte, lowest byte → lowest bit.
        acc |= (x.wrapping_mul(0x0102_0408_1020_4080) >> 56) << (8 * k);
    }
    acc
}

/// Union-find state shared by the sweeps: one provisional label per run
/// (label id == run index), rows merged with their predecessor as they
/// complete, roots compacted to dense region ids at the end.
struct Sweep {
    parents: Vec<u32>,
    runs: Vec<Run>,
    /// Where the previous and the current row begin in `runs`. A row's runs
    /// are pushed consecutively, so the previous row is
    /// `runs[prev_start..cur_start]` and the current one `runs[cur_start..]`
    /// — no per-row copies of the runs are kept.
    prev_start: usize,
    cur_start: usize,
}

impl Sweep {
    fn new() -> Self {
        Self {
            parents: Vec::new(),
            runs: Vec::new(),
            prev_start: 0,
            cur_start: 0,
        }
    }

    /// Append the run `c0..=c1` of `row`. Rows must be pushed in order and
    /// runs left to right within a row.
    #[inline]
    fn push_run(&mut self, row: usize, c0: usize, c1: usize) {
        let label = self.runs.len() as u32;
        self.parents.push(label);
        self.runs.push(Run {
            row: row as u32,
            c0: c0 as u32,
            c1: c1 as u32,
        });
    }

    /// Merge the current row's runs with 8-connected runs of the previous
    /// row, then make it the previous row.
    fn end_row(&mut self) {
        let end = self.runs.len();
        merge_rows(
            &mut self.parents,
            &self.runs,
            self.cur_start..end,
            self.prev_start..self.cur_start,
        );
        self.prev_start = self.cur_start;
        self.cur_start = end;
    }

    /// Resolve roots and compact them to dense region ids in
    /// first-appearance (row-major) order.
    fn finish(self) -> RunRegions {
        let Self {
            mut parents, runs, ..
        } = self;
        let mut region_of_run = vec![0u32; runs.len()];
        let mut id_of_root = vec![u32::MAX; parents.len()];
        let mut n_regions = 0usize;
        for (i, reg) in region_of_run.iter_mut().enumerate() {
            let root = find(&mut parents, i as u32) as usize;
            if id_of_root[root] == u32::MAX {
                id_of_root[root] = n_regions as u32;
                n_regions += 1;
            }
            *reg = id_of_root[root];
        }
        RunRegions {
            runs,
            region_of_run,
            n_regions,
        }
    }
}

/// Union every current-row run with each 8-connected previous-row run.
/// `cur` and `prev` are the two rows' index ranges in `runs` (a run's label
/// is its index); each row is column-sorted, so a two-pointer sweep finds
/// all overlaps.
fn merge_rows(parents: &mut [u32], runs: &[Run], cur: Range<usize>, prev: Range<usize>) {
    let (mut i, mut j) = (cur.start, prev.start);
    while i < cur.end && j < prev.end {
        let (cs, ce) = (runs[i].c0, runs[i].c1);
        let (ps, pe) = (runs[j].c0, runs[j].c1);
        if ce + 1 < ps {
            i += 1; // current run ends left of (and not adjacent to) prev
        } else if pe + 1 < cs {
            j += 1; // prev run ends left of current
        } else {
            union(parents, i as u32, j as u32);
            if ce < pe {
                i += 1;
            } else {
                j += 1;
            }
        }
    }
}

/// Union-find root with path halving.
fn find(parents: &mut [u32], mut x: u32) -> u32 {
    while parents[x as usize] != x {
        let p = parents[x as usize];
        parents[x as usize] = parents[p as usize]; // halve
        x = parents[x as usize];
    }
    x
}

/// Link the roots of two labels.
fn union(parents: &mut [u32], a: u32, b: u32) {
    let ra = find(parents, a);
    let rb = find(parents, b);
    if ra != rb {
        parents[ra as usize] = rb;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn regions_of(mask: &[&[u8]]) -> RunRegions {
        let h = mask.len();
        let w = mask[0].len();
        sweep_runs(w, h, |r, c| mask[r][c] != 0)
    }

    /// Pack a byte mask into `words_per_row` words per row. Padding bits
    /// (at or beyond `w`) and padding words are filled with garbage to
    /// prove the sweep ignores them.
    fn pack(mask: &[&[u8]], words_per_row: usize) -> Vec<u64> {
        let w = mask[0].len();
        let mut out = vec![u64::MAX; words_per_row * mask.len()];
        for (r, row) in mask.iter().enumerate() {
            for wi in 0..words_per_row {
                let mut word = 0u64;
                for b in 0..64 {
                    let c = wi * 64 + b;
                    let lit = if c < w { row[c] != 0 } else { (c + r) % 3 == 0 };
                    word |= (lit as u64) << b;
                }
                out[r * words_per_row + wi] = word;
            }
        }
        out
    }

    fn regions_of_mask(mask: &[&[u8]]) -> RunRegions {
        let h = mask.len();
        let w = mask[0].len();
        let wpr = w.div_ceil(64);
        sweep_runs_mask(w, h, wpr, &pack(mask, wpr))
    }

    fn assert_same(a: &RunRegions, b: &RunRegions) {
        assert_eq!(a.n_regions, b.n_regions);
        assert_eq!(a.runs, b.runs);
        assert_eq!(a.region_of_run, b.region_of_run);
    }

    fn check_basic(rr: RunRegions) {
        assert_eq!(rr.n_regions, 4);
        let (offsets, order) = rr.group_by_region();
        // Region of the first run (top-left pair) contains 2 runs.
        let first = rr.region_of_run[0] as usize;
        assert_eq!((offsets[first + 1] - offsets[first]), 2);
        // Total pixels across all runs.
        let total: usize = order.iter().map(|&i| rr.runs[i as usize].len()).sum();
        assert_eq!(total, 7);
    }

    const BASIC: [&[u8]; 4] = [
        &[1, 1, 0, 0, 1],
        &[0, 1, 0, 1, 0], // diagonal touch joins col-4/row-0 with col-3/row-1
        &[0, 0, 0, 0, 0],
        &[1, 0, 0, 0, 1],
    ];

    #[test]
    fn test_sweep_runs_basic() {
        check_basic(regions_of(&BASIC));
    }

    #[test]
    fn test_sweep_runs_mask_basic() {
        check_basic(regions_of_mask(&BASIC));
        assert_same(&regions_of(&BASIC), &regions_of_mask(&BASIC));
    }

    #[test]
    fn test_sweep_runs_full_row_and_empty() {
        let rr = regions_of(&[&[1, 1, 1], &[1, 1, 1]]);
        assert_eq!(rr.n_regions, 1);
        assert_eq!(rr.runs.len(), 2);

        let rr = regions_of(&[&[0, 0], &[0, 0]]);
        assert_eq!(rr.n_regions, 0);
        assert!(rr.runs.is_empty());
    }

    #[test]
    fn test_sweep_runs_mask_full_row_and_empty() {
        let rr = regions_of_mask(&[&[1, 1, 1], &[1, 1, 1]]);
        assert_eq!(rr.n_regions, 1);
        assert_eq!(rr.runs.len(), 2);

        let rr = regions_of_mask(&[&[0, 0], &[0, 0]]);
        assert_eq!(rr.n_regions, 0);
        assert!(rr.runs.is_empty());
    }

    #[test]
    fn test_sweep_runs_mask_word_boundaries() {
        // Runs ending exactly at, starting exactly at, and spanning word
        // boundaries; a fully lit word inside a run; runs touching the right
        // image edge both with 64 | w and with w % 64 != 0.
        for w in [64usize, 65, 127, 128, 129, 200] {
            let row: Vec<u8> = (0..w)
                .map(|c| {
                    (c == 0
                        || (60..=63).contains(&c)
                        || ((64..=70).contains(&c) && w > 66)
                        || (120..w).contains(&c)
                        || ((30..=191).contains(&c) && w == 200)) as u8
                })
                .collect();
            let blank = vec![0u8; w];
            let mask: [&[u8]; 3] = [&row, &blank, &row];
            assert_same(&regions_of(&mask), &regions_of_mask(&mask));
        }
        // Every pixel lit: one run per row spanning several words.
        let row = vec![1u8; 300];
        let mask: [&[u8]; 2] = [&row, &row];
        let rr = regions_of_mask(&mask);
        assert_same(&regions_of(&mask), &rr);
        assert_eq!(rr.n_regions, 1);
        assert_eq!(
            rr.runs[0],
            Run {
                row: 0,
                c0: 0,
                c1: 299
            }
        );
    }

    #[test]
    fn test_sweep_runs_mask_matches_sweep_runs_random() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let sizes = [
            (1usize, 1usize),
            (7, 3),
            (63, 5),
            (64, 4),
            (65, 6),
            (130, 9),
            (2136, 3),
            (257, 40),
        ];
        for &(w, h) in &sizes {
            for density in [1u64, 8, 32, 60] {
                let rows: Vec<Vec<u8>> = (0..h)
                    .map(|_| (0..w).map(|_| (next() % 64 < density) as u8).collect())
                    .collect();
                let mask: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
                let a = regions_of(&mask);
                let b = regions_of_mask(&mask);
                assert_same(&a, &b);
                // Padding words beyond `w.div_ceil(64)` are ignored too.
                let wpr = w.div_ceil(64) + 2;
                let packed = pack(&mask, wpr);
                let c = sweep_runs_mask(w, h, wpr, &packed);
                assert_same(&a, &c);
                // Banded labeling is identical for every band height,
                // including bands that split the image unevenly (h not a
                // multiple of the band), single-row bands, and one band.
                for band_rows in [1usize, 2, 3, 7, 16, 64, h, h + 5] {
                    let d = sweep_runs_mask_banded(w, h, wpr, &packed, band_rows);
                    assert_same(&a, &d);
                }
            }
        }
    }

    #[test]
    fn test_sweep_runs_mask_banded_stitches_boundaries() {
        // Regions that cross band boundaries: a vertical bar through every
        // band, a diagonal staircase (8-connected only through the corner
        // touch at each boundary), and a bar that ends exactly at a boundary.
        let (w, h) = (70usize, 20usize);
        let rows: Vec<Vec<u8>> = (0..h)
            .map(|r| {
                (0..w)
                    .map(|c| (c == 5 || c == 20 + r || (c == 60 && r < 8)) as u8)
                    .collect()
            })
            .collect();
        let mask: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
        let a = regions_of(&mask);
        assert_eq!(a.n_regions, 3);
        let wpr = w.div_ceil(64);
        let packed = pack(&mask, wpr);
        for band_rows in [1usize, 2, 4, 8, 9, 20, 32] {
            assert_same(&a, &sweep_runs_mask_banded(w, h, wpr, &packed, band_rows));
        }
    }
    #[test]
    fn test_pack_lit_row_matches_predicate() {
        let w = 2136usize; // not a multiple of 64
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let thr: Vec<f32> = (0..w).map(|c| c as f32 * 0.5).collect();
        let mut row: Vec<f32> = (0..w)
            .map(|c| c as f32 * 0.5 + (next() % 5) as f32 - 2.0)
            .collect();
        // Non-finite values must never be lit, whatever the threshold.
        row[3] = f32::NAN;
        row[64] = f32::INFINITY;
        row[65] = f32::NEG_INFINITY;
        row[w - 1] = f32::INFINITY;
        let mut words = vec![u64::MAX; w.div_ceil(64)];
        pack_lit_row(&row, &thr, &mut words);
        for c in 0..w {
            let expect = row[c].is_finite() && row[c] > thr[c];
            let got = words[c / 64] >> (c % 64) & 1 == 1;
            assert_eq!(got, expect, "column {c}");
        }
        // Padding bits are zero.
        assert_eq!(words[w / 64] >> (w % 64), 0);
    }

    #[test]
    fn test_pack_above_row_matches_predicate() {
        let w = 2136usize;
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let thr = 2.0_f32;
        let mut row: Vec<f32> = (0..w).map(|_| (next() % 5) as f32).collect();
        // Plain `>`: NaN never lit, +inf lit, -inf not.
        row[3] = f32::NAN;
        row[64] = f32::INFINITY;
        row[65] = f32::NEG_INFINITY;
        row[w - 1] = f32::INFINITY;
        let mut words = vec![u64::MAX; w.div_ceil(64)];
        pack_above_row(&row, thr, &mut words);
        for c in 0..w {
            let expect = row[c] > thr;
            let got = words[c / 64] >> (c % 64) & 1 == 1;
            assert_eq!(got, expect, "column {c}");
        }
        assert_eq!(words[w / 64] >> (w % 64), 0);
    }
}
