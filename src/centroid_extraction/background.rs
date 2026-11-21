//! Coarse block statistics of the detection image: a block-median level
//! grid, bilinearly interpolated, and a per-block noise σ.

use super::{median_f32, midpoint_f32, par};
use crate::prelude::*;

/// A coarse grid of per-block values with bilinear interpolation between
/// block centers.
///
/// The image is divided into `block × block` tiles. Interpolation
/// **linearly extrapolates** beyond the outermost centers — clamping would
/// leave any gradient un-modeled across the outer `block/2` border band,
/// which lights up as border false positives at tight thresholds.
pub(super) struct BlockGrid {
    pub grid: Vec<f32>,
    pub nx: usize,
    pub ny: usize,
    pub block: usize,
    /// Row-independent half of the interpolation for every image column.
    cols: ColPlan,
}

/// Per-column interpolation parameters of a [`BlockGrid`] for a `w`-wide
/// image — the row-independent half of the bilinear blend, computed once
/// and grouped into segments of constant `(bx0, bx1)`, so blending a whole
/// row is a straight multiply-add loop with no per-pixel divide / floor /
/// clamp. The single-pixel accessors read the same per-column entries, so
/// every way of evaluating the surface shares one set of weights.
struct ColPlan {
    /// Column blend weight `fx` and its complement `1 - fx`.
    fx: Vec<f32>,
    omfx: Vec<f32>,
    /// The two grid columns each image column blends.
    bx0: Vec<u32>,
    bx1: Vec<u32>,
    /// `(c_start, c_end_exclusive, bx0, bx1)` — maximal column ranges that
    /// share the same pair of grid columns.
    segs: Vec<(usize, usize, usize, usize)>,
}

impl ColPlan {
    fn new(nx: usize, block: usize, w: usize) -> Self {
        let mut plan = Self {
            fx: Vec::with_capacity(w),
            omfx: Vec::with_capacity(w),
            bx0: Vec::with_capacity(w),
            bx1: Vec::with_capacity(w),
            segs: Vec::new(),
        };
        for c in 0..w {
            let (bx0, bx1, f) = axis_params(nx, block, c);
            plan.fx.push(f);
            plan.omfx.push(1.0 - f);
            plan.bx0.push(bx0 as u32);
            plan.bx1.push(bx1 as u32);
            match plan.segs.last_mut() {
                Some(seg) if seg.2 == bx0 && seg.3 == bx1 => seg.1 = c + 1,
                _ => plan.segs.push((c, c + 1, bx0, bx1)),
            }
        }
        plan
    }
}

/// Interpolation parameters along one axis for pixel index `i`: the two
/// grid cells to blend and the (unclamped — extrapolating) blend weight.
#[inline]
fn axis_params(n: usize, block: usize, i: usize) -> (usize, usize, f32) {
    if n == 1 {
        return (0, 0, 0.0);
    }
    let bf = (i as f32 - block as f32 / 2.0) / block as f32;
    let b0 = (bf.floor() as isize).clamp(0, n as isize - 2) as usize;
    (b0, b0 + 1, bf - b0 as f32)
}

/// Give every block without data the mean of its already-filled
/// 8-neighbours, growing outward from the blocks that have data until the
/// grid is full (each pass reads only the previous pass's state, so the
/// result does not depend on visiting order). A grid with no data at all is
/// left as it is.
fn fill_empty_blocks(grid: &mut [f32], mut has_data: Vec<bool>, nx: usize, ny: usize) {
    if has_data.iter().all(|&d| d) || !has_data.iter().any(|&d| d) {
        return;
    }
    loop {
        let mut filled: Vec<(usize, f32)> = Vec::new();
        for by in 0..ny {
            for bx in 0..nx {
                if has_data[by * nx + bx] {
                    continue;
                }
                let (mut sum, mut n) = (0.0_f64, 0u32);
                for y in by.saturating_sub(1)..(by + 2).min(ny) {
                    for x in bx.saturating_sub(1)..(bx + 2).min(nx) {
                        if has_data[y * nx + x] {
                            sum += grid[y * nx + x] as f64;
                            n += 1;
                        }
                    }
                }
                if n > 0 {
                    filled.push((by * nx + bx, (sum / n as f64) as f32));
                }
            }
        }
        if filled.is_empty() {
            return;
        }
        for (i, v) in filled {
            grid[i] = v;
            has_data[i] = true;
        }
    }
}

/// Visit the block subsample lattice of block row `by`: `f(bx, y, x)` for
/// every sampled pixel, rows top to bottom, columns left to right.
///
/// The lattice is phase-staggered (row `k` of a block starts at column
/// offset `k mod stride`): every column-residue class is sampled equally,
/// so column-periodic structure — CMOS fixed-pattern noise, Bayer residue —
/// does not alias.
#[inline]
fn for_each_sample(
    (w, h): (usize, usize),
    block: usize,
    stride: usize,
    by: usize,
    mut f: impl FnMut(usize, usize, usize),
) {
    let (y0, y1) = (by * block, ((by + 1) * block).min(h));
    let mut phase = 0usize;
    let mut y = y0;
    while y < y1 {
        let mut x = phase;
        while x < w {
            f(x / block, y, x);
            x += stride;
        }
        phase = (phase + 1) % stride;
        y += stride;
    }
}

/// Background level and noise of an image, on a block grid.
pub(super) struct BlockStats {
    /// Block-median background level.
    pub level: BlockGrid,
    /// Per-block noise σ (median-smoothed over 3×3 blocks, floored).
    pub sigma: Vec<f32>,
    /// Median of the per-block σ.
    pub sigma_global: f32,
}

impl BlockStats {
    /// Measure `s` (a `w × h` row-major image) on `block × block` tiles.
    ///
    /// Two passes over a subsample lattice (about 64 samples per block):
    ///
    /// 1. Block medians of the sampled values give a first level grid.
    /// 2. Residuals against that grid's bilinear surface give, per block,
    ///    a level correction (their median) and the noise σ.
    ///
    /// The second pass is what makes the level accurate. A median taken
    /// over raw values is only as good as the spread of those values, and
    /// on a background gradient that spread is the gradient across the
    /// block, not the noise — far too coarse for a threshold on a smoothed
    /// image, whose σ is several times smaller than the raw frame's.
    /// Residuals against the (correctly sloped) first-pass surface have the
    /// gradient removed, so their median pins the level to ~1.25σ/√n.
    ///
    /// σ is the RMS of the residuals at or below that median: stars only
    /// push residuals up, so the lower half is the uncontaminated noise
    /// (`E[(v − m)² | v ≤ m] = σ²` for a Gaussian). It is median-smoothed
    /// over 3×3 blocks, then floored at half the global value and at
    /// `sigma_floor` (see [`quantization_step`]).
    ///
    /// Non-finite samples are skipped; a block with none takes its
    /// neighbours' level. A `block` larger than the image is one block the
    /// size of the image (the subsample spacing follows the block size, so
    /// an oversized block would otherwise be sampled at a single pixel).
    pub(super) fn measure(s: &[f32], w: usize, h: usize, block: usize, sigma_floor: f32) -> Self {
        let block = block.clamp(1, w.max(h));
        let nx = w.div_ceil(block);
        let ny = h.div_ceil(block);
        let stride = (block / 8).max(1);
        let cap = (block / stride + 1).pow(2);

        // ── Pass 1: block medians ──
        let rows: Vec<Vec<(f32, bool)>> = par::map_indices(ny, |by| {
            let mut vals: Vec<Vec<f32>> = (0..nx).map(|_| Vec::with_capacity(cap)).collect();
            for_each_sample((w, h), block, stride, by, |bx, y, x| {
                let v = s[y * w + x];
                if v.is_finite() {
                    vals[bx].push(v);
                }
            });
            vals.iter_mut()
                .map(|v| (midpoint_f32(v), !v.is_empty()))
                .collect()
        });
        let flat = rows.iter().flatten();
        let mut grid: Vec<f32> = flat.clone().map(|&(m, _)| m).collect();
        let has_data: Vec<bool> = flat.map(|&(_, d)| d).collect();
        fill_empty_blocks(&mut grid, has_data, nx, ny);
        let mut level = BlockGrid::new(grid, nx, ny, block, w);

        // ── Pass 2: residual median and lower-half RMS ──
        let rows: Vec<Vec<(f32, f32)>> = par::map_indices(ny, |by| {
            let mut vals: Vec<Vec<f32>> = (0..nx).map(|_| Vec::with_capacity(cap)).collect();
            let mut row_blend = vec![0.0_f32; nx];
            let mut y_cur = usize::MAX;
            for_each_sample((w, h), block, stride, by, |bx, y, x| {
                if y != y_cur {
                    level.blend_row(level.row_params(y), &mut row_blend);
                    y_cur = y;
                }
                let r = s[y * w + x] - level.value_in_row(&row_blend, x);
                if r.is_finite() {
                    vals[bx].push(r);
                }
            });
            vals.iter_mut()
                .map(|v| {
                    if v.is_empty() {
                        return (0.0, f32::NAN);
                    }
                    let med = median_f32(v);
                    let (mut sq, mut n) = (0.0_f64, 0u32);
                    for &r in v.iter() {
                        if r <= med {
                            sq += ((r - med) as f64).powi(2);
                            n += 1;
                        }
                    }
                    (med, (sq / n.max(1) as f64).sqrt() as f32)
                })
                .collect()
        });
        let flat: Vec<(f32, f32)> = rows.into_iter().flatten().collect();
        for (g, &(delta, _)) in level.grid.iter_mut().zip(&flat) {
            *g += delta;
        }

        let raw: Vec<f32> = flat.iter().map(|&(_, sg)| sg).collect();
        let mut finite: Vec<f32> = raw.iter().copied().filter(|v| v.is_finite()).collect();
        let sigma_global = median_f32(&mut finite).max(sigma_floor);
        let mut sigma = vec![sigma_global; nx * ny];
        let mut nb = [0.0_f32; 9];
        for by in 0..ny {
            for bx in 0..nx {
                let mut m = 0;
                for yy in by.saturating_sub(1)..(by + 2).min(ny) {
                    for xx in bx.saturating_sub(1)..(bx + 2).min(nx) {
                        let v = raw[yy * nx + xx];
                        if v.is_finite() {
                            nb[m] = v;
                            m += 1;
                        }
                    }
                }
                if m > 0 {
                    sigma[by * nx + bx] = median_f32(&mut nb[..m])
                        .max(0.5 * sigma_global)
                        .max(sigma_floor);
                }
            }
        }
        Self {
            level,
            sigma,
            sigma_global,
        }
    }
}

impl BlockGrid {
    /// A grid of `nx × ny` block values for a `w`-wide image.
    pub(super) fn new(grid: Vec<f32>, nx: usize, ny: usize, block: usize, w: usize) -> Self {
        debug_assert_eq!(grid.len(), nx * ny);
        Self {
            grid,
            nx,
            ny,
            block,
            cols: ColPlan::new(nx, block, w),
        }
    }

    /// The same geometry with different block values.
    pub(super) fn with_values(&self, grid: Vec<f32>) -> Self {
        Self::new(grid, self.nx, self.ny, self.block, self.cols.fx.len())
    }

    /// Bound on the rounding error of an interpolated surface value: a few
    /// ulps of the largest block value.
    pub(super) fn rounding_margin(&self) -> f32 {
        let max_abs = self.grid.iter().fold(0.0_f32, |m, g| m.max(g.abs()));
        8.0 * f32::EPSILON * max_abs
    }

    /// Representative value: the midpoint of the block values.
    pub(super) fn midpoint(&self) -> f32 {
        midpoint_f32(&mut self.grid.clone())
    }

    /// Index of the block containing pixel `(x, y)`.
    #[inline]
    pub(super) fn block_index(&self, x: usize, y: usize) -> usize {
        (y / self.block).min(self.ny - 1) * self.nx + (x / self.block).min(self.nx - 1)
    }

    /// Row-constant part of the interpolation for image row `y`.
    #[inline]
    pub(super) fn row_params(&self, y: usize) -> (usize, usize, f32) {
        axis_params(self.ny, self.block, y)
    }

    /// Interpolated value at `(x, row)` given `row_params(row)`. `x` must
    /// be a column of the image the grid was built for.
    #[inline]
    pub(super) fn value_at(&self, x: usize, (by0, by1, fy): (usize, usize, f32)) -> f32 {
        let (bx0, bx1) = (self.cols.bx0[x] as usize, self.cols.bx1[x] as usize);
        let g0 = self.grid[by0 * self.nx + bx0] * (1.0 - fy) + self.grid[by1 * self.nx + bx0] * fy;
        let g1 = self.grid[by0 * self.nx + bx1] * (1.0 - fy) + self.grid[by1 * self.nx + bx1] * fy;
        g0 * self.cols.omfx[x] + g1 * self.cols.fx[x]
    }

    /// Value at column `x` of the row blended into `row_blend` by
    /// [`Self::blend_row`] — [`Self::value_at`] with the row half hoisted.
    #[inline]
    pub(super) fn value_in_row(&self, row_blend: &[f32], x: usize) -> f32 {
        let (bx0, bx1) = (self.cols.bx0[x] as usize, self.cols.bx1[x] as usize);
        row_blend[bx0] * self.cols.omfx[x] + row_blend[bx1] * self.cols.fx[x]
    }

    /// Blend one grid row for `row_params(row)` into `out` (length `nx`).
    #[inline]
    pub(super) fn blend_row(&self, (by0, by1, fy): (usize, usize, f32), out: &mut [f32]) {
        for (bx, g) in out.iter_mut().enumerate() {
            *g = self.grid[by0 * self.nx + bx] * (1.0 - fy) + self.grid[by1 * self.nx + bx] * fy;
        }
    }

    /// Column half of the interpolation for a whole row: with `row_blend`
    /// from [`Self::blend_row`], `out[c] = value_at(c, row)` for every image
    /// column. `out.len()` must equal the image width.
    #[inline]
    pub(super) fn blend_columns(&self, row_blend: &[f32], out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.cols.fx.len());
        for &(c0, c1, bx0, bx1) in &self.cols.segs {
            let (g0, g1) = (row_blend[bx0], row_blend[bx1]);
            let fx = &self.cols.fx[c0..c1];
            let omfx = &self.cols.omfx[c0..c1];
            for ((o, &f), &omf) in out[c0..c1].iter_mut().zip(fx).zip(omfx) {
                *o = g0 * omf + g1 * f;
            }
        }
    }
}

/// The quantization step of the pixel values near the background level:
/// the gap between the median of a sparse sample and the nearest distinct
/// sample value on either side (`0.0` when every sample is equal).
///
/// A signal quantized in steps of `q` carries at least `q/√12` of noise.
/// On continuous data the gap is a tiny fraction of σ and the bound it
/// gives is inert; it only matters when the samples tie heavily at the
/// median (integer frames whose noise is well under one count), where a
/// measured σ collapses to zero and the detection threshold would
/// otherwise sit on the background itself.
pub(super) fn quantization_step(pixels: &[f32]) -> f32 {
    const SAMPLES: usize = 4096;
    // An odd stride walks every column residue of an even-width image.
    let stride = (pixels.len() / SAMPLES).max(1) | 1;
    let mut vals: Vec<f32> = pixels
        .iter()
        .step_by(stride)
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    if vals.is_empty() {
        return 0.0;
    }
    let median = midpoint_f32(&mut vals);
    let gap = vals
        .iter()
        .map(|&v| (v - median).abs())
        .filter(|&d| d > 0.0)
        .fold(f32::INFINITY, f32::min);
    if gap.is_finite() {
        gap
    } else {
        0.0
    }
}
