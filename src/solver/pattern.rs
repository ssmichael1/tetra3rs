//! Pattern key computation, hashing, and hash-table operations.
//!
//! A "pattern" is a group of 4 stars. Its geometric signature is derived from
//! the 6 pairwise edge angles (C(4,2) = 6). The 5 smallest edges are divided
//! by the largest to produce 5 edge ratios in [0, 1], which are then quantized
//! into bins to form the pattern key.
//!
//! The pattern key is hashed into a compact table using open addressing with
//! quadratic probing (following tetra3's default for in-memory databases).

/// Number of stars in each pattern.
pub const PATTERN_SIZE: usize = 4;
/// Number of pairwise edges: C(4,2) = 6.
pub const NUM_EDGES: usize = 6;
/// Number of edge ratios (all edges except the largest): 6 - 1 = 5.
pub const NUM_EDGE_RATIOS: usize = 5;

/// Multiplicative hash constant from tetra3 (Knuth's golden-ratio hash).
const MAGIC_RAND: u64 = 2654435761;

// ── Angle / distance conversions on the unit sphere ─────────────────────────

/// Euclidean distance between two points on the unit sphere → center angle (radians).
#[inline]
pub fn angle_from_distance(dist: f32) -> f32 {
    2.0 * (0.5 * dist).clamp(-1.0, 1.0).asin()
}

/// Center angle (radians) → Euclidean distance between two points on the unit sphere.
#[inline]
pub fn distance_from_angle(angle: f32) -> f32 {
    2.0 * (angle / 2.0).sin()
}

// ── Edge-angle computation ──────────────────────────────────────────────────

/// Euclidean distance between two 3-vectors.
#[inline]
fn vec3_dist(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Compute the 6 pairwise edge angles between 4 star unit-vectors, returned sorted ascending.
pub fn compute_sorted_edge_angles(vectors: &[[f32; 3]; PATTERN_SIZE]) -> [f32; NUM_EDGES] {
    let mut edges = [0.0f32; NUM_EDGES];
    let mut idx = 0;
    for i in 0..PATTERN_SIZE {
        for j in (i + 1)..PATTERN_SIZE {
            edges[idx] = angle_from_distance(vec3_dist(&vectors[i], &vectors[j]));
            idx += 1;
        }
    }
    // Sort ascending (smallest edge first, largest last).
    edges.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    edges
}

/// Given 6 sorted edge angles, compute 5 edge ratios (each divided by the largest).
pub fn compute_edge_ratios(sorted_edges: &[f32; NUM_EDGES]) -> [f32; NUM_EDGE_RATIOS] {
    let largest = sorted_edges[NUM_EDGES - 1];
    let mut ratios = [0.0f32; NUM_EDGE_RATIOS];
    if largest > 0.0 {
        for i in 0..NUM_EDGE_RATIOS {
            ratios[i] = sorted_edges[i] / largest;
        }
    }
    ratios
}

// ── Pattern key quantization and hashing ────────────────────────────────────

/// Quantize edge ratios into integer bins, forming the pattern key.
pub fn compute_pattern_key(
    edge_ratios: &[f32; NUM_EDGE_RATIOS],
    bins: u32,
) -> [u32; NUM_EDGE_RATIOS] {
    let mut key = [0u32; NUM_EDGE_RATIOS];
    for i in 0..NUM_EDGE_RATIOS {
        key[i] = (edge_ratios[i] * bins as f32) as u32;
    }
    key
}

/// Hash a pattern key to a u64 using the polynomial hash from tetra3:
///   hash = sum( key\[i\] * bins^i )  (wrapping u64 arithmetic)
pub fn compute_pattern_key_hash(key: &[u32; NUM_EDGE_RATIOS], bins: u32) -> u64 {
    let bins64 = bins as u64;
    let mut hash: u64 = 0;
    let mut factor: u64 = 1;
    for &k in key.iter() {
        hash = hash.wrapping_add((k as u64).wrapping_mul(factor));
        factor = factor.wrapping_mul(bins64);
    }
    hash
}

/// Map a pattern key hash to a table index.
/// Uses quadratic probing: (hash * MAGIC_RAND) % table_size.
pub fn hash_to_index(hash: u64, table_size: u64) -> u64 {
    hash.wrapping_mul(MAGIC_RAND) % table_size
}

// ── Hash table operations (quadratic probing) ───────────────────────────────

/// Insert a pattern into the hash table at the first available slot
/// using quadratic probing. Returns the table index where it was stored.
pub fn insert_pattern(
    pattern: [u32; PATTERN_SIZE],
    hash_index: u64,
    table: &mut [[u32; PATTERN_SIZE]],
) -> usize {
    let max_ind = table.len() as u64;
    for c in 0u64.. {
        let i = ((hash_index.wrapping_add(c.wrapping_mul(c))) % max_ind) as usize;
        if table[i] == [0, 0, 0, 0] {
            table[i] = pattern;
            return i;
        }
    }
    unreachable!("hash table is full")
}

/// Walk the quadratic-probe chain from `hash_index`, collecting all non-empty
/// slot indices until an empty slot is found.
pub fn get_table_indices(hash_index: u64, table: &[[u32; PATTERN_SIZE]]) -> Vec<usize> {
    let max_ind = table.len() as u64;
    let mut found = Vec::new();
    for c in 0u64.. {
        let i = ((hash_index.wrapping_add(c.wrapping_mul(c))) % max_ind) as usize;
        if table[i] == [0, 0, 0, 0] {
            return found;
        }
        found.push(i);
    }
    unreachable!()
}

// ── Pattern centroid ordering ───────────────────────────────────────────────

/// Sort a pattern's star indices by each star's Euclidean distance from the
/// pattern centroid (average position). This produces a canonical ordering
/// that is invariant to the input star order, allowing image patterns to
/// be matched against catalog patterns.
pub fn sort_pattern_by_centroid_distance(
    pattern: &mut [usize; PATTERN_SIZE],
    vectors: &[[f32; 3]],
) {
    // Compute centroid (average of the 4 vectors).
    let mut cx = 0.0f32;
    let mut cy = 0.0f32;
    let mut cz = 0.0f32;
    for &idx in pattern.iter() {
        let v = &vectors[idx];
        cx += v[0];
        cy += v[1];
        cz += v[2];
    }
    cx /= PATTERN_SIZE as f32;
    cy /= PATTERN_SIZE as f32;
    cz /= PATTERN_SIZE as f32;

    // Sort by squared distance from centroid (ascending).
    pattern.sort_by(|&a, &b| {
        let va = &vectors[a];
        let vb = &vectors[b];
        let da = (va[0] - cx).powi(2) + (va[1] - cy).powi(2) + (va[2] - cz).powi(2);
        let db = (vb[0] - cx).powi(2) + (vb[1] - cy).powi(2) + (vb[2] - cz).powi(2);
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Same as above but operates on [u32; 4] with a separate vector lookup.
pub fn sort_u32_pattern_by_centroid_distance(
    pattern: &mut [u32; PATTERN_SIZE],
    star_vectors: &[[f32; 3]],
) {
    let mut cx = 0.0f32;
    let mut cy = 0.0f32;
    let mut cz = 0.0f32;
    for &idx in pattern.iter() {
        let v = &star_vectors[idx as usize];
        cx += v[0];
        cy += v[1];
        cz += v[2];
    }
    cx /= PATTERN_SIZE as f32;
    cy /= PATTERN_SIZE as f32;
    cz /= PATTERN_SIZE as f32;

    pattern.sort_by(|&a, &b| {
        let va = &star_vectors[a as usize];
        let vb = &star_vectors[b as usize];
        let da = (va[0] - cx).powi(2) + (va[1] - cy).powi(2) + (va[2] - cz).powi(2);
        let db = (vb[0] - cx).powi(2) + (vb[1] - cy).powi(2) + (vb[2] - cz).powi(2);
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    });
}

// ── Prime utilities for hash table sizing ───────────────────────────────────

pub fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 || n == 3 {
        return true;
    }
    if n % 2 == 0 || n % 3 == 0 {
        return false;
    }
    let mut i = 5u64;
    while i * i <= n {
        if n % i == 0 || n % (i + 2) == 0 {
            return false;
        }
        i += 6;
    }
    true
}

/// Return the smallest prime ≥ n.
pub fn next_prime(n: u64) -> u64 {
    if n <= 2 {
        return 2;
    }
    let mut candidate = n | 1; // ensure odd
    while !is_prime(candidate) {
        candidate += 2;
    }
    candidate
}
