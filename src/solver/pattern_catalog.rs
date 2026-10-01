//! Succinct pattern hash table ([`PatternCatalog`]).
//!
//! The table has `n_slots` slots (open addressing, quadratic probing), about
//! half of them empty by construction. Instead of a dense array with a
//! sentinel per empty slot, it stores:
//!
//! - an occupancy bitmap, interleaved with a rank directory in 64-byte
//!   [`RankBlock`]s (one running count + 448 occupancy bits), so a probe's
//!   "is slot `i` empty?" test and the rank of `i` (occupied slots before
//!   it) cost one cache line;
//! - the occupied entries packed in slot order, [`PACKED_ENTRY_BYTES`] each
//!   (four `u32` star indices, `f32` largest edge, `u16` key hash; all
//!   little-endian) — byte-identical to the database file's packed section
//!   (`pattern_wire`), so the table can sit on the file's bytes.
//!
//! Slot `i`'s entry is packed entry `rank(i)`.

use std::ops::Deref;
use std::sync::Arc;

use super::PatternEntry;

/// Bytes per occupied entry in the packed section.
pub(crate) const PACKED_ENTRY_BYTES: usize = 22;

const WORDS_PER_BLOCK: usize = 7;
const SLOTS_PER_BLOCK: usize = WORDS_PER_BLOCK * 64;

/// One cache line of the occupancy bitmap with its rank prefix.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C, align(64))]
struct RankBlock {
    /// Occupied slots in all earlier blocks.
    rank: u64,
    /// Occupancy of this block's slots: bit `s % 64` of word `s / 64`.
    bits: [u64; WORDS_PER_BLOCK],
}

/// The packed section: owned, or a range of a shared buffer — the database
/// file's bytes, kept as the table's backing store so loading does not copy
/// the section ([`super::SolverDatabase::from_vec`]).
#[derive(Clone)]
pub(crate) enum PackedStore {
    Owned(Vec<u8>),
    Shared {
        buf: Arc<Vec<u8>>,
        start: usize,
        len: usize,
    },
}

impl Deref for PackedStore {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &[u8] {
        match self {
            Self::Owned(v) => v,
            Self::Shared { buf, start, len } => &buf[*start..*start + *len],
        }
    }
}

impl PartialEq for PackedStore {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl std::fmt::Debug for PackedStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match self {
            Self::Owned(_) => "owned",
            Self::Shared { .. } => "shared",
        };
        write!(f, "PackedStore({kind}, {} bytes)", self.len())
    }
}

/// Pattern hash table: occupancy bitmap + rank directory + packed entries.
/// See the module docs for the layout.
///
/// Immutable once built. Generation builds it with [`Self::build`]; a dense
/// table (one [`PatternEntry`] per slot, [`PatternEntry::EMPTY`] for empty
/// slots) converts with [`Self::from_dense`] / [`Self::to_dense`].
#[derive(Debug, Clone, PartialEq)]
pub struct PatternCatalog {
    n_slots: usize,
    blocks: Vec<RankBlock>,
    packed: PackedStore,
}

impl PatternEntry {
    pub(crate) fn write_packed(&self, out: &mut Vec<u8>) {
        for i in self.star_indices {
            out.extend_from_slice(&i.to_le_bytes());
        }
        out.extend_from_slice(&self.largest_edge.to_le_bytes());
        out.extend_from_slice(&self.key_hash.to_le_bytes());
    }

    #[inline]
    pub(crate) fn read_packed(b: &[u8; PACKED_ENTRY_BYTES]) -> Self {
        let u = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        Self::new(
            [u(0), u(4), u(8), u(12)],
            f32::from_le_bytes([b[16], b[17], b[18], b[19]]),
            u16::from_le_bytes([b[20], b[21]]),
        )
    }
}

impl PatternCatalog {
    /// Insert `items` (`(hash_index, entry)`, in insertion order) into a
    /// table of `n_slots` slots by quadratic probing — the slot each entry
    /// lands in is the one a dense table would give it. Only the occupancy
    /// bitmap is probed, so no dense table is ever allocated.
    ///
    /// Panics if `items` holds more than `n_slots` entries.
    pub fn build(n_slots: usize, items: impl IntoIterator<Item = (u64, PatternEntry)>) -> Self {
        let mut words = vec![0u64; n_slots.div_ceil(64)];
        let mut slots = Vec::new();
        let mut entries = Vec::new();
        for (hash_index, entry) in items {
            assert!(slots.len() < n_slots, "hash table is full");
            let slot = probe_free(&words, hash_index, n_slots as u64);
            words[slot / 64] |= 1 << (slot % 64);
            slots.push(slot);
            entries.push(entry);
        }
        let blocks = rank_blocks(&words);
        let mut packed = vec![0u8; entries.len() * PACKED_ENTRY_BYTES];
        let mut buf = Vec::with_capacity(PACKED_ENTRY_BYTES);
        for (&slot, entry) in slots.iter().zip(&entries) {
            let o = rank_of(&blocks, slot) * PACKED_ENTRY_BYTES;
            buf.clear();
            entry.write_packed(&mut buf);
            packed[o..o + PACKED_ENTRY_BYTES].copy_from_slice(&buf);
        }
        Self {
            n_slots,
            blocks,
            packed: PackedStore::Owned(packed),
        }
    }

    /// Build from a dense table: one entry per slot, [`PatternEntry::EMPTY`]
    /// (any entry with [`PatternEntry::is_empty`]) for an empty slot.
    pub fn from_dense(slots: &[PatternEntry]) -> Self {
        let mut words = vec![0u64; slots.len().div_ceil(64)];
        let mut packed = Vec::new();
        for (i, e) in slots.iter().enumerate() {
            if !e.is_empty() {
                words[i / 64] |= 1 << (i % 64);
                e.write_packed(&mut packed);
            }
        }
        Self {
            n_slots: slots.len(),
            blocks: rank_blocks(&words),
            packed: PackedStore::Owned(packed),
        }
    }

    /// Assemble from the wire sections: occupancy words (`ceil(n_slots/64)`
    /// of them, no bit at or past `n_slots`) and the packed entries in slot
    /// order, one per set bit. The caller has checked those relations
    /// (`pattern_wire::unpack`).
    pub(crate) fn from_parts(n_slots: usize, words: &[u64], packed: PackedStore) -> Self {
        debug_assert_eq!(words.len(), n_slots.div_ceil(64));
        let blocks = rank_blocks(words);
        debug_assert_eq!(
            blocks.last().map_or(0, |b| b.rank as usize
                + b.bits
                    .iter()
                    .map(|w| w.count_ones() as usize)
                    .sum::<usize>()),
            packed.len() / PACKED_ENTRY_BYTES
        );
        Self {
            n_slots,
            blocks,
            packed,
        }
    }

    /// The equivalent dense table (one entry per slot,
    /// [`PatternEntry::EMPTY`] for empty slots).
    pub fn to_dense(&self) -> Vec<PatternEntry> {
        let mut out = vec![PatternEntry::EMPTY; self.n_slots];
        for (slot, e) in self.iter() {
            out[slot] = e;
        }
        out
    }

    /// Total number of slots.
    #[inline]
    pub fn len(&self) -> usize {
        self.n_slots
    }

    /// Returns `true` if the table has no slots.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n_slots == 0
    }

    /// Number of occupied slots.
    #[inline]
    pub fn num_occupied(&self) -> usize {
        self.packed.len() / PACKED_ENTRY_BYTES
    }

    /// Heap bytes held by the table (for a table sharing the database
    /// file's buffer, its packed range of that buffer).
    pub fn heap_bytes(&self) -> usize {
        self.blocks.len() * size_of::<RankBlock>() + self.packed.len()
    }

    /// The entry in slot `idx`, or `None` if the slot is empty.
    /// Panics if `idx >= len()`.
    #[inline]
    pub fn get(&self, idx: usize) -> Option<PatternEntry> {
        assert!(
            idx < self.n_slots,
            "slot {idx} past a {}-slot table",
            self.n_slots
        );
        let blk = &self.blocks[idx / SLOTS_PER_BLOCK];
        let s = idx % SLOTS_PER_BLOCK;
        let (w, b) = (s / 64, s % 64);
        if blk.bits[w] >> b & 1 == 0 {
            return None;
        }
        Some(self.packed_entry(block_rank(blk, w, b)))
    }

    /// Occupied slots and their entries, in slot order.
    pub fn iter(&self) -> impl Iterator<Item = (usize, PatternEntry)> + '_ {
        let words = self.blocks.iter().flat_map(|b| b.bits);
        words
            .enumerate()
            .flat_map(|(w, mut m)| {
                std::iter::from_fn(move || {
                    (m != 0).then(|| {
                        let i = w * 64 + m.trailing_zeros() as usize;
                        m &= m - 1;
                        i
                    })
                })
            })
            .zip(self.packed_entries())
    }

    /// The occupied entries in slot order.
    pub fn packed_entries(&self) -> impl ExactSizeIterator<Item = PatternEntry> + '_ {
        self.packed
            .as_chunks::<PACKED_ENTRY_BYTES>()
            .0
            .iter()
            .map(PatternEntry::read_packed)
    }

    /// Largest star index of any occupied entry (`None` if none is).
    /// Parallel under the `parallel` feature.
    pub fn max_star_index(&self) -> Option<u32> {
        let chunks = self.packed.as_chunks::<PACKED_ENTRY_BYTES>().0;
        #[cfg(not(feature = "parallel"))]
        let max = chunks.iter().map(entry_max_index).max();
        #[cfg(feature = "parallel")]
        let max = {
            use rayon::prelude::*;
            chunks.par_iter().map(entry_max_index).max()
        };
        max
    }

    #[cfg(test)]
    pub(crate) fn shares_buffer(&self) -> bool {
        matches!(self.packed, PackedStore::Shared { .. })
    }

    /// The packed section (wire form of the occupied entries).
    pub(crate) fn packed_bytes(&self) -> &[u8] {
        &self.packed
    }

    /// Occupancy bitmap as little-endian bytes, `ceil(n_slots/64)` words.
    pub(crate) fn bitmap_bytes(&self) -> Vec<u8> {
        let n_words = self.n_slots.div_ceil(64);
        self.blocks
            .iter()
            .flat_map(|b| b.bits)
            .take(n_words)
            .flat_map(u64::to_le_bytes)
            .collect()
    }

    #[inline]
    fn packed_entry(&self, r: usize) -> PatternEntry {
        let o = r * PACKED_ENTRY_BYTES;
        let b: &[u8; PACKED_ENTRY_BYTES] = self.packed[o..o + PACKED_ENTRY_BYTES]
            .try_into()
            .expect("rank within the packed section");
        PatternEntry::read_packed(b)
    }
}

#[inline]
fn entry_max_index(b: &[u8; PACKED_ENTRY_BYTES]) -> u32 {
    let u = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
    u(0).max(u(4)).max(u(8).max(u(12)))
}

/// Occupied slots before bit `b` of word `w` of `blk`, plus earlier blocks.
#[inline]
fn block_rank(blk: &RankBlock, w: usize, b: usize) -> usize {
    let below: u32 = blk.bits[..w].iter().map(|x| x.count_ones()).sum();
    blk.rank as usize + (below + (blk.bits[w] & ((1u64 << b) - 1)).count_ones()) as usize
}

fn rank_of(blocks: &[RankBlock], slot: usize) -> usize {
    let s = slot % SLOTS_PER_BLOCK;
    block_rank(&blocks[slot / SLOTS_PER_BLOCK], s / 64, s % 64)
}

/// Group occupancy words into rank blocks.
fn rank_blocks(words: &[u64]) -> Vec<RankBlock> {
    let mut acc = 0u64;
    words
        .chunks(WORDS_PER_BLOCK)
        .map(|ws| {
            let mut blk = RankBlock {
                rank: acc,
                ..Default::default()
            };
            blk.bits[..ws.len()].copy_from_slice(ws);
            acc += ws.iter().map(|w| w.count_ones() as u64).sum::<u64>();
            blk
        })
        .collect()
}

/// First free slot on `hash_index`'s quadratic-probe sequence.
fn probe_free(words: &[u64], hash_index: u64, n_slots: u64) -> usize {
    for c in 0u64.. {
        let i = ((hash_index.wrapping_add(c.wrapping_mul(c))) % n_slots) as usize;
        if words[i / 64] >> (i % 64) & 1 == 0 {
            return i;
        }
    }
    unreachable!("hash table is full")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(k: u32) -> PatternEntry {
        PatternEntry::new([k, k + 1, k + 2, k + 3], 0.01 * k as f32, 0x1000 + k as u16)
    }

    fn dense(n: usize, occupied: &[usize]) -> Vec<PatternEntry> {
        let mut d = vec![PatternEntry::EMPTY; n];
        for (k, &i) in occupied.iter().enumerate() {
            d[i] = entry(k as u32 + 1);
        }
        d
    }

    #[test]
    fn get_matches_dense_across_block_boundaries() {
        for (n, occ) in [
            (0usize, vec![]),
            (1, vec![0]),
            (447, vec![0, 63, 64, 446]),
            (448, vec![447]),
            (449, vec![447, 448]),
            (2000, vec![0, 5, 63, 64, 447, 448, 449, 895, 896, 1999]),
        ] {
            let d = dense(n, &occ);
            let cat = PatternCatalog::from_dense(&d);
            assert_eq!(cat.len(), n);
            assert_eq!(cat.num_occupied(), occ.len());
            for (i, e) in d.iter().enumerate() {
                assert_eq!(cat.get(i), (!e.is_empty()).then_some(*e), "n={n} slot {i}");
            }
            assert_eq!(cat.to_dense(), d);
        }
    }

    #[test]
    fn build_places_entries_where_dense_probing_would() {
        // Reference: the dense quadratic-probing insert generation used to do.
        let n = 1009usize;
        let mut d = vec![PatternEntry::EMPTY; n];
        let items: Vec<(u64, PatternEntry)> = (0..500u32)
            .map(|k| ((k as u64 * 7919) % 97, entry(k + 1))) // heavy collisions
            .collect();
        for &(h, e) in &items {
            let mut c = 0u64;
            loop {
                let i = ((h + c * c) % n as u64) as usize;
                if d[i].is_empty() {
                    d[i] = e;
                    break;
                }
                c += 1;
            }
        }
        assert_eq!(
            PatternCatalog::build(n, items),
            PatternCatalog::from_dense(&d)
        );
    }

    #[test]
    #[should_panic(expected = "past a 10-slot table")]
    fn get_past_the_end_panics() {
        PatternCatalog::from_dense(&dense(10, &[9])).get(10);
    }
}
