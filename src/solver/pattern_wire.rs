//! Wire format of the pattern hash table ([`PatternCatalog`]), database
//! format version 2.
//!
//! Serialized as a 3-tuple `(n_slots: u64, bitmap: bytes, packed: bytes)`.
//! `bitmap` holds `ceil(n_slots / 64)` little-endian `u64` words with bit `i`
//! set iff slot `i` is occupied; `packed` holds the occupied entries in slot
//! order, [`PACKED_ENTRY_BYTES`] each (four `u32` star indices, `f32`
//! largest edge, `u16` key hash; all little-endian).
//!
//! Why not the derived `Vec<PatternEntry>` (format version 1): the table is
//! half empty by construction (generation sizes it at `next_prime(2 ·
//! patterns)`) and postcard decoded every slot field by field through serde,
//! which was 70% of a 2 s load for a 127M-slot table. Here an empty slot
//! costs one bit, and the occupied entries are scattered straight from the
//! (borrowed) input buffer into a pre-sized table with no intermediate copy.
//! The file is also ~30% smaller.

use std::borrow::Cow;
use std::fmt;

use serde::de::{self, SeqAccess, Visitor};
use serde::ser::SerializeTuple;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::{PatternCatalog, PatternEntry};

/// Bytes per occupied entry in the packed section.
pub(crate) const PACKED_ENTRY_BYTES: usize = 22;

/// Largest table accepted relative to its occupied entries, plus slack for
/// tiny tables. Generation always produces load factor ≈ 0.5 (2 slots per
/// entry); a file claiming a table 16× sparser than that was not written by
/// this crate, and without the bound a small file could demand an arbitrarily
/// large table allocation through `n_slots` alone.
const MAX_SLOTS_PER_ENTRY: u64 = 16;
const SLOT_SLACK: u64 = 1024;

impl PatternEntry {
    fn write_packed(&self, out: &mut Vec<u8>) {
        for i in self.star_indices {
            out.extend_from_slice(&i.to_le_bytes());
        }
        out.extend_from_slice(&self.largest_edge.to_le_bytes());
        out.extend_from_slice(&self.key_hash.to_le_bytes());
    }

    #[inline]
    fn read_packed(b: &[u8; PACKED_ENTRY_BYTES]) -> Self {
        let u = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        Self::new(
            [u(0), u(4), u(8), u(12)],
            f32::from_le_bytes([b[16], b[17], b[18], b[19]]),
            u16::from_le_bytes([b[20], b[21]]),
        )
    }
}

impl PatternCatalog {
    /// Occupancy bitmap (little-endian `u64` words, bit `i` = slot `i`) and
    /// the occupied entries in slot order.
    fn pack(&self) -> (Vec<u8>, Vec<u8>) {
        let n = self.entries.len();
        let occupied = self.entries.iter().filter(|e| !e.is_empty()).count();
        let mut bitmap = vec![0u8; n.div_ceil(64) * 8];
        let mut packed = Vec::with_capacity(occupied * PACKED_ENTRY_BYTES);
        for (i, e) in self.entries.iter().enumerate() {
            if !e.is_empty() {
                bitmap[i / 8] |= 1 << (i % 8);
                e.write_packed(&mut packed);
            }
        }
        (bitmap, packed)
    }

    /// Rebuild the table from its wire sections, checking every relation the
    /// scatter relies on so a corrupt or hostile file fails here instead of
    /// indexing out of bounds.
    fn unpack(n_slots: u64, bitmap: &[u8], packed: &[u8]) -> Result<Self, String> {
        let n = usize::try_from(n_slots)
            .map_err(|_| format!("pattern table: {n_slots} slots do not fit in memory"))?;
        let n_words = n.div_ceil(64);
        if bitmap.len() != n_words * 8 {
            return Err(format!(
                "pattern table: occupancy bitmap is {} bytes, expected {} for {n} slots",
                bitmap.len(),
                n_words * 8
            ));
        }
        if !packed.len().is_multiple_of(PACKED_ENTRY_BYTES) {
            return Err(format!(
                "pattern table: packed section of {} bytes is not a whole number of \
                 {PACKED_ENTRY_BYTES}-byte entries",
                packed.len()
            ));
        }
        let occupied = packed.len() / PACKED_ENTRY_BYTES;
        let set_bits: usize = bitmap.iter().map(|b| b.count_ones() as usize).sum();
        if set_bits != occupied {
            return Err(format!(
                "pattern table: bitmap marks {set_bits} occupied slots but {occupied} entries \
                 are stored"
            ));
        }
        if n % 64 != 0 {
            let last = u64::from_le_bytes(bitmap[bitmap.len() - 8..].try_into().unwrap());
            if last >> (n % 64) != 0 {
                return Err("pattern table: occupancy bit set past the end of the table".into());
            }
        }
        if n_slots > occupied as u64 * MAX_SLOTS_PER_ENTRY + SLOT_SLACK {
            return Err(format!(
                "pattern table: {n_slots} slots for {occupied} entries is sparser than \
                 1/{MAX_SLOTS_PER_ENTRY}; generated tables are half full"
            ));
        }
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(n)
            .map_err(|_| format!("pattern table: cannot allocate {n} slots"))?;
        entries.resize(n, PatternEntry::EMPTY);
        scatter(bitmap, packed, &mut entries);
        Ok(Self { entries })
    }
}

/// Scatter `packed` entries into `out` at the set bits of `bitmap`.
/// Requires `bitmap.len() == out.len().div_ceil(64) * 8`, no bit set at or
/// past `out.len()`, and at least `popcount(bitmap)` entries in `packed`
/// (all checked by [`PatternCatalog::unpack`]).
fn scatter_words(bitmap: &[u8], packed: &[u8], out: &mut [PatternEntry]) {
    let mut src = packed.as_chunks::<PACKED_ENTRY_BYTES>().0.iter();
    for (w, word) in bitmap.as_chunks::<8>().0.iter().enumerate() {
        let mut m = u64::from_le_bytes(*word);
        while m != 0 {
            let i = w * 64 + m.trailing_zeros() as usize;
            m &= m - 1;
            let b = src.next().expect("entry count checked by unpack");
            out[i] = PatternEntry::read_packed(b);
        }
    }
}

#[cfg(not(feature = "parallel"))]
fn scatter(bitmap: &[u8], packed: &[u8], out: &mut [PatternEntry]) {
    scatter_words(bitmap, packed, out);
}

/// Parallel scatter: the bitmap is cut into word ranges, a prefix popcount
/// gives each range its offset into `packed`, and the ranges fill disjoint
/// slices of `out`. Also spreads the table's first-touch page faults, which
/// dominate the sequential version, across cores.
#[cfg(feature = "parallel")]
fn scatter(bitmap: &[u8], packed: &[u8], out: &mut [PatternEntry]) {
    use rayon::prelude::*;
    let n_words = bitmap.len() / 8;
    let words_per = n_words.div_ceil(rayon::current_num_threads() * 4).max(1);
    let mut offsets = Vec::with_capacity(n_words.div_ceil(words_per));
    let mut acc = 0usize;
    for ws in bitmap.chunks(words_per * 8) {
        offsets.push(acc);
        acc += ws.iter().map(|b| b.count_ones() as usize).sum::<usize>();
    }
    bitmap
        .par_chunks(words_per * 8)
        .zip(out.par_chunks_mut(words_per * 64))
        .zip(offsets.par_iter())
        .for_each(|((ws, o), &off)| scatter_words(ws, &packed[off * PACKED_ENTRY_BYTES..], o));
}

// ── serde glue ──────────────────────────────────────────────────────────────

/// `&[u8]` serialized with `serialize_bytes` (serde's blanket impl for
/// slices would emit a sequence, which postcard decodes element by element).
struct Bytes<'a>(&'a [u8]);

impl Serialize for Bytes<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(self.0)
    }
}

/// A byte section, borrowed from the input when the format allows it
/// (postcard does) so the scatter reads the file buffer directly.
struct CowBytes<'a>(Cow<'a, [u8]>);

impl<'de: 'a, 'a> Deserialize<'de> for CowBytes<'a> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = CowBytes<'de>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a byte section")
            }
            fn visit_borrowed_bytes<E: de::Error>(self, v: &'de [u8]) -> Result<Self::Value, E> {
                Ok(CowBytes(Cow::Borrowed(v)))
            }
            fn visit_bytes<E: de::Error>(self, v: &[u8]) -> Result<Self::Value, E> {
                Ok(CowBytes(Cow::Owned(v.to_vec())))
            }
            fn visit_byte_buf<E: de::Error>(self, v: Vec<u8>) -> Result<Self::Value, E> {
                Ok(CowBytes(Cow::Owned(v)))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut v = Vec::with_capacity(seq.size_hint().unwrap_or(0).min(1 << 20));
                while let Some(b) = seq.next_element::<u8>()? {
                    v.push(b);
                }
                Ok(CowBytes(Cow::Owned(v)))
            }
        }
        deserializer.deserialize_bytes(V)
    }
}

impl Serialize for PatternCatalog {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let (bitmap, packed) = self.pack();
        let mut t = serializer.serialize_tuple(3)?;
        t.serialize_element(&(self.entries.len() as u64))?;
        t.serialize_element(&Bytes(&bitmap))?;
        t.serialize_element(&Bytes(&packed))?;
        t.end()
    }
}

impl<'de> Deserialize<'de> for PatternCatalog {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = PatternCatalog;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a pattern table (n_slots, occupancy bitmap, packed entries)")
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let n_slots: u64 = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let bitmap: CowBytes<'de> = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;
                let packed: CowBytes<'de> = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(2, &self))?;
                PatternCatalog::unpack(n_slots, &bitmap.0, &packed.0).map_err(de::Error::custom)
            }
        }
        deserializer.deserialize_tuple(3, V)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(n: usize, occupied: &[usize]) -> PatternCatalog {
        let mut cat = PatternCatalog::with_capacity(n);
        for (k, &i) in occupied.iter().enumerate() {
            let k = k as u32 + 1;
            *cat.get_mut(i) =
                PatternEntry::new([k, k + 1, k + 2, k + 3], 0.01 * k as f32, 0x1000 + k as u16);
        }
        cat
    }

    fn roundtrip(cat: &PatternCatalog) -> PatternCatalog {
        let bytes = postcard::to_allocvec(cat).unwrap();
        postcard::from_bytes(&bytes).unwrap()
    }

    #[test]
    fn roundtrips_odd_sizes_and_edges() {
        for (n, occ) in [
            (0usize, vec![]),
            (1, vec![0]),
            (63, vec![0, 62]),
            (64, vec![63]),
            (65, vec![64]),
            (1000, vec![0, 5, 63, 64, 127, 128, 500, 999]),
        ] {
            let cat = table(n, &occ);
            assert_eq!(roundtrip(&cat), cat, "n = {n}");
        }
    }

    #[test]
    fn dense_table_roundtrips_and_parallel_scatter_matches() {
        // Larger than one parallel chunk on any core count.
        let n = 70_000;
        let occ: Vec<usize> = (0..n).filter(|i| i % 3 != 1).collect();
        let cat = table(n, &occ);
        assert_eq!(roundtrip(&cat), cat);
    }

    #[test]
    fn packed_section_is_smaller_than_the_table() {
        let cat = table(10_000, &(0..5_000).collect::<Vec<_>>());
        let bytes = postcard::to_allocvec(&cat).unwrap();
        // 5000 × 22 + bitmap 1256 + small framing.
        assert!(
            bytes.len() < 5_000 * PACKED_ENTRY_BYTES + 2_000,
            "{}",
            bytes.len()
        );
    }

    #[test]
    fn bit_layout_is_little_endian_words() {
        let (bitmap, _) = table(130, &[5, 64, 129]).pack();
        assert_eq!(bitmap.len(), 24);
        assert_eq!(bitmap[0], 1 << 5);
        assert_eq!(bitmap[8], 1);
        assert_eq!(bitmap[16], 1 << 1);
    }

    fn encode(n_slots: u64, bitmap: &[u8], packed: &[u8]) -> Vec<u8> {
        postcard::to_allocvec(&(n_slots, Bytes(bitmap), Bytes(packed))).unwrap()
    }

    #[test]
    fn tampered_sections_are_errors_not_panics() {
        let (bitmap, packed) = table(100, &[3, 70]).pack();
        let ok: PatternCatalog = postcard::from_bytes(&encode(100, &bitmap, &packed)).unwrap();
        assert_eq!(ok.len(), 100);

        let cases: Vec<(&str, Vec<u8>)> = vec![
            ("bitmap too short", encode(100, &bitmap[..8], &packed)),
            (
                "bitmap too long",
                encode(100, &[&bitmap[..], &[0; 8]].concat(), &packed),
            ),
            (
                "packed not whole entries",
                encode(100, &bitmap, &packed[..PACKED_ENTRY_BYTES + 1]),
            ),
            (
                "fewer entries than bits",
                encode(100, &bitmap, &packed[..PACKED_ENTRY_BYTES]),
            ),
            (
                "more entries than bits",
                encode(
                    100,
                    &bitmap,
                    &[&packed[..], &packed[..PACKED_ENTRY_BYTES]].concat(),
                ),
            ),
            ("bit past the end", {
                let mut bm = bitmap.clone();
                bm[12] |= 1 << 7; // bit 103 of a 100-slot table
                encode(
                    100,
                    &bm,
                    &[&packed[..], &packed[..PACKED_ENTRY_BYTES]].concat(),
                )
            }),
            ("absurdly sparse table", {
                let mut bm = vec![0u8; (1u64 << 20).div_ceil(64) as usize * 8];
                bm[0] = 1;
                encode(1 << 20, &bm, &packed[..PACKED_ENTRY_BYTES])
            }),
            ("truncated", encode(100, &bitmap, &packed)[..20].to_vec()),
        ];
        for (what, bytes) in cases {
            let r: Result<PatternCatalog, _> = postcard::from_bytes(&bytes);
            assert!(r.is_err(), "{what} should fail");
        }
    }
}
