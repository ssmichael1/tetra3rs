//! Wire format of the pattern hash table ([`PatternCatalog`]), database
//! format version 2.
//!
//! Serialized as a 3-tuple `(n_slots: u64, bitmap: bytes, packed: bytes)`.
//! `bitmap` holds `ceil(n_slots / 64)` little-endian `u64` words with bit `i`
//! set iff slot `i` is occupied; `packed` holds the occupied entries in slot
//! order, [`PACKED_ENTRY_BYTES`] each (four `u32` star indices, `f32`
//! largest edge, `u16` key hash; all little-endian).
//!
//! This is the in-memory layout of [`PatternCatalog`] minus its rank
//! directory, so loading is a bounds check, a copy of the packed section,
//! and a popcount pass over the bitmap — nothing is expanded per slot.
//! (Format version 1 was the dense `Vec<PatternEntry>`, decoded field by
//! field through serde: 70% of a 2 s load for a 127M-slot table.)

use std::borrow::Cow;
use std::fmt;

use serde::de::{self, SeqAccess, Visitor};
use serde::ser::SerializeTuple;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::pattern_catalog::{PackedStore, PACKED_ENTRY_BYTES};
use super::PatternCatalog;

/// Largest table accepted relative to its occupied entries, plus slack for
/// tiny tables. Generation always produces load factor ≈ 0.5 (2 slots per
/// entry). The bitmap-length check already ties `n_slots` to the file size
/// (8 slots per bitmap byte); this tightens the slots-per-file-byte ratio
/// for a file not written by this crate.
const MAX_SLOTS_PER_ENTRY: u64 = 16;
const SLOT_SLACK: u64 = 1024;

/// Rebuild the table from its wire sections, checking every relation the
/// rank directory relies on so a corrupt or hostile file fails here instead
/// of indexing out of bounds at probe time. `store` supplies the packed
/// section's storage once the checks pass (a copy, or a range of the
/// buffer `packed` borrows from).
pub(crate) fn unpack(
    n_slots: u64,
    bitmap: &[u8],
    packed: &[u8],
    store: impl FnOnce() -> PackedStore,
) -> Result<PatternCatalog, String> {
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
    let words: Vec<u64> = bitmap
        .as_chunks::<8>()
        .0
        .iter()
        .map(|w| u64::from_le_bytes(*w))
        .collect();
    let set_bits: usize = words.iter().map(|w| w.count_ones() as usize).sum();
    if set_bits != occupied {
        return Err(format!(
            "pattern table: bitmap marks {set_bits} occupied slots but {occupied} entries \
             are stored"
        ));
    }
    if n % 64 != 0 && words[n_words - 1] >> (n % 64) != 0 {
        return Err("pattern table: occupancy bit set past the end of the table".into());
    }
    if n_slots > occupied as u64 * MAX_SLOTS_PER_ENTRY + SLOT_SLACK {
        return Err(format!(
            "pattern table: {n_slots} slots for {occupied} entries is sparser than \
             1/{MAX_SLOTS_PER_ENTRY}; generated tables are half full"
        ));
    }
    Ok(PatternCatalog::from_parts(n, &words, store()))
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
/// (postcard does).
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
        let mut t = serializer.serialize_tuple(3)?;
        t.serialize_element(&(self.len() as u64))?;
        t.serialize_element(&Bytes(&self.bitmap_bytes()))?;
        t.serialize_element(&Bytes(self.packed_bytes()))?;
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
                let packed = packed.0;
                unpack(n_slots, &bitmap.0, &packed, || {
                    PackedStore::Owned(packed.clone().into_owned())
                })
                .map_err(de::Error::custom)
            }
        }
        deserializer.deserialize_tuple(3, V)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::PatternEntry;

    fn table(n: usize, occupied: &[usize]) -> PatternCatalog {
        let mut d = vec![PatternEntry::EMPTY; n];
        for (k, &i) in occupied.iter().enumerate() {
            let k = k as u32 + 1;
            d[i] = PatternEntry::new([k, k + 1, k + 2, k + 3], 0.01 * k as f32, 0x1000 + k as u16);
        }
        PatternCatalog::from_dense(&d)
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
    fn dense_table_roundtrips() {
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
        let bitmap = table(130, &[5, 64, 129]).bitmap_bytes();
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
        let cat = table(100, &[3, 70]);
        let (bitmap, packed) = (cat.bitmap_bytes(), cat.packed_bytes().to_vec());
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
