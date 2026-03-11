//! rkyv wrappers for numeris types.
//!
//! These implement `ArchiveWith`/`SerializeWith`/`DeserializeWith` so that structs
//! containing `Matrix2<f32>` or `Quaternion<f32>` can `#[derive(rkyv::Archive, ...)]`
//! by annotating the field with `#[rkyv(with = WrapperType)]`.

use rkyv::with::{ArchiveWith, DeserializeWith, SerializeWith};
use rkyv::{Archive, Deserialize, Place, Serialize};

// ── Matrix2<f32> ↔ [[f32; 2]; 2] ───────────────────────────────────────────

/// rkyv wrapper: serializes `Matrix2<f32>` as `[[f32; 2]; 2]` (row-major).
pub struct AsMatrix2Array;

impl ArchiveWith<numeris::Matrix2<f32>> for AsMatrix2Array {
    type Archived = <[[f32; 2]; 2] as Archive>::Archived;
    type Resolver = <[[f32; 2]; 2] as Archive>::Resolver;

    fn resolve_with(
        field: &numeris::Matrix2<f32>,
        resolver: Self::Resolver,
        out: Place<Self::Archived>,
    ) {
        let arr = [
            [field[(0, 0)], field[(0, 1)]],
            [field[(1, 0)], field[(1, 1)]],
        ];
        arr.resolve(resolver, out);
    }
}

impl<S: rkyv::rancor::Fallible + ?Sized> SerializeWith<numeris::Matrix2<f32>, S> for AsMatrix2Array
where
    [[f32; 2]; 2]: Serialize<S>,
{
    fn serialize_with(
        field: &numeris::Matrix2<f32>,
        serializer: &mut S,
    ) -> Result<Self::Resolver, S::Error> {
        let arr = [
            [field[(0, 0)], field[(0, 1)]],
            [field[(1, 0)], field[(1, 1)]],
        ];
        arr.serialize(serializer)
    }
}

impl<D: rkyv::rancor::Fallible + ?Sized>
    DeserializeWith<<[[f32; 2]; 2] as Archive>::Archived, numeris::Matrix2<f32>, D>
    for AsMatrix2Array
where
    <[[f32; 2]; 2] as Archive>::Archived: Deserialize<[[f32; 2]; 2], D>,
{
    fn deserialize_with(
        field: &<[[f32; 2]; 2] as Archive>::Archived,
        deserializer: &mut D,
    ) -> Result<numeris::Matrix2<f32>, D::Error> {
        let arr: [[f32; 2]; 2] = field.deserialize(deserializer)?;
        Ok(numeris::Matrix2::new([
            [arr[0][0], arr[0][1]],
            [arr[1][0], arr[1][1]],
        ]))
    }
}

// ── Quaternion<f32> ↔ [f32; 4] ─────────────────────────────────────────

/// rkyv wrapper: serializes `Quaternion<f32>` as `[x, y, z, w]`.
pub struct AsQuatArray;

impl ArchiveWith<numeris::Quaternion<f32>> for AsQuatArray {
    type Archived = <[f32; 4] as Archive>::Archived;
    type Resolver = <[f32; 4] as Archive>::Resolver;

    fn resolve_with(
        field: &numeris::Quaternion<f32>,
        resolver: Self::Resolver,
        out: Place<Self::Archived>,
    ) {
        let arr = [field.x, field.y, field.z, field.w];
        arr.resolve(resolver, out);
    }
}

impl<S: rkyv::rancor::Fallible + ?Sized> SerializeWith<numeris::Quaternion<f32>, S>
    for AsQuatArray
where
    [f32; 4]: Serialize<S>,
{
    fn serialize_with(
        field: &numeris::Quaternion<f32>,
        serializer: &mut S,
    ) -> Result<Self::Resolver, S::Error> {
        let arr = [field.x, field.y, field.z, field.w];
        arr.serialize(serializer)
    }
}

impl<D: rkyv::rancor::Fallible + ?Sized>
    DeserializeWith<<[f32; 4] as Archive>::Archived, numeris::Quaternion<f32>, D>
    for AsQuatArray
where
    <[f32; 4] as Archive>::Archived: Deserialize<[f32; 4], D>,
{
    fn deserialize_with(
        field: &<[f32; 4] as Archive>::Archived,
        deserializer: &mut D,
    ) -> Result<numeris::Quaternion<f32>, D::Error> {
        let [x, y, z, w]: [f32; 4] = field.deserialize(deserializer)?;
        Ok(numeris::Quaternion::new(w, x, y, z))
    }
}
