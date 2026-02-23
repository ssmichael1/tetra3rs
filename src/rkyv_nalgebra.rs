//! rkyv wrappers for nalgebra types that don't natively support rkyv 0.8.
//!
//! These implement `ArchiveWith`/`SerializeWith`/`DeserializeWith` so that structs
//! containing `Matrix2<f32>` or `UnitQuaternion<f32>` can `#[derive(rkyv::Archive, ...)]`
//! by annotating the field with `#[rkyv(with = WrapperType)]`.

use rkyv::with::{ArchiveWith, DeserializeWith, SerializeWith};
use rkyv::{Archive, Deserialize, Place, Serialize};

// ── Matrix2<f32> ↔ [[f32; 2]; 2] ───────────────────────────────────────────

/// rkyv wrapper: serializes `Matrix2<f32>` as `[[f32; 2]; 2]` (row-major).
pub struct AsMatrix2Array;

impl ArchiveWith<nalgebra::Matrix2<f32>> for AsMatrix2Array {
    type Archived = <[[f32; 2]; 2] as Archive>::Archived;
    type Resolver = <[[f32; 2]; 2] as Archive>::Resolver;

    fn resolve_with(
        field: &nalgebra::Matrix2<f32>,
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

impl<S: rkyv::rancor::Fallible + ?Sized> SerializeWith<nalgebra::Matrix2<f32>, S> for AsMatrix2Array
where
    [[f32; 2]; 2]: Serialize<S>,
{
    fn serialize_with(
        field: &nalgebra::Matrix2<f32>,
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
    DeserializeWith<<[[f32; 2]; 2] as Archive>::Archived, nalgebra::Matrix2<f32>, D>
    for AsMatrix2Array
where
    <[[f32; 2]; 2] as Archive>::Archived: Deserialize<[[f32; 2]; 2], D>,
{
    fn deserialize_with(
        field: &<[[f32; 2]; 2] as Archive>::Archived,
        deserializer: &mut D,
    ) -> Result<nalgebra::Matrix2<f32>, D::Error> {
        let arr: [[f32; 2]; 2] = field.deserialize(deserializer)?;
        Ok(nalgebra::Matrix2::new(
            arr[0][0], arr[0][1], arr[1][0], arr[1][1],
        ))
    }
}

// ── UnitQuaternion<f32> ↔ [f32; 4] ─────────────────────────────────────────

/// rkyv wrapper: serializes `UnitQuaternion<f32>` as `[x, y, z, w]`.
pub struct AsQuatArray;

impl ArchiveWith<nalgebra::UnitQuaternion<f32>> for AsQuatArray {
    type Archived = <[f32; 4] as Archive>::Archived;
    type Resolver = <[f32; 4] as Archive>::Resolver;

    fn resolve_with(
        field: &nalgebra::UnitQuaternion<f32>,
        resolver: Self::Resolver,
        out: Place<Self::Archived>,
    ) {
        let c = field.as_ref().coords;
        let arr = [c.x, c.y, c.z, c.w];
        arr.resolve(resolver, out);
    }
}

impl<S: rkyv::rancor::Fallible + ?Sized> SerializeWith<nalgebra::UnitQuaternion<f32>, S>
    for AsQuatArray
where
    [f32; 4]: Serialize<S>,
{
    fn serialize_with(
        field: &nalgebra::UnitQuaternion<f32>,
        serializer: &mut S,
    ) -> Result<Self::Resolver, S::Error> {
        let c = field.as_ref().coords;
        let arr = [c.x, c.y, c.z, c.w];
        arr.serialize(serializer)
    }
}

impl<D: rkyv::rancor::Fallible + ?Sized>
    DeserializeWith<<[f32; 4] as Archive>::Archived, nalgebra::UnitQuaternion<f32>, D>
    for AsQuatArray
where
    <[f32; 4] as Archive>::Archived: Deserialize<[f32; 4], D>,
{
    fn deserialize_with(
        field: &<[f32; 4] as Archive>::Archived,
        deserializer: &mut D,
    ) -> Result<nalgebra::UnitQuaternion<f32>, D::Error> {
        let [x, y, z, w]: [f32; 4] = field.deserialize(deserializer)?;
        Ok(nalgebra::UnitQuaternion::new_unchecked(
            nalgebra::Quaternion::new(w, x, y, z),
        ))
    }
}
