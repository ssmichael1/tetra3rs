use crate::error::{Error, Result};
use crate::prelude::*;
use log::debug;

pub struct GaiaStar {
    pub source_id: i64,
    pub ra_deg: f32,
    pub dec_deg: f32,
    pub phot_g_mean_mag: f32,
    /// Proper motion in RA (mas/yr, as μα·cos δ). The binary format encodes
    /// missing proper motion as 0.0, so this is never optional.
    pub pmra: f32,
    /// Proper motion in Dec (mas/yr).
    pub pmdec: f32,
}

/// Parse a Gaia catalog from its binary format, held in memory.
///
/// Binary format spec:
/// - Header: magic "GDR3" (4 bytes) + version (u32 LE, value 1) + num_stars (u64 LE)
/// - Per star (36 bytes): source_id (i64 LE) + ra (f64 LE) + dec (f64 LE) + mag (f32 LE) + pmra (f32 LE) + pmdec (f32 LE)
///
/// Trailing bytes after the star block are tolerated (and ignored).
pub fn parse_gaia_binary(bytes: &[u8]) -> Result<Vec<GaiaStar>> {
    let header: &[u8; 16] = bytes
        .get(..16)
        .and_then(|h| h.try_into().ok())
        .ok_or_else(|| Error::InvalidCatalog("Gaia binary: truncated header".into()))?;

    // Validate magic
    if &header[0..4] != b"GDR3" {
        return Err(Error::InvalidCatalog(
            "Gaia binary: expected magic 'GDR3'".into(),
        ));
    }

    // Validate version
    let version = u32::from_le_bytes(header[4..8].try_into().unwrap());
    if version != 1 {
        return Err(Error::InvalidCatalog(format!(
            "Gaia binary: unsupported version {}, expected 1",
            version
        )));
    }

    // Convert the header's u64 count checked — `as usize` would silently
    // truncate on a 32-bit target, loading a fraction of the claimed stars.
    let num_stars: usize = u64::from_le_bytes(header[8..16].try_into().unwrap())
        .try_into()
        .map_err(|_| {
            Error::InvalidCatalog("Gaia binary: num_stars exceeds addressable memory".into())
        })?;

    let record_size = 36;
    // A corrupt header could claim a huge `num_stars`; check it against the
    // bytes actually present before sizing anything by it.
    let expected_bytes = num_stars.checked_mul(record_size).ok_or_else(|| {
        Error::InvalidCatalog("Gaia binary: num_stars * record_size overflows".into())
    })?;
    let data = &bytes[header.len()..];
    // Require *at least* the claimed star block. Trailing bytes — padding or
    // appended metadata that 0.8 read past fine — are tolerated: we consume
    // exactly `expected_bytes` below.
    if data.len() < expected_bytes {
        return Err(Error::InvalidCatalog(format!(
            "Gaia binary: header claims {num_stars} stars ({expected_bytes} bytes) \
             but file has only {} data bytes",
            data.len()
        )));
    }
    if data.len() > expected_bytes {
        debug!(
            "Gaia binary: {} trailing byte(s) after the {num_stars}-star block; ignoring",
            data.len() - expected_bytes
        );
    }
    let buf = &data[..expected_bytes];

    let mut stars = Vec::with_capacity(num_stars);
    let mut non_finite = 0usize;
    for i in 0..num_stars {
        let offset = i * record_size;
        let rec = &buf[offset..offset + record_size];

        let source_id = i64::from_le_bytes(rec[0..8].try_into().unwrap());
        let ra = f64::from_le_bytes(rec[8..16].try_into().unwrap());
        let dec = f64::from_le_bytes(rec[16..24].try_into().unwrap());
        let mag = f32::from_le_bytes(rec[24..28].try_into().unwrap());
        let pmra = f32::from_le_bytes(rec[28..32].try_into().unwrap());
        let pmdec = f32::from_le_bytes(rec[32..36].try_into().unwrap());

        // A NaN/inf position or magnitude would land the star in a bogus
        // spatial-index bin (or drop it from every magnitude cut), and a
        // non-finite proper motion poisons epoch propagation — all with no
        // diagnostic. Skip such records and report the count instead.
        if !(ra.is_finite()
            && dec.is_finite()
            && mag.is_finite()
            && pmra.is_finite()
            && pmdec.is_finite())
        {
            non_finite += 1;
            continue;
        }

        stars.push(GaiaStar {
            source_id,
            ra_deg: ra as f32,
            dec_deg: dec as f32,
            phot_g_mean_mag: mag,
            pmra,
            pmdec,
        });
    }
    if non_finite > 0 {
        log::warn!("Gaia binary: skipped {non_finite} record(s) with non-finite ra/dec/mag");
    }

    Ok(stars)
}
