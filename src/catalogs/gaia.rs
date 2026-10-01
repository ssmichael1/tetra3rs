use crate::error::{Error, Result};
use log::debug;
use std::path::Path;

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

/// Load a Gaia catalog from the binary format.
///
/// Binary format spec:
/// - Header: magic "GDR3" (4 bytes) + version (u32 LE, value 1) + num_stars (u64 LE)
/// - Per star (36 bytes): source_id (i64 LE) + ra (f64 LE) + dec (f64 LE) + mag (f32 LE) + pmra (f32 LE) + pmdec (f32 LE)
pub fn load_gaia_binary<P: AsRef<Path>>(path: P) -> Result<Vec<GaiaStar>> {
    use std::io::Read;

    let mut file = std::fs::File::open(path)?;
    let mut header = [0u8; 16];
    file.read_exact(&mut header)?;

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
    // A corrupt/truncated header could claim a huge `num_stars`, driving a
    // multi-gigabyte allocation (or a capacity-overflow abort) before
    // `read_exact` gets a chance to fail. Require the star block to match the
    // file's remaining bytes exactly first.
    let expected_bytes = num_stars.checked_mul(record_size).ok_or_else(|| {
        Error::InvalidCatalog("Gaia binary: num_stars * record_size overflows".into())
    })?;
    let data_bytes = file.metadata()?.len().saturating_sub(header.len() as u64);
    // Require *at least* the claimed star block (guards a truncated header that
    // would otherwise drive a multi-gigabyte allocation before `read_exact`
    // fails). Trailing bytes — padding or appended metadata that 0.8 read past
    // fine — are tolerated: we consume exactly `expected_bytes` below.
    if data_bytes < expected_bytes as u64 {
        return Err(Error::InvalidCatalog(format!(
            "Gaia binary: header claims {num_stars} stars ({expected_bytes} bytes) \
             but file has only {data_bytes} data bytes"
        )));
    }
    if data_bytes > expected_bytes as u64 {
        debug!(
            "Gaia binary: {} trailing byte(s) after the {num_stars}-star block; ignoring",
            data_bytes - expected_bytes as u64
        );
    }
    let mut buf = vec![0u8; expected_bytes];
    file.read_exact(&mut buf)?;

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
