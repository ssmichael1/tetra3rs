//! Types and helpers for working with Hipparcos catalog stars.
//!
//! This module contains the `HipparcosStar` representation, magnitude
//! conversion utilities, and helpers to load the Hipparcos catalog file
//! shipped with this crate.
//!
//! The Hipparcos catalog (new reduction, I/311) can be downloaded from
//! <http://cdsarc.u-strasbg.fr/ftp/I/311/hip2.dat.gz>.
//! The file `data/hip2.dat` in this crate is a copy of the catalog
//! as of 2025-11-15.

/// A star from the Hipparcos catalog.
///
/// Only the fields the solver consumes (position, proper motion, and the
/// magnitude/colour used for the Hp→V transform) are parsed and stored; the
/// catalog's per-field uncertainties, parallax, and V−I colour are skipped.
#[derive(Debug, Clone, PartialEq)]
pub struct HipparcosStar {
    pub hip: u32,
    pub ra_rad: f64,
    pub dec_rad: f64,
    pub pm_ra: f64,
    pub pm_dec: f64,
    pub hpmag: f32,
    pub b_v: f32,
}

impl HipparcosStar {
    /// Convert Hipparcos Hp magnitude and Johnson B−V colour
    /// to Johnson V using the standard 4th-order polynomial.
    ///
    /// Reference: ESA SP-1200, Volume 1, Table 1.3.5 (magnitude transformations).
    /// PDF mirror: https://www.cosmos.esa.int/documents/532822/552851/vol1_all.pdf
    ///
    /// Valid for roughly -0.2 < (B−V) < 1.8.
    pub fn hp_to_v(&self) -> f32 {
        let b = self.b_v;
        let delta = 0.304 * b - 0.202 * b * b + 0.107 * b * b * b - 0.045 * b * b * b * b;
        self.hpmag - delta
    }
}

/// Extract a fixed-width column as `&str`, or `None` if the bytes in that
/// range are not valid UTF-8. Slicing the record as bytes (not `&str`) keeps
/// a multi-byte character straddling a column boundary from panicking — such
/// a line is malformed and gets skipped like any other unparseable record.
fn column(record: &[u8], range: std::ops::Range<usize>) -> Option<&str> {
    std::str::from_utf8(record.get(range)?).ok()
}

/// Parse a single Hipparcos catalog record into a `HipparcosStar`.
fn parse_hipparcos_star(record: &str) -> Option<HipparcosStar> {
    let bytes = record.as_bytes();
    // Highest column offset the parser reads is the B−V field ending at 158.
    if bytes.len() < 158 {
        return None;
    }

    Some(HipparcosStar {
        hip: column(bytes, 0..6)?.trim().parse().ok()?,
        ra_rad: column(bytes, 15..28)?.trim().parse().ok()?,
        dec_rad: column(bytes, 29..42)?.trim().parse().ok()?,
        pm_ra: column(bytes, 51..59)?.trim().parse().ok()?,
        pm_dec: column(bytes, 60..68)?.trim().parse().ok()?,
        hpmag: column(bytes, 129..136)?.trim().parse().ok()?,
        b_v: column(bytes, 152..158)?.trim().parse().ok()?,
    })
}

/// Load the Hipparcos catalog from an in-memory string.
pub fn load_hipparcos_catalog(data: &str) -> Vec<HipparcosStar> {
    data.lines().filter_map(parse_hipparcos_star).collect()
}

pub fn load_hipparcos_catalog_from_file<P: AsRef<std::path::Path>>(
    path: P,
) -> crate::Result<Vec<HipparcosStar>> {
    let data = std::fs::read_to_string(path)?;
    let total_lines = data.lines().filter(|l| !l.trim().is_empty()).count();
    let stars = load_hipparcos_catalog(&data);
    // A wrong-format file parses to zero stars but would otherwise build a
    // useless (empty) database without any error; surface that here.
    if total_lines > 0 && stars.is_empty() {
        return Err(crate::error::Error::InvalidCatalog(format!(
            "Hipparcos catalog: parsed 0 stars from {total_lines} non-empty lines (wrong format?)"
        )));
    }
    let dropped = total_lines - stars.len();
    if dropped > 0 {
        log::warn!("Hipparcos catalog: skipped {dropped} unparseable line(s)");
    }
    Ok(stars)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multibyte_utf8_line_is_skipped_not_panicked() {
        // Regression: `&str` byte-range slicing panicked when a multi-byte
        // character straddled a fixed-width column boundary (e.g. at byte 5).
        let mut line = String::from("  12é"); // 'é' spans bytes 4..6 — across the 0..6 boundary
        line.push_str(&" ".repeat(160));
        assert!(parse_hipparcos_star(&line).is_none());

        // A fully multi-byte line long enough to pass the length gate.
        let junk = "é".repeat(100);
        assert!(parse_hipparcos_star(&junk).is_none());
    }

    #[test]
    #[ignore]
    fn load_hipparcos_from_file() {
        let fname = "data/hip2.dat";
        let data = std::fs::read_to_string(fname).expect("Failed to read Hipparcos catalog file");
        let stars = load_hipparcos_catalog(&data);
        assert!(!stars.is_empty());
    }
}
