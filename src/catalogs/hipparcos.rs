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
#[derive(Debug, Clone, PartialEq)]
pub struct HipparcosStar {
    pub hip: u32,
    pub ra_rad: f64,
    pub dec_rad: f64,
    pub plx: f64,
    pub pm_ra: f64,
    pub pm_dec: f64,
    pub e_ra_rad: f64,
    pub e_dec_rad: f64,
    pub e_plx: f64,
    pub e_pm_ra: f64,
    pub e_pm_dec: f64,
    pub hpmag: f32,
    pub e_hpmag: f32,
    pub b_v: f32,
    pub e_b_v: f32,
    pub v_i: f32,
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

/// Parse a single Hipparcos catalog record into a `HipparcosStar`.
fn parse_hipparcos_star(record: &str) -> Option<HipparcosStar> {
    if record.len() < 171 {
        return None;
    }

    Some(HipparcosStar {
        hip: record[0..6].trim().parse().ok()?,
        ra_rad: record[15..28].trim().parse().ok()?,
        dec_rad: record[29..42].trim().parse().ok()?,
        plx: record[43..50].trim().parse().ok()?,
        pm_ra: record[51..59].trim().parse().ok()?,
        pm_dec: record[60..68].trim().parse().ok()?,
        e_ra_rad: record[69..75].trim().parse().ok()?,
        e_dec_rad: record[76..82].trim().parse().ok()?,
        e_plx: record[83..89].trim().parse().ok()?,
        e_pm_ra: record[90..96].trim().parse().ok()?,
        e_pm_dec: record[97..103].trim().parse().ok()?,
        hpmag: record[129..136].trim().parse().ok()?,
        e_hpmag: record[137..143].trim().parse().ok()?,
        b_v: record[152..158].trim().parse().ok()?,
        e_b_v: record[159..164].trim().parse().ok()?,
        v_i: record[165..171].trim().parse().ok()?,
    })
}

/// Load the Hipparcos catalog from an in-memory string.
pub fn load_hipparcos_catalog(data: &str) -> Vec<HipparcosStar> {
    data.lines().filter_map(parse_hipparcos_star).collect()
}

pub fn load_hipparcos_catalog_from_file<P: AsRef<std::path::Path>>(
    path: P,
) -> anyhow::Result<Vec<HipparcosStar>> {
    let data = std::fs::read_to_string(path)?;
    Ok(load_hipparcos_catalog(&data))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn load_hipparcos_from_file() {
        let fname = "data/hip2.dat";
        let data = std::fs::read_to_string(fname).expect("Failed to read Hipparcos catalog file");
        let stars = load_hipparcos_catalog(&data);
        assert!(!stars.is_empty());
    }
}
