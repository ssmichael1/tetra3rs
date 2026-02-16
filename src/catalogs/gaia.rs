use std::path::Path;

#[allow(dead_code)]
pub struct GaiaStar {
    pub source_id: u64,
    pub ra_deg: f32,
    pub dec_deg: f32,
    pub phot_g_mean_mag: f32,
    pub phot_bp_mean_mag: f32,
    pub phot_rp_mean_mag: f32,
    pub parallax: Option<f32>,
    pub pmra: Option<f32>,
    pub pmdec: Option<f32>,
}

#[allow(dead_code)]
pub fn read_gaia_csv<P: AsRef<Path>>(file: P) -> anyhow::Result<Vec<GaiaStar>> {
    let mut rdr = csv::Reader::from_path(file)?;
    rdr.records()
        .into_iter()
        .skip(1)
        .map(|result| {
            let record = result?;
            let source_id: u64 = record.get(0).unwrap_or("").parse().unwrap_or(0);
            let ra: f32 = record.get(1).unwrap_or("").parse().unwrap_or(0.0);
            let dec: f32 = record.get(2).unwrap_or("").parse().unwrap_or(0.0);
            let phot_g_mean_mag: f32 = record.get(3).unwrap_or("").parse().unwrap_or(0.0);
            let phot_bp_mean_mag: f32 = record.get(4).unwrap_or("").parse().unwrap_or(0.0);
            let phot_rp_mean_mag: f32 = record.get(5).unwrap_or("").parse().unwrap_or(0.0);
            let parallax: Option<f32> = match record.get(6) {
                Some(s) if !s.is_empty() => s.parse().ok(),
                _ => None,
            };
            let pmra: Option<f32> = match record.get(7) {
                Some(s) if !s.is_empty() => s.parse().ok(),
                _ => None,
            };
            let pmdec: Option<f32> = match record.get(8) {
                Some(s) if !s.is_empty() => s.parse().ok(),
                _ => None,
            };

            Ok(GaiaStar {
                source_id,
                ra_deg: ra,
                dec_deg: dec,
                phot_g_mean_mag,
                phot_bp_mean_mag,
                phot_rp_mean_mag,
                parallax,
                pmra,
                pmdec,
            })
        })
        .collect::<Result<Vec<GaiaStar>, csv::Error>>()
        .map_err(|e| e.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn load_gaia_catalog() {
        let fname = "data/gaia_bright_stars.csv";
        let stars = read_gaia_csv(fname).expect("Failed to read Gaia catalog file");
        assert!(!stars.is_empty());
    }
}
