//! Regression test for issue #80: the bundled `gaia_merged.bin` once carried
//! BP/RP magnitudes in its proper-motion fields (a CSV column mix-up in the
//! converter), so every star stayed at its J2016.0 position — up to 93″ off
//! by 2025. Checks two high-proper-motion stars against positions propagated
//! from the published Gaia DR3 / Hipparcos values.
#![cfg(feature = "std")] // generate_from_gaia reads the catalog file

mod test_data;

use tetra3::{GenerateDatabaseConfig, SolverDatabase};

/// (name, catalog id, RA°, Dec° at J2025.0, total motion since J2016.0 in arcsec)
const HIGH_PM_STARS: [(&str, i64, f64, f64, f64); 2] = [
    // Gaia DR3 4472832130942575872: pm = (−801.55, +10362.39) mas/yr.
    (
        "Barnard's star",
        4472832130942575872,
        269.446_491_773,
        4.765_326_037,
        93.5,
    ),
    // HIP 69673, a Hipparcos gap-fill row (negative id): pm = (−1093.39, −2000.06) mas/yr.
    ("Arcturus", -69673, 213.907_260_872, 19.168_519_898, 20.5),
];

#[test]
fn test_catalog_proper_motion_is_applied() {
    let catalog_path = test_data::ensure_test_file("data/gaia_merged.bin");
    // The pattern table is irrelevant here; keep it small. Barnard's star is G = 8.19.
    let config = GenerateDatabaseConfig {
        max_fov_deg: 20.0,
        min_fov_deg: None,
        star_max_magnitude: Some(8.5),
        pattern_max_error: 0.005,
        lattice_field_oversampling: 30,
        patterns_per_lattice_field: 25,
        verification_stars_per_fov: 50,
        epoch_proper_motion_year: Some(2025.0),
        ..Default::default()
    };
    let db =
        SolverDatabase::generate_from_gaia(&catalog_path, &config).expect("database generation");

    for (name, id, ra_deg, dec_deg, moved_arcsec) in HIGH_PM_STARS {
        let idx = db
            .star_catalog_ids
            .iter()
            .position(|&i| i == id)
            .unwrap_or_else(|| panic!("{name} (id {id}) missing from the database"));
        let v = db.star_vectors[idx];

        let (ra, dec) = (ra_deg.to_radians(), dec_deg.to_radians());
        let expected = [dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin()];
        let cross = [
            v[1] as f64 * expected[2] - v[2] as f64 * expected[1],
            v[2] as f64 * expected[0] - v[0] as f64 * expected[2],
            v[0] as f64 * expected[1] - v[1] as f64 * expected[0],
        ];
        let err_arcsec = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2])
            .sqrt()
            .asin()
            .to_degrees()
            * 3600.0;
        println!("{name}: {err_arcsec:.3}″ from the J2025.0 position (moved {moved_arcsec}″)");
        // f32 storage limits positions to ~0.1″; a catalog without the proper
        // motion is off by `moved_arcsec`.
        assert!(
            err_arcsec < 1.0,
            "{name}: {err_arcsec:.2}″ from its J2025.0 position — proper motion not applied \
             (the star has moved {moved_arcsec}″ since J2016.0)"
        );
    }
}
