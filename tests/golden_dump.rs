//! Golden solve dump: solves a deterministic set of synthetic fields and
//! writes every solution (quaternion, FOV, match count, p-value, residuals,
//! parity, WCS, matched ids and centroid indices) to a text file, one line
//! per field, so a refactor can be checked for bit-identical behavior.
//!
//! Ignored by default (it needs `data/gaia_merged.bin` and an output path).
//! Usage:
//!
//! ```sh
//! TETRA3_GOLDEN_OUT=/tmp/before.txt cargo test --release --test golden_dump -- --ignored --nocapture
//! # ... make the change ...
//! TETRA3_GOLDEN_OUT=/tmp/after.txt  cargo test --release --test golden_dump -- --ignored --nocapture
//! diff /tmp/before.txt /tmp/after.txt
//! # ignore the p-value column (e.g. after a verification-statistic change):
//! diff <(sed 's/prob=[^ ]* //' /tmp/before.txt) <(sed 's/prob=[^ ]* //' /tmp/after.txt)
//! ```
//!
//! Scenarios (1500 fields, 10° FOV, 1024 px): plain; 20 spurious + 0.3 px
//! noise; +0.4 % FOV bias with a 0.02 rad sweep; parity-flipped; hinted
//! tracking with a 0.3° hint error; 30 spurious; aberration; −3 % FOV with
//! a wide sweep. The database is generated in-test (deterministic since the
//! pattern list is sorted before the table is built).

mod common;

use numeris::{Quaternion, Vector3};
use std::io::Write;
use tetra3::{Centroid, SolveConfig, SolverDatabase};

#[test]
#[ignore = "golden dump: needs data/gaia_merged.bin and TETRA3_GOLDEN_OUT; see the module docs"]
#[allow(clippy::type_complexity)]
fn golden_dump() {
    let out_path = std::env::var("TETRA3_GOLDEN_OUT")
        .expect("set TETRA3_GOLDEN_OUT to the output file path (see module docs)");
    let db =
        SolverDatabase::generate_from_gaia("data/gaia_merged.bin", &common::profiler_db_config())
            .expect("data/gaia_merged.bin (downloaded by the --features image integration tests)");
    let fov = 10.0_f32.to_radians();
    let half = fov / 2.0;
    let w = 1024u32;
    let ps = 1.0 / ((w as f32 / 2.0) / half.tan());
    let mut rng = common::Rng::new(0xDEAD_BEEF_1234_5678);
    let mut f = std::fs::File::create(&out_path).unwrap();
    // scenario: (n, spurious, fov_bias, parity, noise_px, hint, fov_max_error, aberration)
    let scenarios: [(usize, usize, f32, bool, f32, bool, Option<f32>, bool); 8] = [
        (400, 0, 0.0, false, 0.0, false, None, false),
        (200, 20, 0.0, false, 0.3, false, None, false),
        (200, 0, 0.004, false, 0.0, false, Some(0.02), false),
        (200, 0, 0.0, true, 0.2, false, None, false),
        (200, 5, 0.0, false, 0.3, true, None, false),
        (100, 30, 0.0, false, 0.0, false, None, false), // random-ish, many spurious
        (100, 0, 0.0, false, 0.0, false, None, true),
        (100, 0, -0.03, false, 0.2, false, Some(0.05), false),
    ];
    let mut n_ok = 0usize;
    for (si, &(n, spurious, bias, parity, noise, hint, fov_err, aber)) in
        scenarios.iter().enumerate()
    {
        for t in 0..n {
            let ra = rng.unit() * std::f32::consts::TAU;
            let dec = (rng.unit() * 2.0 - 1.0).asin();
            let roll = rng.unit() * std::f32::consts::TAU;
            let r = common::rotation_from_ra_dec_roll(ra, dec, roll);
            let mut c = Vec::new();
            for (i, sv) in db.star_vectors.iter().enumerate() {
                let v = r * Vector3::from_array([sv[0], sv[1], sv[2]]);
                if v[2] > 0.01 {
                    let (cx, cy) = (v[0] / v[2], v[1] / v[2]);
                    if cx.abs() < half && cy.abs() < half {
                        let x = cx / ps + (rng.unit() - 0.5) * 2.0 * noise;
                        let y = cy / ps + (rng.unit() - 0.5) * 2.0 * noise;
                        c.push(Centroid {
                            x: if parity { -x } else { x },
                            y,
                            mass: Some(10.0 - db.star_catalog.stars()[i].mag),
                            cov: None,
                        });
                    }
                }
            }
            for _ in 0..spurious {
                c.push(Centroid {
                    x: (rng.unit() * 2.0 - 1.0) * half / ps,
                    y: (rng.unit() * 2.0 - 1.0) * half / ps,
                    mass: Some(rng.unit() * 5.0),
                    cov: None,
                });
            }
            let mut sc = SolveConfig::new(fov * (1.0 + bias), w, w);
            sc.fov_max_error_rad = fov_err;
            if hint {
                // perturb the true rotation by ~0.3° about a random axis
                let q = Quaternion::from_rotation_matrix(&r);
                let ax =
                    Vector3::from_array([rng.unit() - 0.5, rng.unit() - 0.5, rng.unit() - 0.5])
                        .normalize();
                let dq = Quaternion::from_axis_angle(ax, 0.3_f32.to_radians());
                sc.attitude_hint = Some(dq * q);
            }
            if aber {
                sc.observer_velocity_km_s = Some([10.0, -20.0, 5.0]);
            }
            match db.solve_from_centroids(&c, &sc) {
                Ok(s) => {
                    n_ok += 1;
                    let q = s.qicrs2cam;
                    writeln!(f, "{si} {t} OK q={:?} fov={} n={} prob={:e} rmse={} parity={} theta={} crval={:?} ids={:?} cent={:?}",
                        q, s.fov_rad, s.num_matches, s.prob, s.rmse_rad, s.parity_flip, s.theta_rad, s.crval_rad, s.matched_catalog_ids, s.matched_centroid_indices).unwrap();
                }
                Err(e) => writeln!(f, "{si} {t} FAIL {:?}", e.status).unwrap(),
            }
        }
    }
    println!(
        "golden: {n_ok} solved of {}",
        scenarios.iter().map(|s| s.0).sum::<usize>()
    );
}
