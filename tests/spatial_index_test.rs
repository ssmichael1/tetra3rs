//! Whole-sphere check of the catalog spatial index against brute force.
//!
//! Random directions over the full sphere plus a set forced within 5° of
//! either pole, several radii, and several `nside` resolutions — the polar
//! bug fixed in #59 was invisible to a single mid-latitude spot check.

mod common;

use numeris::Vector3;
use tetra3::StarCatalog;

#[test]
fn cone_query_matches_brute_force_over_the_sphere() {
    let stars = common::uniform_sky(50_000, 0x5EED_1234_ABCD_0001);
    let uvecs: Vec<Vector3<f32>> = stars.iter().map(|s| s.uvec()).collect();
    let mut rng = common::Rng::new(0x0BAD_CAFE_F00D_0002);

    let mut directions: Vec<Vector3<f32>> = (0..1500).map(|_| rng.direction()).collect();
    // Polar caps: |dec| > 85°.
    for _ in 0..300 {
        let dec =
            (85.0 + 5.0 * rng.unit()).to_radians() * if rng.unit() < 0.5 { 1.0 } else { -1.0 };
        let ra = rng.unit() * std::f32::consts::TAU;
        directions.push(Vector3::from_array([
            dec.cos() * ra.cos(),
            dec.cos() * ra.sin(),
            dec.sin(),
        ]));
    }

    let mut failures: Vec<String> = Vec::new();
    for nside in [4u32, 16, 64] {
        let catalog = StarCatalog::new(nside, stars.clone());
        let mut checked = 0usize;
        for dir in &directions {
            for radius_deg in [0.5f32, 2.0, 7.0, 20.0, 45.0, 80.0] {
                let radius = radius_deg.to_radians();
                let cos_r = radius.cos();
                let got = catalog.query_indices_from_uvec(*dir, radius);
                // Stars within a few f32 ulps of the boundary may fall on
                // either side of the query's own dot-product test; every
                // star clearly inside must be returned, and nothing clearly
                // outside may be.
                const EPS: f32 = 4e-6;
                let must_have: Vec<usize> = uvecs
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| dir.dot(v) >= cos_r + EPS)
                    .map(|(i, _)| i)
                    .collect();
                let missing = must_have
                    .iter()
                    .filter(|i| got.binary_search(i).is_err())
                    .count();
                let extra = got
                    .iter()
                    .filter(|&&i| dir.dot(&uvecs[i]) < cos_r - EPS)
                    .count();
                if missing > 0 || extra > 0 {
                    let dec = dir[2].asin().to_degrees();
                    failures.push(format!(
                        "nside {nside}, dec {dec:6.2}°, radius {radius_deg:4}°: expected {} got {} (missing {missing}, extra {extra})",
                        must_have.len(),
                        got.len()
                    ));
                }
                checked += 1;
            }
        }
        assert_eq!(checked, directions.len() * 6);
    }
    assert!(
        failures.is_empty(),
        "{} cone queries disagree with brute force, e.g.:\n{}",
        failures.len(),
        failures
            .iter()
            .take(12)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}
