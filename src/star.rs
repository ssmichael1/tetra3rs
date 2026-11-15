#[cfg(not(feature = "std"))]
use crate::prelude::*;
use serde::{Deserialize, Serialize};

/// A "Generic" star type that will be used for star matching
/// The star RA & Dec assume proper motion has already been applied to the observation epoch.
/// The magnitude is a generic "brightness" value that can be used for filtering, but the exact meaning is catalog-dependent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Star {
    pub id: i64,
    pub ra_rad: f32,
    pub dec_rad: f32,
    pub mag: f32,
}

impl Star {
    /// Unit vector pointing to the star's position on the celestial sphere.
    pub fn uvec(&self) -> numeris::Vector3<f32> {
        crate::starcatalog::radec_to_uvec(self.ra_rad, self.dec_rad)
    }
}

/// Position (radians) after `dt_years` of linear proper motion; `None`
/// leaves the catalog position as is.
///
/// `pm_ra_mas` is `μα·cos δ` and `pm_dec_mas` is `μδ`, both in
/// milliarcseconds per year (the Hipparcos and Gaia convention).
fn propagate_proper_motion(
    ra_rad: f64,
    dec_rad: f64,
    pm_ra_mas: f64,
    pm_dec_mas: f64,
    dt_years: Option<f64>,
) -> (f32, f32) {
    const MAS_PER_YR_TO_RAD_PER_YR: f64 = 2.0 * core::f64::consts::PI / (3600.0 * 1000.0 * 360.0);
    let Some(dt_years) = dt_years else {
        return (ra_rad as f32, dec_rad as f32);
    };
    let cos_dec = dec_rad.cos();
    // Near the poles, cos(dec) -> 0 makes the RA correction blow up.
    // Following tetra3: skip proper motion for |dec| > ~87 degrees.
    let (mu_ra, mu_dec) = if cos_dec.abs() > 0.05 {
        (
            pm_ra_mas * MAS_PER_YR_TO_RAD_PER_YR / cos_dec,
            pm_dec_mas * MAS_PER_YR_TO_RAD_PER_YR,
        )
    } else {
        (0.0, 0.0)
    };
    (
        (ra_rad + mu_ra * dt_years) as f32,
        (dec_rad + mu_dec * dt_years) as f32,
    )
}

/// Convert a Hipparcos star to a generic Star, optionally propagating proper motion.
///
/// `epoch_year`: Target year for proper motion propagation (e.g. 2025.0).
/// If None, the catalog position at the Hipparcos reference epoch (J1991.25) is used.
///
/// Proper motion near the celestial poles (|dec| > ~87°) is ignored because
/// the cos(dec) divisor becomes numerically unstable, following the same
/// convention as tetra3/cedar-solve.
#[cfg(feature = "hipparcos")]
pub(crate) fn star_from_hipparcos(
    star: &crate::catalogs::hipparcos::HipparcosStar,
    epoch_year: Option<f64>,
) -> Star {
    // Hipparcos reference epoch is J1991.25
    const HIPPARCOS_EPOCH_YEAR: f64 = 1991.25;
    let (ra, dec) = propagate_proper_motion(
        star.ra_rad,
        star.dec_rad,
        star.pm_ra,
        star.pm_dec,
        epoch_year.map(|year| year - HIPPARCOS_EPOCH_YEAR),
    );

    Star {
        id: star.hip as i64,
        ra_rad: ra,
        dec_rad: dec,
        mag: star.hp_to_v(),
    }
}

/// Convert a Gaia star to a generic Star, optionally propagating proper motion.
///
/// `epoch_year`: Target year for proper motion propagation (e.g. 2025.0).
/// If None, the catalog position at the Gaia DR3 reference epoch (J2016.0) is used.
///
/// The source_id is stored directly as `i64` — negative values indicate Hipparcos
/// gap-fill stars from the merged catalog.
pub(crate) fn star_from_gaia(
    star: &crate::catalogs::gaia::GaiaStar,
    epoch_year: Option<f64>,
) -> Star {
    // Gaia DR3 reference epoch is J2016.0
    const GAIA_EPOCH_YEAR: f64 = 2016.0;
    let (ra, dec) = propagate_proper_motion(
        (star.ra_deg as f64).to_radians(),
        (star.dec_deg as f64).to_radians(),
        star.pmra as f64,
        star.pmdec as f64,
        epoch_year.map(|year| year - GAIA_EPOCH_YEAR),
    );

    Star {
        id: star.source_id,
        ra_rad: ra,
        dec_rad: dec,
        mag: star.phot_g_mean_mag,
    }
}
