use rkyv::{Archive, Deserialize, Serialize};

/// A "Generic" star type that will be used for star matching
/// The star RA & Dec assume proper motion has already been applied to the observation epoch.
/// The magnitude is a generic "brightness" value that can be used for filtering, but the exact meaning is catalog-dependent.
#[derive(Debug, Clone, PartialEq, Archive, Serialize, Deserialize)]
pub struct Star {
    pub id: u64,
    pub ra_rad: f32,
    pub dec_rad: f32,
    pub mag: f32,
}

impl Star {
    /// Unit vector pointing to the star's position on the celestial sphere.
    pub fn uvec(&self) -> nalgebra::Vector3<f32> {
        let ra = self.ra_rad;
        let dec = self.dec_rad;
        // fast cosine, sine at once:
        let (rasin, racos) = ra.sin_cos();
        let (decsin, deccos) = dec.sin_cos();
        nalgebra::Vector3::new(deccos * racos, deccos * rasin, decsin)
    }
}

/// Convert a Hipparcos star to a generic Star, optionally propagating proper motion.
///
/// `epoch_year`: Target year for proper motion propagation (e.g. 2025.0).
/// If None, the catalog position at the Hipparcos reference epoch (J1991.25) is used.
///
/// Proper motion near the celestial poles (|dec| > ~87°) is ignored because
/// the cos(dec) divisor becomes numerically unstable, following the same
/// convention as tetra3/cedar-solve.
pub fn star_from_hipparcos(
    star: &crate::catalogs::hipparcos::HipparcosStar,
    epoch_year: Option<f64>,
) -> Star {
    // Hipparcos reference epoch is J1991.25
    const HIPPARCOS_EPOCH_YEAR: f64 = 1991.25;
    // Convert milliarcseconds/year to radians/year
    const MAS_PER_YR_TO_RAD_PER_YR: f64 =
        2.0 * std::f64::consts::PI / (3600.0 * 1000.0 * 360.0);

    let (ra, dec) = if let Some(target_year) = epoch_year {
        let dt_years = target_year - HIPPARCOS_EPOCH_YEAR;
        let cos_dec = star.dec_rad.cos();

        // Near the poles, cos(dec) -> 0 makes the RA correction blow up.
        // Following tetra3: skip proper motion for |dec| > ~87 degrees.
        let (mu_ra, mu_dec) = if cos_dec.abs() > 0.05 {
            let mu_alpha_cos_delta = star.pm_ra * MAS_PER_YR_TO_RAD_PER_YR;
            let mu_delta = star.pm_dec * MAS_PER_YR_TO_RAD_PER_YR;
            // pm_ra from Hipparcos is mu_alpha*cos(delta), so divide by cos(dec)
            (mu_alpha_cos_delta / cos_dec, mu_delta)
        } else {
            (0.0, 0.0)
        };

        let ra = star.ra_rad + mu_ra * dt_years;
        let dec = star.dec_rad + mu_dec * dt_years;
        (ra as f32, dec as f32)
    } else {
        (star.ra_rad as f32, star.dec_rad as f32)
    };

    Star {
        id: star.hip as u64,
        ra_rad: ra,
        dec_rad: dec,
        mag: star.hp_to_v(),
    }
}
