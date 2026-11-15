//! FITS access for the image integration tests, via the `fitskit` crate.
#![allow(dead_code)]

pub use fitskit::Header;
use fitskit::{FitsFile, HduData, PixelData};

/// One image HDU: its header and its pixels as `f32`, row-major.
pub struct FitsImage {
    pub header: Header,
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<f32>,
}

/// Read the image in HDU `hdu_index` (0 = primary) of the file at `path`.
pub fn read_image(path: &str, hdu_index: usize) -> FitsImage {
    let fits = FitsFile::from_file(path).unwrap_or_else(|e| panic!("reading {path}: {e}"));
    let hdu = fits
        .iter()
        .nth(hdu_index)
        .unwrap_or_else(|| panic!("{path} has no HDU {hdu_index}"));
    let HduData::Image(image) = &hdu.data else {
        panic!("HDU {hdu_index} of {path} is not an image");
    };
    let pixels = match &image.pixels {
        PixelData::F32(v) => v.clone(),
        _ => image
            .scaled_values(
                hdu.header.get_float("BSCALE").unwrap_or(1.0),
                hdu.header.get_float("BZERO").unwrap_or(0.0),
            )
            .into_iter()
            .map(|v| v as f32)
            .collect(),
    };
    let width = image.width().expect("image has no NAXIS1") as u32;
    let height = image.height().expect("image has no NAXIS2") as u32;
    assert_eq!(pixels.len(), width as usize * height as usize);
    FitsImage {
        header: hdu.header.clone(),
        width,
        height,
        pixels,
    }
}

/// A numeric header value as `f64` (integer cards included).
pub fn get_f64(header: &Header, key: &str) -> Option<f64> {
    header.get_float(key)
}
