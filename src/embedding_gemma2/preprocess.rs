use std::sync::Arc;

use image::{imageops::FilterType, DynamicImage};
use ndarray::Array2;

use crate::{
    common::{Error, Result},
    image_embedding::ResizeFn,
};

use super::{
    audio::{self, LogMelSpectrogram},
    init::{EmbeddingGemma2Settings, VisionTokenBudget},
    PATCH, POOL,
};

/// One image or video frame, patchified and padded for the vision encoder.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingGemma2Image {
    /// `[budget * 9, 768]`: 16x16 RGB patches in row-major order, then zero padding.
    pub(crate) pixel_values: Array2<f32>,
    /// `[budget * 9, 2]`: each patch's `(column, row)`, then `-1` padding.
    pub(crate) positions: Array2<i64>,
    pub(crate) soft_tokens: usize,
    pub(crate) budget: VisionTokenBudget,
}

impl EmbeddingGemma2Image {
    /// Soft tokens this image contributes: at most the budget, depending on its aspect ratio.
    pub fn soft_tokens(&self) -> usize {
        self.soft_tokens
    }

    pub fn budget(&self) -> VisionTokenBudget {
        self.budget
    }
}

/// Sampled video frames, each preprocessed at the video budget.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingGemma2Video {
    pub(crate) frames: Vec<EmbeddingGemma2Image>,
}

impl EmbeddingGemma2Video {
    pub fn frames(&self) -> &[EmbeddingGemma2Image] {
        &self.frames
    }

    /// Soft tokens across all frames.
    pub fn soft_tokens(&self) -> usize {
        self.frames
            .iter()
            .map(EmbeddingGemma2Image::soft_tokens)
            .sum()
    }
}

/// A log-mel spectrogram for the audio encoder.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingGemma2Audio {
    /// `[frames, 128]`, zero where `mask` is false.
    pub(crate) features: Array2<f32>,
    pub(crate) mask: Vec<bool>,
    pub(crate) soft_tokens: usize,
}

impl EmbeddingGemma2Audio {
    /// Soft tokens this clip contributes, 25 per second.
    pub fn soft_tokens(&self) -> usize {
        self.soft_tokens
    }
}

/// CPU preprocessing for every EmbeddingGemma 2 modality. Cloneable and thread-safe, so batches
/// can be preprocessed on worker threads while the model runs.
#[derive(Clone)]
pub struct EmbeddingGemma2Preprocessor {
    settings: EmbeddingGemma2Settings,
    resize: Option<Arc<ResizeFn>>,
    mel: Arc<LogMelSpectrogram>,
}

impl EmbeddingGemma2Preprocessor {
    pub fn new(settings: EmbeddingGemma2Settings) -> Self {
        Self {
            settings,
            resize: None,
            mel: Arc::new(LogMelSpectrogram::new()),
        }
    }

    /// Use a custom image resize implementation. The model was trained with bicubic resizing.
    pub fn with_resize<F>(mut self, resize: F) -> Self
    where
        F: Fn(DynamicImage, u32, u32, FilterType) -> Result<DynamicImage> + Send + Sync + 'static,
    {
        self.resize = Some(Arc::new(resize));
        self
    }

    pub fn settings(&self) -> EmbeddingGemma2Settings {
        self.settings
    }

    /// Preprocess an image at the image budget.
    pub fn preprocess_image(&self, image: DynamicImage) -> Result<EmbeddingGemma2Image> {
        self.patchify(image, self.settings.image_budget)
    }

    /// Preprocess decoded video frames, already sampled at the rate the caller wants (the model was
    /// trained on one frame per second). More than `max_video_frames` frames are subsampled
    /// uniformly.
    pub fn preprocess_video(&self, frames: Vec<DynamicImage>) -> Result<EmbeddingGemma2Video> {
        if frames.is_empty() {
            return Err(Error::InvalidArgument(
                "a video needs at least one frame".into(),
            ));
        }
        let limit = self.settings.max_video_frames.max(1);
        let keep = uniform_indices(frames.len(), limit);
        let frames = frames
            .into_iter()
            .enumerate()
            .filter(|(index, _)| keep.binary_search(index).is_ok())
            .map(|(_, frame)| self.patchify(frame, self.settings.video_budget))
            .collect::<Result<_>>()?;
        Ok(EmbeddingGemma2Video { frames })
    }

    /// Preprocess mono 16 kHz samples in -1 to 1. Clips longer than 30 seconds are truncated.
    pub fn preprocess_audio(&self, samples: &[f32]) -> Result<EmbeddingGemma2Audio> {
        let (features, mask) = self.mel.compute(samples);
        let soft_tokens = audio::soft_tokens(&mask);
        if soft_tokens == 0 {
            return Err(Error::InvalidArgument("audio clip is too short".into()));
        }
        Ok(EmbeddingGemma2Audio {
            features,
            mask,
            soft_tokens,
        })
    }

    fn patchify(
        &self,
        image: DynamicImage,
        budget: VisionTokenBudget,
    ) -> Result<EmbeddingGemma2Image> {
        let max_patches = budget.patches();
        let (height, width) = target_size(image.height(), image.width(), max_patches)?;
        let image = if (image.width(), image.height()) == (width, height) {
            image
        } else {
            match &self.resize {
                Some(resize) => resize(image, width, height, FilterType::CatmullRom)?,
                None => image.resize_exact(width, height, FilterType::CatmullRom),
            }
        };
        let rgb = image.into_rgb8();
        if (rgb.width(), rgb.height()) != (width, height) {
            return Err(Error::ImageTransform(format!(
                "resize returned {}x{}, expected {width}x{height}",
                rgb.width(),
                rgb.height()
            )));
        }
        let (rows, columns) = ((height / PATCH) as usize, (width / PATCH) as usize);
        let patch = PATCH as usize;
        let mut pixel_values = Array2::<f32>::zeros((max_patches, patch * patch * 3));
        let mut positions = Array2::<i64>::from_elem((max_patches, 2), -1);
        let raw = rgb.as_raw();
        let stride = width as usize * 3;
        for row in 0..rows {
            for column in 0..columns {
                let index = row * columns + column;
                let mut values = pixel_values.row_mut(index);
                let values = values
                    .as_slice_mut()
                    .ok_or_else(|| Error::Other("patch row is not contiguous".into()))?;
                for y in 0..patch {
                    let start = (row * patch + y) * stride + column * patch * 3;
                    for (offset, &value) in raw[start..start + patch * 3].iter().enumerate() {
                        values[y * patch * 3 + offset] = f32::from(value) * RESCALE;
                    }
                }
                positions[[index, 0]] = column as i64;
                positions[[index, 1]] = row as i64;
            }
        }
        Ok(EmbeddingGemma2Image {
            pixel_values,
            positions,
            soft_tokens: rows * columns / (POOL * POOL),
            budget,
        })
    }
}

const RESCALE: f32 = 1.0 / 255.0;

/// The largest aspect-preserving size within `max_patches` whose sides are multiples of the
/// pooled patch size, as `(height, width)`.
pub(crate) fn target_size(height: u32, width: u32, max_patches: usize) -> Result<(u32, u32)> {
    if height == 0 || width == 0 {
        return Err(Error::InvalidArgument("image has no pixels".into()));
    }
    let side = (POOL as u32) * PATCH;
    let factor = ((max_patches as f64 * f64::from(PATCH * PATCH))
        / (f64::from(height) * f64::from(width)))
    .sqrt();
    let mut target_height = ((factor * f64::from(height)) / f64::from(side)).floor() as u32 * side;
    let mut target_width = ((factor * f64::from(width)) / f64::from(side)).floor() as u32 * side;
    let max_side = (max_patches / (POOL * POOL)) as u32 * side;
    if target_height == 0 && target_width == 0 {
        return Err(Error::InvalidArgument(
            "image is too small to patchify".into(),
        ));
    }
    if target_height == 0 {
        target_height = side;
        target_width = ((width / height) * side).min(max_side);
    } else if target_width == 0 {
        target_width = side;
        target_height = ((height / width) * side).min(max_side);
    }
    Ok((target_height, target_width))
}

/// `limit` indices spread evenly over `0..count`, or all of them when they fit.
fn uniform_indices(count: usize, limit: usize) -> Vec<usize> {
    if count <= limit {
        return (0..count).collect();
    }
    if limit == 1 {
        return vec![0];
    }
    let step = (count - 1) as f64 / (limit - 1) as f64;
    let mut indices: Vec<usize> = (0..limit).map(|i| (i as f64 * step) as usize).collect();
    indices.dedup();
    indices
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    fn preprocessor() -> EmbeddingGemma2Preprocessor {
        EmbeddingGemma2Preprocessor::new(
            EmbeddingGemma2Settings::new()
                .with_image_budget(VisionTokenBudget::T70)
                .with_video_budget(VisionTokenBudget::T70)
                .with_max_video_frames(4),
        )
    }

    fn solid(width: u32, height: u32) -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_pixel(width, height, image::Rgb([255, 0, 0])))
    }

    #[test]
    fn target_size_fits_the_budget_in_pooled_patches() {
        assert_eq!(target_size(1000, 1000, 630).unwrap(), (384, 384));
        assert_eq!(target_size(1000, 2000, 630).unwrap(), (240, 528));
        assert_eq!(target_size(3000, 4000, 2520).unwrap(), (672, 912));
        for (height, width) in [(1, 1), (17, 913), (4000, 3000), (768, 768)] {
            for budget in [VisionTokenBudget::T70, VisionTokenBudget::T280] {
                let (h, w) = target_size(height, width, budget.patches()).unwrap();
                assert_eq!((h % 48, w % 48), (0, 0));
                assert!((h / 16) as usize * (w / 16) as usize <= budget.patches());
            }
        }
    }

    #[test]
    fn target_size_keeps_one_pool_row_for_extreme_aspect_ratios() {
        assert_eq!(target_size(10, 2000, 630).unwrap(), (48, 3360));
        assert_eq!(target_size(2000, 10, 630).unwrap(), (3360, 48));
        assert!(target_size(0, 10, 630).is_err());
    }

    #[test]
    fn image_patches_are_row_major_then_padded() {
        let mut rgb = RgbImage::new(384, 384);
        for (x, y, pixel) in rgb.enumerate_pixels_mut() {
            *pixel = image::Rgb([(x % 256) as u8, (y % 256) as u8, 7]);
        }
        let image = preprocessor()
            .preprocess_image(DynamicImage::ImageRgb8(rgb))
            .unwrap();
        assert_eq!(image.soft_tokens(), 64);
        assert_eq!(image.budget(), VisionTokenBudget::T70);
        assert_eq!(image.pixel_values.dim(), (630, 768));
        assert_eq!(image.positions.dim(), (630, 2));

        // Patch 25 is row 1, column 1: its first pixel is (16, 16) and the next row starts 48
        // values later.
        let patch = image.pixel_values.row(25);
        assert_eq!(image.positions.row(25).to_vec(), vec![1, 1]);
        assert_eq!(patch[0], 16.0 * RESCALE);
        assert_eq!(patch[1], 16.0 * RESCALE);
        assert_eq!(patch[2], 7.0 * RESCALE);
        assert_eq!(patch[3], 17.0 * RESCALE);
        assert_eq!(patch[48 + 1], 17.0 * RESCALE);
        assert_eq!(image.positions.row(575).to_vec(), vec![23, 23]);

        for row in 576..630 {
            assert_eq!(image.positions.row(row).to_vec(), vec![-1, -1]);
            assert!(image
                .pixel_values
                .row(row)
                .iter()
                .all(|&value| value == 0.0));
        }
    }

    #[test]
    fn custom_resize_must_return_the_requested_size() {
        let preprocessor = preprocessor()
            .with_resize(|image, _, _, _| Ok(image.resize_exact(10, 10, FilterType::Nearest)));
        assert!(matches!(
            preprocessor.preprocess_image(solid(1000, 1000)),
            Err(Error::ImageTransform(_))
        ));
    }

    #[test]
    fn video_frames_are_subsampled_uniformly() {
        assert_eq!(uniform_indices(3, 4), vec![0, 1, 2]);
        assert_eq!(uniform_indices(10, 4), vec![0, 3, 6, 9]);
        assert_eq!(uniform_indices(10, 1), vec![0]);
        let indices = uniform_indices(100, 32);
        assert_eq!((indices.len(), indices[0], indices[31]), (32, 0, 99));

        let video = preprocessor()
            .preprocess_video((0..10).map(|_| solid(384, 384)).collect())
            .unwrap();
        assert_eq!(video.frames().len(), 4);
        assert_eq!(video.soft_tokens(), 4 * 64);
        assert!(preprocessor().preprocess_video(Vec::new()).is_err());
    }

    #[test]
    fn audio_needs_at_least_one_soft_token() {
        assert!(preprocessor().preprocess_audio(&[0.0; 100]).is_err());
        let clip = preprocessor().preprocess_audio(&[0.0; 16_000]).unwrap();
        assert_eq!(clip.soft_tokens(), 25);
    }
}
