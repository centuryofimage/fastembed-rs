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
