//! The Gemma 4 audio front end: a log-mel spectrogram of 16 kHz mono audio.
//!
//! 20 ms periodic-Hann frames every 10 ms, a 512-point FFT magnitude, 128 HTK mel bands from 0
//! to 8 kHz without normalization, then `ln(mel + 0.001)`. The FFT and filters run in `f64`
//! like the NumPy reference.

use ndarray::Array2;

const SAMPLE_RATE: usize = 16_000;
const FRAME: usize = 320;
const HOP: usize = 160;
const FFT: usize = 512;
const BINS: usize = FFT / 2 + 1;
const MELS: usize = 128;
const MEL_FLOOR: f64 = 1e-3;
/// The reference feature extractor's default truncation, 30 seconds.
const MAX_SAMPLES: usize = 480_000;
/// The reference pads each clip with silence to a multiple of this many samples.
const PAD_MULTIPLE: usize = 128;

pub(crate) struct LogMelSpectrogram {
    /// `f32`, as the reference multiplies `f32` samples by an `f32` window before its FFT.
    window: Vec<f32>,
    /// `[BINS][MELS]`
    filters: Vec<[f64; MELS]>,
    fft: Fft,
}

impl LogMelSpectrogram {
    pub(crate) fn new() -> Self {
        let window = (0..FRAME)
            .map(|n| {
                (0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / FRAME as f64).cos()) as f32
            })
            .collect();
        Self {
            window,
            filters: mel_filters(),
            fft: Fft::new(FFT),
        }
    }

    /// Features `[frames, MELS]` and each frame's validity. Invalid frames are zero.
    pub(crate) fn compute(&self, samples: &[f32]) -> (Array2<f32>, Vec<bool>) {
        let real = samples.len().min(MAX_SAMPLES);
        let padded = real.div_ceil(PAD_MULTIPLE) * PAD_MULTIPLE;
        // Semicausal padding: FRAME / 2 zeros in front center the first frame on t = 0.
        let left = FRAME / 2;
        let total = left + padded;
        let frames = if total > FRAME {
            (total - (FRAME + 1)) / HOP + 1
        } else {
            0
        };
        let sample = |index: usize| -> f32 {
            index
                .checked_sub(left)
                .filter(|&i| i < real)
                .map_or(0.0, |i| samples[i])
        };
        let mut features = Array2::<f32>::zeros((frames, MELS));
        let mut mask = Vec::with_capacity(frames);
        let mut re = vec![0.0; FFT];
        let mut im = vec![0.0; FFT];
        for frame in 0..frames {
            // A frame is valid when its whole window, including the unfold's extra sample, is
            // real audio.
            let valid = frame * HOP + FRAME < left + real;
            mask.push(valid);
            if !valid {
                continue;
            }
            let start = frame * HOP;
            for i in 0..FFT {
                re[i] = if i < FRAME {
                    f64::from(sample(start + i) * self.window[i])
                } else {
                    0.0
                };
                im[i] = 0.0;
            }
            self.fft.run(&mut re, &mut im);
            let mut mel = [0.0f64; MELS];
            for bin in 0..BINS {
                let magnitude = (re[bin] * re[bin] + im[bin] * im[bin]).sqrt();
                for (band, weight) in mel.iter_mut().zip(&self.filters[bin]) {
                    *band += magnitude * weight;
                }
            }
            for (out, band) in features.row_mut(frame).iter_mut().zip(mel) {
                *out = (band + MEL_FLOOR).ln() as f32;
            }
        }
        (features, mask)
    }
}

/// Soft tokens after the encoder's two stride-2 subsampling convolutions.
pub(crate) fn soft_tokens(mask: &[bool]) -> usize {
    let mut mask = mask.to_vec();
    for _ in 0..2 {
        let length = (mask.len() + 2).checked_sub(3).map_or(0, |n| n / 2 + 1);
        mask = mask.into_iter().step_by(2).take(length).collect();
    }
    mask.into_iter().filter(|&valid| valid).count()
}

fn hertz_to_mel(hertz: f64) -> f64 {
    2595.0 * (1.0 + hertz / 700.0).log10()
}

fn mel_to_hertz(mel: f64) -> f64 {
    700.0 * (10f64.powf(mel / 2595.0) - 1.0)
}

/// Triangular HTK filters between evenly spaced mel points, as `[BINS][MELS]`.
fn mel_filters() -> Vec<[f64; MELS]> {
    let (low, high) = (hertz_to_mel(0.0), hertz_to_mel(8000.0));
    let edges: Vec<f64> = (0..MELS + 2)
        .map(|i| mel_to_hertz(low + (high - low) * i as f64 / (MELS + 1) as f64))
        .collect();
    let nyquist = (SAMPLE_RATE / 2) as f64;
    (0..BINS)
        .map(|bin| {
            let frequency = nyquist * bin as f64 / (BINS - 1) as f64;
            let mut row = [0.0; MELS];
            for (band, weight) in row.iter_mut().enumerate() {
                let down = (frequency - edges[band]) / (edges[band + 1] - edges[band]);
                let up = (edges[band + 2] - frequency) / (edges[band + 2] - edges[band + 1]);
                *weight = down.min(up).max(0.0);
            }
            row
        })
        .collect()
}

/// An in-place radix-2 complex FFT of one fixed power-of-two size.
struct Fft {
    cos: Vec<f64>,
    sin: Vec<f64>,
}

impl Fft {
    fn new(size: usize) -> Self {
        debug_assert!(size.is_power_of_two());
        let angle = |k: usize| -2.0 * std::f64::consts::PI * k as f64 / size as f64;
        Self {
            cos: (0..size / 2).map(|k| angle(k).cos()).collect(),
            sin: (0..size / 2).map(|k| angle(k).sin()).collect(),
        }
    }

    fn run(&self, re: &mut [f64], im: &mut [f64]) {
        let size = re.len();
        let mut j = 0;
        for i in 1..size {
            let mut bit = size >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j |= bit;
            if i < j {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let mut length = 2;
        while length <= size {
            let stride = size / length;
            for start in (0..size).step_by(length) {
                for k in 0..length / 2 {
                    let (c, s) = (self.cos[k * stride], self.sin[k * stride]);
                    let (a, b) = (start + k, start + k + length / 2);
                    let tr = re[b] * c - im[b] * s;
                    let ti = re[b] * s + im[b] * c;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
            length <<= 1;
        }
    }
}
