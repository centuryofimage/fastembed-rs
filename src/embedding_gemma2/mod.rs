/// Pixels per side of one vision patch.
const PATCH: u32 = 16;
/// Patches per side of the pool that forms one soft token.
const POOL: usize = 3;
/// Width of the soft tokens every encoder hands the text backbone.
const FEATURE_WIDTH: usize = 512;
/// The shared context window across all modalities.
const MAX_TOKENS: usize = 8192;
const DEFAULT_BATCH_SIZE: usize = 8;
/// Images or frames per vision encoder call.
const VISION_CHUNK: usize = 16;

mod audio;
mod init;
pub use init::*;

mod preprocess;
pub use preprocess::{
    EmbeddingGemma2Audio, EmbeddingGemma2Image, EmbeddingGemma2Preprocessor, EmbeddingGemma2Video,
};

mod r#impl;
pub use r#impl::EmbeddingGemma2Input;
