use std::path::PathBuf;

use ort::{environment::Environment, ep::ExecutionProviderDispatch, session::Session};
use tokenizers::Tokenizer;

use crate::{get_cache_dir, init::ProviderNames, models::embedding_gemma2::EmbeddingGemma2Model};

use super::preprocess::EmbeddingGemma2Preprocessor;

/// Text prefix for search queries. EmbeddingGemma 2 is trained with task prefixes on text inputs;
/// images, video and audio take none.
pub const EMBEDDING_GEMMA2_QUERY_PROMPT: &str = "task: search result | query: ";

/// The most soft tokens the vision encoder produces for one image or video frame.
///
/// Images are resized, preserving aspect ratio, to fit `tokens * 9` patches of 16x16 pixels.
/// Higher budgets keep more detail at a higher cost. Each budget produces different embeddings,
/// so vectors compared with each other must use the same budget.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum VisionTokenBudget {
    T70,
    T140,
    #[default]
    T280,
    T560,
    T1120,
}

impl VisionTokenBudget {
    pub const fn tokens(self) -> usize {
        match self {
            Self::T70 => 70,
            Self::T140 => 140,
            Self::T280 => 280,
            Self::T560 => 560,
            Self::T1120 => 1120,
        }
    }

    /// Patch rows in each vision encoder input, real and padding.
    pub const fn patches(self) -> usize {
        self.tokens() * super::POOL * super::POOL
    }
}

/// Model behavior shared by every constructor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct EmbeddingGemma2Settings {
    /// Soft-token budget for images. Defaults to 280.
    pub image_budget: VisionTokenBudget,
    /// Soft-token budget for each video frame. Defaults to 140.
    pub video_budget: VisionTokenBudget,
    /// Longer videos are subsampled uniformly to this many frames. Defaults to 32.
    pub max_video_frames: usize,
    /// Matryoshka truncation to 512, 256 or 128 dimensions before normalization. `None` keeps
    /// all 768.
    pub dimensions: Option<usize>,
}

impl Default for EmbeddingGemma2Settings {
    fn default() -> Self {
        Self {
            image_budget: VisionTokenBudget::T280,
            video_budget: VisionTokenBudget::T140,
            max_video_frames: 32,
            dimensions: None,
        }
    }
}

impl EmbeddingGemma2Settings {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_image_budget(mut self, budget: VisionTokenBudget) -> Self {
        self.image_budget = budget;
        self
    }

    pub fn with_video_budget(mut self, budget: VisionTokenBudget) -> Self {
        self.video_budget = budget;
        self
    }

    pub fn with_max_video_frames(mut self, frames: usize) -> Self {
        self.max_video_frames = frames;
        self
    }

    pub fn with_dimensions(mut self, dimensions: usize) -> Self {
        self.dimensions = Some(dimensions);
        self
    }
}

/// Options for [`EmbeddingGemma2::try_new`].
#[derive(Clone)]
#[non_exhaustive]
pub struct EmbeddingGemma2InitOptions {
    pub model_name: EmbeddingGemma2Model,
    pub environment: Option<Environment>,
    pub execution_providers: Vec<ExecutionProviderDispatch>,
    pub cache_dir: PathBuf,
    pub show_download_progress: bool,
    /// Number of intra-op threads for ONNX Runtime. `None` (the default) uses
    /// every available CPU core via `std::thread::available_parallelism`.
    pub intra_threads: Option<usize>,
    /// ONNX Runtime session configuration entries, applied with
    /// `SessionBuilder::with_config_entry`.
    pub session_config: Vec<(String, String)>,
    pub settings: EmbeddingGemma2Settings,
    /// Download and load the vision encoder, for images and video. Defaults to true.
    pub vision: bool,
    /// Download and load the audio encoder. Defaults to false, since it adds about 1.2 GB.
    pub audio: bool,
}

impl Default for EmbeddingGemma2InitOptions {
    fn default() -> Self {
        Self {
            model_name: EmbeddingGemma2Model::default(),
            environment: None,
            execution_providers: Vec::new(),
            cache_dir: get_cache_dir().into(),
            show_download_progress: true,
            intra_threads: None,
            session_config: Vec::new(),
            settings: EmbeddingGemma2Settings::default(),
            vision: true,
            audio: false,
        }
    }
}

impl EmbeddingGemma2InitOptions {
    pub fn new(model_name: EmbeddingGemma2Model) -> Self {
        Self {
            model_name,
            ..Default::default()
        }
    }

    pub fn with_cache_dir(mut self, cache_dir: PathBuf) -> Self {
        self.cache_dir = cache_dir;
        self
    }

    pub fn with_environment(mut self, environment: Environment) -> Self {
        self.environment = Some(environment);
        self
    }

    pub fn with_execution_providers(
        mut self,
        execution_providers: Vec<ExecutionProviderDispatch>,
    ) -> Self {
        self.execution_providers = execution_providers;
        self
    }

    pub fn with_show_download_progress(mut self, show_download_progress: bool) -> Self {
        self.show_download_progress = show_download_progress;
        self
    }

    pub fn with_intra_threads(mut self, intra_threads: usize) -> Self {
        self.intra_threads = Some(intra_threads);
        self
    }

    pub fn with_session_config(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.session_config.push((key.into(), value.into()));
        self
    }

    pub fn with_settings(mut self, settings: EmbeddingGemma2Settings) -> Self {
        self.settings = settings;
        self
    }

    pub fn with_vision(mut self, vision: bool) -> Self {
        self.vision = vision;
        self
    }

    pub fn with_audio(mut self, audio: bool) -> Self {
        self.audio = audio;
        self
    }
}

/// Options for [`EmbeddingGemma2::try_new_from_user_defined`].
#[derive(Clone, Default)]
#[non_exhaustive]
pub struct EmbeddingGemma2InitOptionsUserDefined {
    pub environment: Option<Environment>,
    pub execution_providers: Vec<ExecutionProviderDispatch>,
    pub intra_threads: Option<usize>,
    pub session_config: Vec<(String, String)>,
    pub settings: EmbeddingGemma2Settings,
}

impl EmbeddingGemma2InitOptionsUserDefined {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_environment(mut self, environment: Environment) -> Self {
        self.environment = Some(environment);
        self
    }

    pub fn with_execution_providers(
        mut self,
        execution_providers: Vec<ExecutionProviderDispatch>,
    ) -> Self {
        self.execution_providers = execution_providers;
        self
    }

    pub fn with_intra_threads(mut self, intra_threads: usize) -> Self {
        self.intra_threads = Some(intra_threads);
        self
    }

    pub fn with_session_config(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.session_config.push((key.into(), value.into()));
        self
    }

    pub fn with_settings(mut self, settings: EmbeddingGemma2Settings) -> Self {
        self.settings = settings;
        self
    }
}

impl std::fmt::Debug for EmbeddingGemma2InitOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingGemma2InitOptions")
            .field("model_name", &self.model_name)
            .field("environment", &self.environment)
            .field(
                "execution_providers",
                &ProviderNames(&self.execution_providers),
            )
            .field("cache_dir", &self.cache_dir)
            .field("show_download_progress", &self.show_download_progress)
            .field("intra_threads", &self.intra_threads)
            .field("session_config", &self.session_config)
            .field("settings", &self.settings)
            .field("vision", &self.vision)
            .field("audio", &self.audio)
            .finish()
    }
}

impl std::fmt::Debug for EmbeddingGemma2InitOptionsUserDefined {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingGemma2InitOptionsUserDefined")
            .field("environment", &self.environment)
            .field(
                "execution_providers",
                &ProviderNames(&self.execution_providers),
            )
            .field("intra_threads", &self.intra_threads)
            .field("session_config", &self.session_config)
            .field("settings", &self.settings)
            .finish()
    }
}

impl From<EmbeddingGemma2InitOptions> for EmbeddingGemma2InitOptionsUserDefined {
    fn from(options: EmbeddingGemma2InitOptions) -> Self {
        Self {
            environment: options.environment,
            execution_providers: options.execution_providers,
            intra_threads: options.intra_threads,
            session_config: options.session_config,
            settings: options.settings,
        }
    }
}

/// "Bring your own" EmbeddingGemma 2 graphs, such as re-exported or provider-specific copies.
///
/// Graphs are paths rather than bytes because their external `.onnx_data` weights resolve
/// relative to the graph file. Leave an encoder out to load only the modalities you need.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct UserDefinedEmbeddingGemma2Model {
    pub text_graph: PathBuf,
    pub vision_graph: Option<PathBuf>,
    pub audio_graph: Option<PathBuf>,
    pub tokenizer_file: Vec<u8>,
}

impl UserDefinedEmbeddingGemma2Model {
    pub fn new(text_graph: impl Into<PathBuf>, tokenizer_file: Vec<u8>) -> Self {
        Self {
            text_graph: text_graph.into(),
            vision_graph: None,
            audio_graph: None,
            tokenizer_file,
        }
    }

    pub fn with_vision_graph(mut self, path: impl Into<PathBuf>) -> Self {
        self.vision_graph = Some(path.into());
        self
    }

    pub fn with_audio_graph(mut self, path: impl Into<PathBuf>) -> Self {
        self.audio_graph = Some(path.into());
        self
    }
}

/// Sessions the caller built, for [`EmbeddingGemma2::try_new_from_sessions`].
#[non_exhaustive]
pub struct EmbeddingGemma2Sessions {
    pub text: Session,
    pub vision: Option<Session>,
    pub audio: Option<Session>,
}

impl EmbeddingGemma2Sessions {
    pub fn new(text: Session) -> Self {
        Self {
            text,
            vision: None,
            audio: None,
        }
    }

    pub fn with_vision(mut self, session: Session) -> Self {
        self.vision = Some(session);
        self
    }

    pub fn with_audio(mut self, session: Session) -> Self {
        self.audio = Some(session);
        self
    }
}

/// Token ids the model's input layout uses, read from its tokenizer.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SpecialTokens {
    pub bos: u32,
    pub eos: u32,
    pub pad: u32,
    pub image_start: u32,
    pub image: u32,
    pub image_end: u32,
    pub video: u32,
    pub audio_start: u32,
    pub audio: u32,
    pub audio_end: u32,
}

/// EmbeddingGemma 2: text, image, video and audio embeddings in one space.
pub struct EmbeddingGemma2 {
    pub(crate) tokenizer: Tokenizer,
    pub(crate) tokens: SpecialTokens,
    pub(crate) text: Session,
    pub(crate) vision: Option<Session>,
    pub(crate) audio: Option<Session>,
    pub(crate) preprocessor: EmbeddingGemma2Preprocessor,
    pub(crate) settings: EmbeddingGemma2Settings,
}
