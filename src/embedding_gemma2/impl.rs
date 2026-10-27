use std::{io::Cursor, path::Path};

use image::DynamicImage;
use ndarray::{Array2, Array3, ArrayView2};
use ort::{
    session::{builder::SessionBuilder, Session},
    value::Tensor,
};
use tokenizers::Tokenizer;

use crate::{
    common::{init_session_builder, normalize, Error, Result},
    models::embedding_gemma2::{models_list, EmbeddingGemma2Model},
    Embedding, ModelInfo,
};

#[cfg(feature = "hf-hub")]
use super::init::EmbeddingGemma2InitOptions;
use super::{
    init::{
        EmbeddingGemma2, EmbeddingGemma2InitOptionsUserDefined, EmbeddingGemma2Sessions,
        EmbeddingGemma2Settings, SpecialTokens, UserDefinedEmbeddingGemma2Model, VisionTokenBudget,
    },
    preprocess::{
        EmbeddingGemma2Audio, EmbeddingGemma2Image, EmbeddingGemma2Preprocessor,
        EmbeddingGemma2Video,
    },
    DEFAULT_BATCH_SIZE, FEATURE_WIDTH, MAX_TOKENS, VISION_CHUNK,
};

/// One part of an interleaved input. A sequence of parts produces one embedding.
#[derive(Debug, Clone, PartialEq)]
pub enum EmbeddingGemma2Input {
    Text(String),
    Image(EmbeddingGemma2Image),
    Video(EmbeddingGemma2Video),
    Audio(EmbeddingGemma2Audio),
}

impl EmbeddingGemma2 {
    /// Download the checkpoint's text backbone and the enabled encoders, then load them.
    #[cfg(feature = "hf-hub")]
    pub fn try_new(options: EmbeddingGemma2InitOptions) -> Result<Self> {
        let EmbeddingGemma2InitOptions {
            model_name,
            environment,
            execution_providers,
            cache_dir,
            show_download_progress,
            intra_threads,
            session_config,
            settings,
            vision,
            audio,
        } = options;
        let model_info = Self::get_model_info(&model_name);
        let repo =
            crate::common::pull_from_hf(model_info.model_code, cache_dir, show_download_progress)?;
        let get = |file: &str| {
            repo.get(file).map_err(|e| Error::ModelRetrieval {
                file: file.into(),
                source: Box::new(e),
            })
        };
        // External data first, so the graph resolves it from the same snapshot directory.
        let graph = |stem: &str| -> Result<std::path::PathBuf> {
            let [graph, data] = model_name.graph_files(stem);
            get(&data)?;
            get(&graph)
        };
        let tokenizer = std::fs::read(get("tokenizer.json")?)?;
        let model = UserDefinedEmbeddingGemma2Model {
            text_graph: graph("model")?,
            vision_graph: vision.then(|| graph("vision_encoder")).transpose()?,
            audio_graph: audio.then(|| graph("audio_encoder")).transpose()?,
            tokenizer_file: tokenizer,
        };
        Self::try_new_from_user_defined(
            model,
            EmbeddingGemma2InitOptionsUserDefined {
                environment,
                execution_providers,
                intra_threads,
                session_config,
                settings,
            },
        )
    }

    /// Load graphs from disk, for example re-exported or provider-specific copies.
    pub fn try_new_from_user_defined(
        model: UserDefinedEmbeddingGemma2Model,
        options: EmbeddingGemma2InitOptionsUserDefined,
    ) -> Result<Self> {
        let settings = options.settings;
        let load = |path: &Path| -> Result<Session> {
            Ok(init_session_builder(
                options.environment.clone(),
                options.execution_providers.clone(),
                options.intra_threads,
                options.session_config.clone(),
            )?
            .commit_from_file(path)?)
        };
        let text = load(&model.text_graph)?;
        let vision = model.vision_graph.as_deref().map(load).transpose()?;
        let audio = model.audio_graph.as_deref().map(load).transpose()?;
        Self::new(text, vision, audio, &model.tokenizer_file, settings)
    }

    /// A session builder configured like the ones this type builds itself, for callers that
    /// build their own sessions for [`Self::try_new_from_sessions`].
    pub fn session_builder(
        options: &EmbeddingGemma2InitOptionsUserDefined,
    ) -> Result<SessionBuilder> {
        init_session_builder(
            options.environment.clone(),
            options.execution_providers.clone(),
            options.intra_threads,
            options.session_config.clone(),
        )
    }

    /// Wrap sessions the caller built, for example with dimension overrides or with encoders on
    /// different execution providers.
    pub fn try_new_from_sessions(
        sessions: EmbeddingGemma2Sessions,
        tokenizer_file: &[u8],
        settings: EmbeddingGemma2Settings,
    ) -> Result<Self> {
        let EmbeddingGemma2Sessions {
            text,
            vision,
            audio,
        } = sessions;
        Self::new(text, vision, audio, tokenizer_file, settings)
    }

    fn new(
        text: Session,
        vision: Option<Session>,
        audio: Option<Session>,
        tokenizer_file: &[u8],
        settings: EmbeddingGemma2Settings,
    ) -> Result<Self> {
        if let Some(dimensions) = settings.dimensions {
            if ![128, 256, 512, 768].contains(&dimensions) {
                return Err(Error::InvalidArgument(format!(
                    "EmbeddingGemma 2 supports 768, 512, 256 or 128 dimensions, not {dimensions}"
                )));
            }
        }
        let mut tokenizer = Tokenizer::from_bytes(tokenizer_file)
            .map_err(|e| Error::TokenizerConfig(e.to_string()))?;
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(None)
            .map_err(|e| Error::TokenizerConfig(e.to_string()))?;
        let id = |token: &str| {
            tokenizer
                .token_to_id(token)
                .ok_or_else(|| Error::TokenizerConfig(format!("tokenizer has no {token} token")))
        };
        let tokens = SpecialTokens {
            bos: id("<bos>")?,
            eos: id("<eos>")?,
            pad: id("<pad>")?,
            image_start: id("<|image>")?,
            image: id("<|image|>")?,
            image_end: id("<image|>")?,
            video: id("<|video|>")?,
            audio_start: id("<|audio>")?,
            audio: id("<|audio|>")?,
            audio_end: id("<audio|>")?,
        };
        Ok(Self {
            tokenizer,
            tokens,
            text,
            vision,
            audio,
            preprocessor: EmbeddingGemma2Preprocessor::new(settings),
            settings,
        })
    }

    pub fn list_supported_models() -> Vec<ModelInfo<EmbeddingGemma2Model>> {
        models_list()
    }

    pub fn get_model_info(model: &EmbeddingGemma2Model) -> ModelInfo<EmbeddingGemma2Model> {
        models_list()
            .into_iter()
            .find(|m| &m.model == model)
            .expect("Model not found in supported models list. This is a bug - please report it.")
    }

    pub fn settings(&self) -> EmbeddingGemma2Settings {
        self.settings
    }

    /// Width of every embedding this instance returns.
    pub fn dimensions(&self) -> usize {
        self.settings.dimensions.unwrap_or(768)
    }

    /// Whether the vision encoder is loaded, for images and video.
    pub fn supports_vision(&self) -> bool {
        self.vision.is_some()
    }

    pub fn supports_audio(&self) -> bool {
        self.audio.is_some()
    }

    /// A cloneable CPU preprocessor for this instance's budgets. Preprocess inputs on worker
    /// threads, then pass them to [`Self::embed_preprocessed_images`] or [`Self::embed_inputs`].
    pub fn preprocessor(&self) -> EmbeddingGemma2Preprocessor {
        self.preprocessor.clone()
    }

    /// Embed text. Prefix search queries with [`super::EMBEDDING_GEMMA2_QUERY_PROMPT`] and
    /// documents with `title: {title or "none"} | text: `. Text beyond the 8,192-token context
    /// is truncated.
    pub fn embed_text<S: AsRef<str> + Send + Sync>(
        &mut self,
        texts: impl AsRef<[S]>,
        batch_size: Option<usize>,
    ) -> Result<Vec<Embedding>> {
        let batch_size = positive_batch(batch_size)?;
        let mut output = Vec::with_capacity(texts.as_ref().len());
        for batch in texts.as_ref().chunks(batch_size) {
            let encodings = self
                .tokenizer
                .encode_batch(batch.iter().map(|text| text.as_ref()).collect(), true)
                .map_err(|e| Error::Tokenization(e.to_string()))?;
            let rows = encodings
                .iter()
                .map(|encoding| {
                    let mut ids = encoding.get_ids().to_vec();
                    if ids.len() > MAX_TOKENS {
                        ids.truncate(MAX_TOKENS - 1);
                        ids.push(self.tokens.eos);
                    }
                    ids
                })
                .collect();
            let (ids, mask) = self.pad_rows(rows);
            output.extend(self.run_backbone(ids, mask, empty(), empty(), empty())?);
        }
        Ok(output)
    }

    /// Embed images from files.
    pub fn embed<S: AsRef<Path> + Send + Sync>(
        &mut self,
        images: impl AsRef<[S]>,
        batch_size: Option<usize>,
    ) -> Result<Vec<Embedding>> {
        let batch_size = positive_batch(batch_size)?;
        let mut output = Vec::with_capacity(images.as_ref().len());
        for batch in images.as_ref().chunks(batch_size) {
            let images = batch
                .iter()
                .map(|path| {
                    image::ImageReader::open(path)?
                        .decode()
                        .map_err(|e| Error::ImageDecode(e.to_string()))
                })
                .collect::<Result<Vec<_>>>()?;
            output.extend(self.embed_images(images)?);
        }
        Ok(output)
    }

    /// Embed encoded image files, such as JPEG or PNG bytes.
    pub fn embed_bytes(
        &mut self,
        images: &[&[u8]],
        batch_size: Option<usize>,
    ) -> Result<Vec<Embedding>> {
        let batch_size = positive_batch(batch_size)?;
        let mut output = Vec::with_capacity(images.len());
        for batch in images.chunks(batch_size) {
            let images = batch
                .iter()
                .map(|bytes| {
                    image::ImageReader::new(Cursor::new(bytes))
                        .with_guessed_format()?
                        .decode()
                        .map_err(|e| Error::ImageDecode(e.to_string()))
                })
                .collect::<Result<Vec<_>>>()?;
            output.extend(self.embed_images(images)?);
        }
        Ok(output)
    }

    /// Embed decoded images in one batch.
    pub fn embed_images(&mut self, images: Vec<DynamicImage>) -> Result<Vec<Embedding>> {
        let preprocessed = images
            .into_iter()
            .map(|image| self.preprocessor.preprocess_image(image))
            .collect::<Result<_>>()?;
        self.embed_preprocessed_images(preprocessed)
    }

    /// Embed images preprocessed by [`EmbeddingGemma2Preprocessor::preprocess_image`] in one batch.
    ///
    /// Every row has `budget` image slots, so every batch of the same size has the same input
    /// shapes whatever the images' aspect ratios. Unused slots are masked and do not change the
    /// embedding. Accelerators that compile per shape then reuse one compiled graph.
    pub fn embed_preprocessed_images(
        &mut self,
        images: Vec<EmbeddingGemma2Image>,
    ) -> Result<Vec<Embedding>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }
        let budget = self.settings.image_budget;
        check_budget(&images, budget, "image")?;
        let features = self.encode_vision(&images.iter().collect::<Vec<_>>())?;
        let slots = budget.tokens();
        let tokens = self.tokens;
        let mut ids = Array2::<i64>::from_elem((images.len(), slots + 4), i64::from(tokens.image));
        let mut mask = Array2::<i64>::zeros(ids.dim());
        let mut padded = Array2::<f32>::zeros((images.len() * slots, FEATURE_WIDTH));
        let mut offset = 0;
        for (row, image) in images.iter().enumerate() {
            let count = image.soft_tokens();
            let layout = std::iter::once(tokens.bos)
                .chain(std::iter::once(tokens.image_start))
                .chain(std::iter::repeat_n(tokens.image, count))
                .chain([tokens.image_end, tokens.eos]);
            for (column, id) in layout.enumerate() {
                ids[[row, column]] = i64::from(id);
                mask[[row, column]] = 1;
            }
            padded
                .slice_mut(ndarray::s![row * slots..row * slots + count, ..])
                .assign(&features.slice(ndarray::s![offset..offset + count, ..]));
            offset += count;
        }
        self.run_backbone(ids, mask, padded, empty(), empty())
    }

    /// Embed interleaved inputs in one batch, one embedding per input.
    ///
    /// Media is placed between text exactly where it appears. Text parts take the same prefixes
    /// as [`Self::embed_text`].
    pub fn embed_inputs(
        &mut self,
        inputs: Vec<Vec<EmbeddingGemma2Input>>,
    ) -> Result<Vec<Embedding>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let tokens = self.tokens;
        let mut images = Vec::new();
        let mut frames = Vec::new();
        let mut clips = Vec::new();
        let mut rows = Vec::with_capacity(inputs.len());
        for input in &inputs {
            if input.is_empty() {
                return Err(Error::InvalidArgument(
                    "an input needs at least one part".into(),
                ));
            }
            // Without text, the reference processor separates media placeholders with spaces.
            let media_only = !input
                .iter()
                .any(|part| matches!(part, EmbeddingGemma2Input::Text(_)));
            let space = if media_only {
                let encoding = self
                    .tokenizer
                    .encode(" ", false)
                    .map_err(|e| Error::Tokenization(e.to_string()))?;
                encoding.get_ids().to_vec()
            } else {
                Vec::new()
            };
            let mut row = vec![tokens.bos];
            for (index, part) in input.iter().enumerate() {
                if index > 0 {
                    row.extend(&space);
                }
                match part {
                    EmbeddingGemma2Input::Text(text) => {
                        let encoding = self
                            .tokenizer
                            .encode(text.as_str(), false)
                            .map_err(|e| Error::Tokenization(e.to_string()))?;
                        row.extend(encoding.get_ids());
                    }
                    EmbeddingGemma2Input::Image(image) => {
                        row.push(tokens.image_start);
                        row.extend(std::iter::repeat_n(tokens.image, image.soft_tokens()));
                        row.push(tokens.image_end);
                        images.push(image);
                    }
                    EmbeddingGemma2Input::Video(video) => {
                        for frame in video.frames() {
                            row.push(tokens.image_start);
                            row.extend(std::iter::repeat_n(tokens.video, frame.soft_tokens()));
                            row.push(tokens.image_end);
                            frames.push(frame);
                        }
                    }
                    EmbeddingGemma2Input::Audio(audio) => {
                        row.push(tokens.audio_start);
                        row.extend(std::iter::repeat_n(tokens.audio, audio.soft_tokens()));
                        row.push(tokens.audio_end);
                        clips.push(audio);
                    }
                }
            }
            row.push(tokens.eos);
            if row.len() > MAX_TOKENS {
                return Err(Error::InvalidArgument(format!(
                    "input takes {} tokens; EmbeddingGemma 2 accepts {MAX_TOKENS}",
                    row.len()
                )));
            }
            rows.push(row);
        }
        check_budget(images.iter().copied(), self.settings.image_budget, "image")?;
        check_budget(
            frames.iter().copied(),
            self.settings.video_budget,
            "video frame",
        )?;
        let image_features = self.encode_vision(&images)?;
        let video_features = self.encode_vision(&frames)?;
        let audio_features = self.encode_audio(&clips)?;
        let (ids, mask) = self.pad_rows(rows);
        self.run_backbone(ids, mask, image_features, video_features, audio_features)
    }

    /// Embed videos preprocessed by [`EmbeddingGemma2Preprocessor::preprocess_video`] in one batch.
    pub fn embed_preprocessed_videos(
        &mut self,
        videos: Vec<EmbeddingGemma2Video>,
    ) -> Result<Vec<Embedding>> {
        self.embed_inputs(
            videos
                .into_iter()
                .map(|video| vec![EmbeddingGemma2Input::Video(video)])
                .collect(),
        )
    }

    /// Embed audio preprocessed by [`EmbeddingGemma2Preprocessor::preprocess_audio`] in one batch.
    pub fn embed_preprocessed_audio(
        &mut self,
        clips: Vec<EmbeddingGemma2Audio>,
    ) -> Result<Vec<Embedding>> {
        self.embed_inputs(
            clips
                .into_iter()
                .map(|audio| vec![EmbeddingGemma2Input::Audio(audio)])
                .collect(),
        )
    }

    /// Pad token rows to the longest with masked padding.
    fn pad_rows(&self, rows: Vec<Vec<u32>>) -> (Array2<i64>, Array2<i64>) {
        let length = rows.iter().map(Vec::len).max().unwrap_or(0);
        let mut ids = Array2::<i64>::from_elem((rows.len(), length), i64::from(self.tokens.pad));
        let mut mask = Array2::<i64>::zeros(ids.dim());
        for (index, row) in rows.iter().enumerate() {
            for (column, &id) in row.iter().enumerate() {
                ids[[index, column]] = i64::from(id);
                mask[[index, column]] = 1;
            }
        }
        (ids, mask)
    }

    /// Soft tokens for every image or frame, concatenated in order.
    #[tracing::instrument(name = "gemma_vision", level = "debug", skip_all, fields(images = images.len()))]
    fn encode_vision(&mut self, images: &[&EmbeddingGemma2Image]) -> Result<Array2<f32>> {
        if images.is_empty() {
            return Ok(empty());
        }
        let session = self
            .vision
            .as_mut()
            .ok_or_else(|| Error::InvalidArgument("the vision encoder is not loaded".into()))?;
        let mut parts = Vec::new();
        for chunk in images.chunks(VISION_CHUNK) {
            let pixels = stack3(chunk.iter().map(|image| image.pixel_values.view()))?;
            let positions = stack3(chunk.iter().map(|image| image.positions.view()))?;
            let outputs = session.run(ort::inputs![
                "pixel_values" => Tensor::from_array(pixels)?,
                "pixel_position_ids" => Tensor::from_array(positions)?,
            ])?;
            parts.push(vision_features(&outputs["image_features"], chunk)?);
        }
        concatenate(parts)
    }

    #[tracing::instrument(name = "gemma_audio", level = "debug", skip_all, fields(clips = clips.len()))]
    fn encode_audio(&mut self, clips: &[&EmbeddingGemma2Audio]) -> Result<Array2<f32>> {
        if clips.is_empty() {
            return Ok(empty());
        }
        let session = self
            .audio
            .as_mut()
            .ok_or_else(|| Error::InvalidArgument("the audio encoder is not loaded".into()))?;
        let frames = clips.iter().map(|clip| clip.mask.len()).max().unwrap_or(0);
        let mels = clips[0].features.ncols();
        let mut input = Array3::<f32>::zeros((clips.len(), frames, mels));
        let mut mask = Array2::<bool>::from_elem((clips.len(), frames), false);
        for (index, clip) in clips.iter().enumerate() {
            let length = clip.mask.len();
            input
                .slice_mut(ndarray::s![index, ..length, ..])
                .assign(&clip.features);
            for (column, &valid) in clip.mask.iter().enumerate() {
                mask[[index, column]] = valid;
            }
        }
        let outputs = session.run(ort::inputs![
            "input_features" => Tensor::from_array(input)?,
            "input_features_mask" => Tensor::from_array(mask)?,
        ])?;
        let expected = clips.iter().map(|clip| clip.soft_tokens()).sum();
        features(&outputs["audio_features"], expected, "audio")
    }

    #[tracing::instrument(name = "gemma_text", level = "debug", skip_all, fields(rows = ids.nrows()))]
    fn run_backbone(
        &mut self,
        ids: Array2<i64>,
        mask: Array2<i64>,
        image: Array2<f32>,
        video: Array2<f32>,
        audio: Array2<f32>,
    ) -> Result<Vec<Embedding>> {
        let rows = ids.nrows();
        let dimensions = self.dimensions();
        let outputs = self.text.run(ort::inputs![
            "input_ids" => Tensor::from_array(ids)?,
            "attention_mask" => Tensor::from_array(mask)?,
            "image_features" => Tensor::from_array(image)?,
            "video_features" => Tensor::from_array(video)?,
            "audio_features" => Tensor::from_array(audio)?,
        ])?;
        let (shape, data) = outputs["sentence_embedding"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::TensorExtraction(e.to_string()))?;
        if shape.len() != 2 || shape[0] != rows as i64 {
            return Err(Error::InvalidShape(format!(
                "sentence_embedding has shape {shape:?} for {rows} inputs"
            )));
        }
        let width = shape[1] as usize;
        let keep = dimensions.min(width);
        Ok(data
            .chunks_exact(width)
            .map(|row| normalize(&row[..keep]))
            .collect())
    }
}

fn positive_batch(batch_size: Option<usize>) -> Result<usize> {
    match batch_size.unwrap_or(DEFAULT_BATCH_SIZE) {
        0 => Err(Error::InvalidArgument(
            "batch_size must be greater than 0".into(),
        )),
        size => Ok(size),
    }
}

fn empty() -> Array2<f32> {
    Array2::zeros((0, FEATURE_WIDTH))
}

fn check_budget<'a>(
    images: impl IntoIterator<Item = &'a EmbeddingGemma2Image>,
    budget: VisionTokenBudget,
    kind: &str,
) -> Result<()> {
    match images.into_iter().find(|image| image.budget() != budget) {
        Some(image) => Err(Error::InvalidArgument(format!(
            "{kind} was preprocessed for {} tokens, but this model uses {}",
            image.budget().tokens(),
            budget.tokens()
        ))),
        None => Ok(()),
    }
}

fn stack3<'a, T: Copy + 'a>(views: impl Iterator<Item = ArrayView2<'a, T>>) -> Result<Array3<T>> {
    let views: Vec<_> = views.collect();
    ndarray::stack(ndarray::Axis(0), &views).map_err(|e| Error::InvalidShape(e.to_string()))
}

fn features(value: &ort::value::DynValue, expected: usize, encoder: &str) -> Result<Array2<f32>> {
    let (shape, data) = value
        .try_extract_tensor::<f32>()
        .map_err(|e| Error::TensorExtraction(e.to_string()))?;
    if shape.len() != 2 || shape[0] != expected as i64 || shape[1] != FEATURE_WIDTH as i64 {
        return Err(Error::InvalidShape(format!(
            "{encoder} encoder returned {shape:?}, expected [{expected}, {FEATURE_WIDTH}]"
        )));
    }
    Array2::from_shape_vec((expected, FEATURE_WIDTH), data.to_vec())
        .map_err(|e| Error::InvalidShape(e.to_string()))
}

/// The occupied soft tokens of each image, in order. Encoders return either only those, or every
/// pooled slot (`budget` per image, occupied ones first), which keeps their output shape fixed.
fn vision_features(
    value: &ort::value::DynValue,
    images: &[&EmbeddingGemma2Image],
) -> Result<Array2<f32>> {
    let occupied: usize = images.iter().map(|image| image.soft_tokens()).sum();
    let slots = images[0].budget().tokens();
    let (shape, data) = value
        .try_extract_tensor::<f32>()
        .map_err(|e| Error::TensorExtraction(e.to_string()))?;
    if shape.len() != 2 || shape[1] != FEATURE_WIDTH as i64 {
        return Err(Error::InvalidShape(format!(
            "vision encoder returned {shape:?}, expected rows of {FEATURE_WIDTH}"
        )));
    }
    let rows = shape[0] as usize;
    if rows == occupied {
        return Array2::from_shape_vec((rows, FEATURE_WIDTH), data.to_vec())
            .map_err(|e| Error::InvalidShape(e.to_string()));
    }
    if rows != images.len() * slots {
        return Err(Error::InvalidShape(format!(
            "vision encoder returned {rows} rows for {occupied} soft tokens in {} slots",
            images.len() * slots
        )));
    }
    let mut compact = Vec::with_capacity(occupied * FEATURE_WIDTH);
    for (index, image) in images.iter().enumerate() {
        let start = index * slots * FEATURE_WIDTH;
        compact.extend_from_slice(&data[start..start + image.soft_tokens() * FEATURE_WIDTH]);
    }
    Array2::from_shape_vec((occupied, FEATURE_WIDTH), compact)
        .map_err(|e| Error::InvalidShape(e.to_string()))
}

fn concatenate(parts: Vec<Array2<f32>>) -> Result<Array2<f32>> {
    if parts.len() == 1 {
        return Ok(parts.into_iter().next().unwrap_or_else(empty));
    }
    let views: Vec<_> = parts.iter().map(|part| part.view()).collect();
    ndarray::concatenate(ndarray::Axis(0), &views).map_err(|e| Error::InvalidShape(e.to_string()))
}
