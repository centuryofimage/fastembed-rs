#![cfg(all(feature = "image-models", feature = "hf-hub"))]

use fastembed::{
    Embedding, EmbeddingGemma2, EmbeddingGemma2InitOptions, EmbeddingGemma2Input,
    EmbeddingGemma2Model, EmbeddingGemma2Settings, VisionTokenBudget,
    EMBEDDING_GEMMA2_QUERY_PROMPT,
};

/// A small epsilon value for floating point comparisons.
const EPS: f32 = 2e-2;

const DOCUMENTS: [&str; 2] = [
    "task: search result | query: blue cat",
    "title: none | text: A red cat sitting on a mat.",
];

// tests/assets/image_0.png is a blue cat
// tests/assets/image_1.png is a red cat
const IMAGES: [&str; 2] = ["tests/assets/image_0.png", "tests/assets/image_1.png"];

/// Sums of each document's embedding. The full precision values match transformers'
/// EmbeddingGemma2Processor with ONNX Runtime on the same graphs.
fn expected_sums(model: &EmbeddingGemma2Model) -> [f32; 2] {
    match model {
        EmbeddingGemma2Model::EmbeddingGemma2 => [0.385064, 0.965766],
        EmbeddingGemma2Model::EmbeddingGemma2Q => [0.38833222, 0.9711747],
        EmbeddingGemma2Model::EmbeddingGemma2Q4 => [0.44552585, 0.936469],
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn assert_normalized(embeddings: &[Embedding], dimensions: usize) {
    for embedding in embeddings {
        assert_eq!(embedding.len(), dimensions);
        assert!((dot(embedding, embedding) - 1.0).abs() < 1e-4);
    }
}

fn load(settings: EmbeddingGemma2Settings, audio: bool) -> EmbeddingGemma2 {
    EmbeddingGemma2::try_new(
        EmbeddingGemma2InitOptions::new(EmbeddingGemma2Model::EmbeddingGemma2)
            .with_settings(settings)
            .with_audio(audio),
    )
    .unwrap()
}

#[test]
fn test_embedding_gemma2_text() {
    for supported_model in EmbeddingGemma2::list_supported_models() {
        let mut model = EmbeddingGemma2::try_new(
            EmbeddingGemma2InitOptions::new(supported_model.model.clone()).with_vision(false),
        )
        .unwrap();
        assert!(!model.supports_vision());

        let embeddings = model.embed_text(DOCUMENTS, None).unwrap();
        assert_normalized(&embeddings, supported_model.dim);

        for (embedding, expected) in embeddings.iter().zip(expected_sums(&supported_model.model)) {
            let sum: f32 = embedding.iter().sum();
            assert!(
                (sum - expected).abs() < EPS,
                "{}: {sum} != {expected} (expected)",
                supported_model.model
            );
        }

        assert!(model.embed(IMAGES, None).is_err());
    }
}

#[test]
fn test_embedding_gemma2_images() {
    let mut model = load(EmbeddingGemma2Settings::default(), false);

    let image_embeddings = model.embed(IMAGES, None).unwrap();
    assert_normalized(&image_embeddings, 768);

    let bytes: Vec<Vec<u8>> = IMAGES
        .iter()
        .map(|path| std::fs::read(path).unwrap())
        .collect();
    let bytes: Vec<&[u8]> = bytes.iter().map(Vec::as_slice).collect();
    let byte_embeddings = model.embed_bytes(&bytes, None).unwrap();
    assert_eq!(image_embeddings, byte_embeddings);

    // Padding every row to the token budget leaves each image's embedding unchanged.
    let alone = model.embed(&IMAGES[..1], None).unwrap();
    assert!(dot(&alone[0], &image_embeddings[0]) > 0.9999);

    let queries =
        ["blue cat", "red cat"].map(|query| format!("{EMBEDDING_GEMMA2_QUERY_PROMPT}{query}"));
    let text_embeddings = model.embed_text(queries, None).unwrap();
    for (index, query) in text_embeddings.iter().enumerate() {
        let other = 1 - index;
        assert!(dot(query, &image_embeddings[index]) > dot(query, &image_embeddings[other]));
    }
}

#[test]
fn test_embedding_gemma2_interleaved_and_video() {
    let mut model = load(EmbeddingGemma2Settings::default(), false);
    let preprocessor = model.preprocessor();
    let cat = image::open(IMAGES[0]).unwrap();
    let red_cat = image::open(IMAGES[1]).unwrap();
    let images = model.embed(IMAGES, None).unwrap();

    let embeddings = model
        .embed_inputs(vec![
            vec![
                EmbeddingGemma2Input::Text("A photo of ".into()),
                EmbeddingGemma2Input::Image(preprocessor.preprocess_image(cat.clone()).unwrap()),
                EmbeddingGemma2Input::Text(" next to some text".into()),
            ],
            vec![
                EmbeddingGemma2Input::Image(preprocessor.preprocess_image(cat.clone()).unwrap()),
                EmbeddingGemma2Input::Image(preprocessor.preprocess_image(red_cat).unwrap()),
            ],
        ])
        .unwrap();
    assert_normalized(&embeddings, 768);
    assert!(model.embed_inputs(vec![Vec::new()]).is_err());

    let video = preprocessor
        .preprocess_video(vec![cat.clone(), cat.fliph(), cat.rotate180()])
        .unwrap();
    assert_eq!(video.frames().len(), 3);
    let videos = model.embed_preprocessed_videos(vec![video]).unwrap();
    assert_normalized(&videos, 768);
    assert!(dot(&videos[0], &images[0]) > dot(&videos[0], &images[1]));

    // Inputs preprocessed for another budget are rejected.
    let t70 = model
        .preprocessor()
        .settings()
        .with_image_budget(VisionTokenBudget::T70);
    let image = fastembed::EmbeddingGemma2Preprocessor::new(t70)
        .preprocess_image(cat)
        .unwrap();
    assert!(model.embed_preprocessed_images(vec![image]).is_err());
}

#[test]
fn test_embedding_gemma2_audio() {
    let mut model = load(EmbeddingGemma2Settings::default(), true);
    assert!(model.supports_audio());

    // 1.5 seconds of two sines at 16 kHz.
    let samples: Vec<f32> = (0..24_000)
        .map(|n| {
            let t = std::f64::consts::TAU * n as f64 / 16_000.0;
            (0.5 * (440.0 * t).sin() + 0.25 * (1250.0 * t).sin()) as f32
        })
        .collect();
    let audio = model.preprocessor().preprocess_audio(&samples).unwrap();
    assert_eq!(audio.soft_tokens(), 38);

    let embeddings = model.embed_preprocessed_audio(vec![audio]).unwrap();
    assert_normalized(&embeddings, 768);
    let sum: f32 = embeddings[0].iter().sum();
    assert!((sum - -0.049087).abs() < EPS, "{sum}");
}

#[test]
fn test_embedding_gemma2_matryoshka_dimensions() {
    let mut full = load(EmbeddingGemma2Settings::default(), false);
    let mut truncated = load(EmbeddingGemma2Settings::new().with_dimensions(256), false);
    assert_eq!(truncated.dimensions(), 256);

    let full = full.embed_text(DOCUMENTS, None).unwrap();
    let truncated = truncated.embed_text(DOCUMENTS, None).unwrap();
    assert_normalized(&truncated, 256);
    for (full, truncated) in full.iter().zip(&truncated) {
        let norm = dot(&full[..256], &full[..256]).sqrt();
        assert!((dot(&full[..256], truncated) / norm - 1.0).abs() < 1e-4);
    }

    assert!(EmbeddingGemma2::try_new(
        EmbeddingGemma2InitOptions::default()
            .with_vision(false)
            .with_settings(EmbeddingGemma2Settings::new().with_dimensions(300)),
    )
    .is_err());
}
