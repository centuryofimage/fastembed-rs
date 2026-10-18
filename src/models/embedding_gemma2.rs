use crate::ModelInfo;
use std::{fmt::Display, str::FromStr};

/// Multimodal EmbeddingGemma 2 checkpoints.
///
/// Text, images, video and audio share one 768-dimensional space. Each checkpoint is a text
/// backbone plus optional vision and audio encoders that are downloaded only when enabled.
#[derive(Default, Debug, Clone, PartialEq, Eq, Hash)]
pub enum EmbeddingGemma2Model {
    /// onnx-community/embeddinggemma-2-ONNX
    #[default]
    EmbeddingGemma2,
    /// Quantized onnx-community/embeddinggemma-2-ONNX
    EmbeddingGemma2Q,
    /// Quantized (4-bit) onnx-community/embeddinggemma-2-ONNX
    EmbeddingGemma2Q4,
}

#[cfg(feature = "hf-hub")]
impl EmbeddingGemma2Model {
    /// Suffix of every graph file of this checkpoint, as in `onnx/model{suffix}.onnx`.
    pub(crate) fn graph_suffix(&self) -> &'static str {
        match self {
            Self::EmbeddingGemma2 => "",
            Self::EmbeddingGemma2Q => "_quantized",
            Self::EmbeddingGemma2Q4 => "_q4",
        }
    }

    /// Repository paths of one graph and its external tensor data.
    pub(crate) fn graph_files(&self, stem: &str) -> [String; 2] {
        let graph = format!("onnx/{stem}{}.onnx", self.graph_suffix());
        [graph.clone(), format!("{graph}_data")]
    }
}

pub fn models_list() -> Vec<ModelInfo<EmbeddingGemma2Model>> {
    vec![
        ModelInfo {
            model: EmbeddingGemma2Model::EmbeddingGemma2,
            dim: 768,
            description: String::from(
                "EmbeddingGemma 2 multimodal embeddings for text, images, video and audio",
            ),
            model_code: String::from("onnx-community/embeddinggemma-2-ONNX"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: vec![String::from("onnx/model.onnx_data")],
            output_key: None,
        },
        ModelInfo {
            model: EmbeddingGemma2Model::EmbeddingGemma2Q,
            dim: 768,
            description: String::from(
                "Quantized EmbeddingGemma 2 multimodal embeddings for text, images, video and audio",
            ),
            model_code: String::from("onnx-community/embeddinggemma-2-ONNX"),
            model_file: String::from("onnx/model_quantized.onnx"),
            additional_files: vec![String::from("onnx/model_quantized.onnx_data")],
            output_key: None,
        },
        ModelInfo {
            model: EmbeddingGemma2Model::EmbeddingGemma2Q4,
            dim: 768,
            description: String::from(
                "Quantized (4-bit) EmbeddingGemma 2 multimodal embeddings for text, images, video and audio",
            ),
            model_code: String::from("onnx-community/embeddinggemma-2-ONNX"),
            model_file: String::from("onnx/model_q4.onnx"),
            additional_files: vec![String::from("onnx/model_q4.onnx_data")],
            output_key: None,
        },
    ]
}

impl Display for EmbeddingGemma2Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl FromStr for EmbeddingGemma2Model {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        models_list()
            .into_iter()
            .find(|m| format!("{:?}", m.model).eq_ignore_ascii_case(s))
            .map(|m| m.model)
            .ok_or_else(|| format!("Unknown EmbeddingGemma 2 model: {s}"))
    }
}

impl TryFrom<String> for EmbeddingGemma2Model {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        value.parse()
    }
}

#[cfg(test)]
pub(crate) fn all_variants() -> Vec<EmbeddingGemma2Model> {
    fn _exhaustive_guard(m: &EmbeddingGemma2Model) {
        match m {
            EmbeddingGemma2Model::EmbeddingGemma2 => (),
            EmbeddingGemma2Model::EmbeddingGemma2Q => (),
            EmbeddingGemma2Model::EmbeddingGemma2Q4 => (),
        }
    }
    vec![
        EmbeddingGemma2Model::EmbeddingGemma2,
        EmbeddingGemma2Model::EmbeddingGemma2Q,
        EmbeddingGemma2Model::EmbeddingGemma2Q4,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_variant_has_model_info() {
        let listed: Vec<_> = models_list().into_iter().map(|i| i.model).collect();
        for variant in all_variants() {
            assert!(
                listed.contains(&variant),
                "{variant:?} is missing from models_list(); get_model_info would panic"
            );
        }
    }

    #[test]
    fn every_variant_parses_from_its_name() {
        for variant in all_variants() {
            assert_eq!(variant.to_string().parse(), Ok(variant.clone()));
        }
    }

    #[test]
    #[cfg(feature = "hf-hub")]
    fn every_variant_lists_its_text_backbone() {
        for info in models_list() {
            let [model_file, data_file] = info.model.graph_files("model");
            assert_eq!(info.model_file, model_file);
            assert_eq!(info.additional_files, vec![data_file]);
        }
    }
}
