const DEFAULT_BATCH_SIZE: usize = 256;

mod utils;
pub(crate) use utils::ResizeFn;

mod init;
pub use init::*;

mod r#impl;
