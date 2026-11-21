#![forbid(unsafe_code)]

pub mod fpquant;
pub mod gguf;
pub mod quant;
pub mod safetensors;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    I8,
    /// Raw bytes: the packed experts of a release (NVFP4 nibbles) and their e4m3 scales.
    U8,
    /// e4m3 bytes: MXFP8 weights.
    F8E4M3,
    I32,
    I64,
    Mls2,
    Mls3,
    Mls4,
    Mls8,
    /// 4-bit rows read whole on the host (Engram memory tables), with one f16 scale per row in `<name>.scales`.
    RowI4,
}

impl DType {
    pub fn size_of_elems(self, n: usize) -> usize {
        match self {
            DType::F32 | DType::I32 => n * 4,
            DType::I64 => n * 8,
            DType::F16 | DType::BF16 => n * 2,
            DType::I8 | DType::U8 | DType::F8E4M3 => n,
            // MLS2 packs 4 codes per byte, MLS3 8 codes per 3 bytes; quantized tensors always have a K that is a
            // multiple of 128.
            DType::Mls2 => n / 4,
            DType::Mls3 => n / 8 * 3,
            DType::Mls4 => n / 2,
            DType::Mls8 => n,
            // A row of K 4-bit codes is K / 2 bytes (Engram rows are 96 wide: 48 bytes).
            DType::RowI4 => n / 2,
        }
    }
}

pub struct TensorInfo {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub offset: usize,
    pub size: usize,
}

impl TensorFile {
    pub fn new(file: std::fs::File, offset: u64, size: usize) -> Self {
        TensorFile { file, offset, size }
    }
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// One tensor of a file, read with positional reads at any time (tables kept on disk, e.g. the n-gram lookup table).
pub struct TensorFile {
    file: std::fs::File,
    offset: u64,
    pub size: usize,
}

impl TensorFile {
    /// The same tensor behind a file opened again, so that access hints set on it (which belong to the open file
    /// description, shared by duplicated descriptors) do not reach the loader's sequential reads. Unchanged when
    /// the file cannot be opened again.
    pub fn reopened(self) -> Self {
        use std::os::fd::AsRawFd;
        match std::fs::File::open(format!("/proc/self/fd/{}", self.file.as_raw_fd())) {
            Ok(file) => TensorFile { file, ..self },
            Err(_) => self,
        }
    }

    /// The file and the byte range of the tensor in it, for access hints.
    pub fn range(&self, start: usize, len: usize) -> Option<(&std::fs::File, u64, u64)> {
        (start + len <= self.size).then(|| (&self.file, self.offset + start as u64, len as u64))
    }

    pub fn read_at(&self, start: usize, buf: &mut [u8]) -> std::io::Result<()> {
        use std::os::unix::fs::FileExt;
        if start + buf.len() > self.size {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "read past tensor end",
            ));
        }
        self.file.read_exact_at(buf, self.offset + start as u64)
    }
}

/// Read-only set of tensors stored in files; data is read with positional reads.
pub trait TensorSource {
    fn info(&self, name: &str) -> Option<&TensorInfo>;
    fn names(&self) -> Vec<&str>;
    /// Reads `buf.len()` bytes of tensor `name` starting at byte `start` of the tensor.
    fn read(&self, name: &str, start: usize, buf: &mut [u8]) -> std::io::Result<()>;

    /// A handle reading tensor `name` after the source is dropped (None if the source cannot provide one).
    fn open_tensor(&self, _name: &str) -> Option<TensorFile> {
        None
    }

    fn read_all(&self, name: &str) -> anyhow::Result<Vec<u8>> {
        let info = self
            .info(name)
            .ok_or_else(|| anyhow::anyhow!("missing tensor {name}"))?;
        let mut v = vec![0u8; info.size];
        self.read(name, 0, &mut v)?;
        Ok(v)
    }
}
