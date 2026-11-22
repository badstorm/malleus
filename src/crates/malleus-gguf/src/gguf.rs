use crate::{DType, TensorInfo, TensorSource};
use anyhow::{Context, Result, bail, ensure};
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, Write};
use std::os::unix::fs::FileExt;
use std::path::Path;

const MAGIC: &[u8; 4] = b"GGUF";
const VERSION: u32 = 3;
pub const DEFAULT_ALIGNMENT: usize = 32;

/// File extension of Malleus model files. They use the GGUF v3 container with Malleus tensor types, which
/// other GGUF engines cannot run: the distinct extension and the `openmalleus.` architecture prefix make that clear.
pub const EXTENSION: &str = "malleus";
/// Prefix of `general.architecture` in Malleus model files (e.g. `openmalleus.qwen35`).
///
/// It reads `openmalleus.` and it stays that way, as do the `openmalleus.*` metadata keys written by the converter:
/// the prefix is not branding, it is the ABI inside every converted file, and the files in use are 20-67 GB each.
/// Renaming it would make them unreadable; changing it would need a new `layout_version` and a migration.
pub const ARCH_PREFIX: &str = "openmalleus.";

const GGML_F32: u32 = 0;
const GGML_F16: u32 = 1;
const GGML_I8: u32 = 24;
const GGML_I32: u32 = 26;
const GGML_I64: u32 = 27;
const GGML_BF16: u32 = 30;
pub const GGML_MLS2: u32 = 0x4D4C_0002;
pub const GGML_MLS3: u32 = 0x4D4C_0003;
pub const GGML_MLS4: u32 = 0x4D4C_0004;
pub const GGML_MLS8: u32 = 0x4D4C_0008;
pub const GGML_ROW_I4: u32 = 0x4D4C_0014;
/// Raw bytes and e4m3 bytes: what a release's packed experts are read as before dequantisation. A `.malleus` file
/// never carries them; the ids exist so the dtype round-trips.
pub const GGML_U8: u32 = 0x4D4C_0015;
pub const GGML_F8_E4M3: u32 = 0x4D4C_0016;
/// The scale and packed-expert dtypes of a DeepSeek-V4.1 checkpoint: read on the way in, never written to a
/// `.malleus` file.
pub const GGML_F8_E8M0: u32 = 0x4D4C_0017;
pub const GGML_F4_E2M1: u32 = 0x4D4C_0018;

fn dtype_to_ggml(d: DType) -> u32 {
    match d {
        DType::F32 => GGML_F32,
        DType::F16 => GGML_F16,
        DType::BF16 => GGML_BF16,
        DType::I8 => GGML_I8,
        DType::I32 => GGML_I32,
        DType::I64 => GGML_I64,
        DType::Mls2 => GGML_MLS2,
        DType::Mls3 => GGML_MLS3,
        DType::Mls4 => GGML_MLS4,
        DType::Mls8 => GGML_MLS8,
        DType::RowI4 => GGML_ROW_I4,
        DType::U8 => GGML_U8,
        DType::F8E4M3 => GGML_F8_E4M3,
        DType::F8E8M0 => GGML_F8_E8M0,
        DType::F4E2M1 => GGML_F4_E2M1,
    }
}

fn ggml_to_dtype(t: u32) -> Result<DType> {
    Ok(match t {
        GGML_F32 => DType::F32,
        GGML_F16 => DType::F16,
        GGML_BF16 => DType::BF16,
        GGML_I8 => DType::I8,
        GGML_I32 => DType::I32,
        GGML_I64 => DType::I64,
        GGML_MLS2 => DType::Mls2,
        GGML_MLS3 => DType::Mls3,
        GGML_MLS4 => DType::Mls4,
        GGML_MLS8 => DType::Mls8,
        GGML_ROW_I4 => DType::RowI4,
        GGML_U8 => DType::U8,
        GGML_F8_E4M3 => DType::F8E4M3,
        GGML_F8_E8M0 => DType::F8E8M0,
        GGML_F4_E2M1 => DType::F4E2M1,
        _ => bail!("unsupported ggml type {t}"),
    })
}

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Array(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U8(v) => Some(v as u64),
            Value::U16(v) => Some(v as u64),
            Value::U32(v) => Some(v as u64),
            Value::U64(v) => Some(v),
            Value::I32(v) if v >= 0 => Some(v as u64),
            Value::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::F32(v) => Some(v as f64),
            Value::F64(v) => Some(v),
            _ => self.as_u64().map(|v| v as f64),
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
}

/// Longest metadata string a file may carry. A tokenizer of a 200k vocabulary is ~26 MB (`tokenizer.json` of
/// Inkling-2), so the old 16 MiB cap refused a legitimate file; 256 MiB still keeps a corrupt length from asking
/// the allocator for a machine's worth of memory.
const MAX_STRING: usize = 1 << 28;

struct Rd<R: Read> {
    r: R,
}

impl<R: Read> Rd<R> {
    fn bytes<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut b = [0u8; N];
        self.r.read_exact(&mut b)?;
        Ok(b)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u64()? as usize;
        if n > MAX_STRING {
            bail!("string too long");
        }
        let mut b = vec![0u8; n];
        self.r.read_exact(&mut b)?;
        Ok(String::from_utf8(b)?)
    }
    fn value(&mut self, ty: u32, depth: u32) -> Result<Value> {
        Ok(match ty {
            0 => Value::U8(self.bytes::<1>()?[0]),
            1 => Value::I8(self.bytes::<1>()?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(self.bytes()?)),
            3 => Value::I16(i16::from_le_bytes(self.bytes()?)),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(i32::from_le_bytes(self.bytes()?)),
            6 => Value::F32(f32::from_le_bytes(self.bytes()?)),
            7 => Value::Bool(self.bytes::<1>()?[0] != 0),
            8 => Value::Str(self.string()?),
            9 => {
                if depth > 2 {
                    bail!("nested arrays too deep");
                }
                let ety = self.u32()?;
                let n = self.u64()? as usize;
                if n > 1 << 26 {
                    bail!("array too long");
                }
                let mut v = Vec::with_capacity(n.min(1 << 20));
                for _ in 0..n {
                    v.push(self.value(ety, depth + 1)?);
                }
                Value::Array(v)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(i64::from_le_bytes(self.bytes()?)),
            12 => Value::F64(f64::from_le_bytes(self.bytes()?)),
            _ => bail!("bad gguf value type {ty}"),
        })
    }
}

pub struct Gguf {
    file: File,
    pub metadata: BTreeMap<String, Value>,
    tensors: Vec<TensorInfo>,
    by_name: HashMap<String, usize>,
}

impl Gguf {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| path.display().to_string())?;
        let len = file.metadata()?.len() as usize;
        let mut rd = Rd {
            r: BufReader::new(file.try_clone()?),
        };
        if &rd.bytes::<4>()? != MAGIC {
            bail!("not a GGUF file");
        }
        let ver = rd.u32()?;
        if ver != VERSION {
            bail!("unsupported GGUF version {ver}");
        }
        let n_tensors = rd.u64()? as usize;
        let n_kv = rd.u64()? as usize;
        if n_tensors > 1 << 20 || n_kv > 1 << 20 {
            bail!("implausible header counts");
        }
        let mut metadata = BTreeMap::new();
        for _ in 0..n_kv {
            let k = rd.string()?;
            let ty = rd.u32()?;
            metadata.insert(k, rd.value(ty, 0)?);
        }
        let mut raw = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = rd.string()?;
            let nd = rd.u32()? as usize;
            if nd == 0 || nd > 8 {
                bail!("bad n_dims for {name}");
            }
            let mut dims = Vec::with_capacity(nd);
            for _ in 0..nd {
                dims.push(rd.u64()? as usize);
            }
            dims.reverse();
            let dtype = ggml_to_dtype(rd.u32()?)?;
            let off = rd.u64()? as usize;
            raw.push((name, dims, dtype, off));
        }
        let align = metadata
            .get("general.alignment")
            .and_then(Value::as_u64)
            .unwrap_or(DEFAULT_ALIGNMENT as u64) as usize;
        let pos = rd.r.stream_position()? as usize;
        let base = pos.div_ceil(align) * align;
        let mut tensors = Vec::with_capacity(n_tensors);
        let mut by_name = HashMap::new();
        for (name, shape, dtype, off) in raw {
            let size = dtype.size_of_elems(shape.iter().product());
            if base + off + size > len {
                bail!("tensor {name} out of file bounds");
            }
            by_name.insert(name.clone(), tensors.len());
            tensors.push(TensorInfo {
                name,
                dtype,
                shape,
                offset: base + off,
                size,
            });
        }
        Ok(Gguf {
            file,
            metadata,
            tensors,
            by_name,
        })
    }

    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    pub fn meta_u64(&self, k: &str) -> Result<u64> {
        self.metadata
            .get(k)
            .and_then(Value::as_u64)
            .with_context(|| format!("missing metadata {k}"))
    }

    pub fn meta_f64(&self, k: &str) -> Result<f64> {
        self.metadata
            .get(k)
            .and_then(Value::as_f64)
            .with_context(|| format!("missing metadata {k}"))
    }

    pub fn meta_str(&self, k: &str) -> Result<&str> {
        self.metadata
            .get(k)
            .and_then(Value::as_str)
            .with_context(|| format!("missing metadata {k}"))
    }

    /// Model family of an Malleus model file (`qwen35` for `openmalleus.qwen35`), after checking that the file
    /// was written by `malleus convert` with a weight layout this build reads.
    pub fn malleus_arch(&self) -> Result<&str> {
        let arch = self.meta_str("general.architecture")?;
        let layout = self.meta_u64("openmalleus.layout_version").with_context(|| {
            format!("not an Malleus model (architecture `{arch}`): convert it with `malleus convert`")
        })?;
        ensure!(
            layout == crate::quant::LAYOUT_VERSION as u64,
            "weight layout version {layout}, this build reads {}: convert the model again",
            crate::quant::LAYOUT_VERSION
        );
        // Files converted before the prefix was introduced carry the bare family name.
        Ok(arch.strip_prefix(ARCH_PREFIX).unwrap_or(arch))
    }
}

/// Rewrites a model file keeping only the tensors `keep` accepts, with the metadata unchanged.
///
/// The copy is byte-for-byte: the tensors that stay are the same bytes, so a rewrite cannot change what the model
/// computes, only what it carries. Used to drop the MLS8 prefill copies from a file converted before they became
/// opt-in without re-reading the bf16 checkpoint, which also means the MLS4 calibration is preserved exactly.
/// Returns `(kept, dropped)` tensor counts.
pub fn rewrite_keeping(
    src: &dyn TensorSource,
    metadata: &BTreeMap<String, Value>,
    out: &Path,
    keep: &dyn Fn(&str) -> bool,
) -> Result<(usize, usize)> {
    let names: Vec<&str> = src.names();
    let kept: Vec<&str> = names.iter().copied().filter(|n| keep(n)).collect();
    let dropped = names.len() - kept.len();
    if kept.is_empty() {
        bail!("every tensor is excluded: refusing to write an empty model");
    }
    let mut w = GgufWriter::new();
    for (k, v) in metadata {
        w.meta(k, v.clone());
    }
    for n in &kept {
        let info = src.info(n).with_context(|| format!("missing tensor {n}"))?;
        w.add_tensor(n, info.dtype, &info.shape);
    }
    let mut sink = w.begin(out)?;
    // 64 MiB: the n-gram table alone is tens of GiB, and the copy is bound by the disk, not by this buffer.
    let mut buf = vec![0u8; 64 << 20];
    for n in &kept {
        let size = src.info(n).context("info")?.size;
        let mut off = 0;
        while off < size {
            let len = buf.len().min(size - off);
            src.read(n, off, &mut buf[..len])?;
            sink.write(&buf[..len])?;
            off += len;
        }
    }
    sink.finish()?;
    Ok((kept.len(), dropped))
}

impl TensorSource for Gguf {
    fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.by_name.get(name).map(|&i| &self.tensors[i])
    }

    fn names(&self) -> Vec<&str> {
        self.tensors.iter().map(|t| t.name.as_str()).collect()
    }

    fn read(&self, name: &str, start: usize, buf: &mut [u8]) -> std::io::Result<()> {
        let info = self
            .info(name)
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, name.to_string()))?;
        if start + buf.len() > info.size {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "read past tensor end",
            ));
        }
        self.file.read_exact_at(buf, (info.offset + start) as u64)
    }

    fn open_tensor(&self, name: &str) -> Option<crate::TensorFile> {
        let info = self.info(name)?;
        Some(crate::TensorFile::new(
            self.file.try_clone().ok()?,
            info.offset as u64,
            info.size,
        ))
    }
}

/// Streaming GGUF writer: declare all tensors first, then write their data in the same order.
pub struct GgufWriter {
    metadata: Vec<(String, Value)>,
    tensors: Vec<(String, DType, Vec<usize>)>,
}

fn w_str(w: &mut impl Write, s: &str) -> Result<()> {
    w.write_all(&(s.len() as u64).to_le_bytes())?;
    w.write_all(s.as_bytes())?;
    Ok(())
}

fn value_type(v: &Value) -> u32 {
    match v {
        Value::U8(_) => 0,
        Value::I8(_) => 1,
        Value::U16(_) => 2,
        Value::I16(_) => 3,
        Value::U32(_) => 4,
        Value::I32(_) => 5,
        Value::F32(_) => 6,
        Value::Bool(_) => 7,
        Value::Str(_) => 8,
        Value::Array(_) => 9,
        Value::U64(_) => 10,
        Value::I64(_) => 11,
        Value::F64(_) => 12,
    }
}

fn w_value(w: &mut impl Write, v: &Value) -> Result<()> {
    match v {
        Value::U8(x) => w.write_all(&[*x])?,
        Value::I8(x) => w.write_all(&[*x as u8])?,
        Value::U16(x) => w.write_all(&x.to_le_bytes())?,
        Value::I16(x) => w.write_all(&x.to_le_bytes())?,
        Value::U32(x) => w.write_all(&x.to_le_bytes())?,
        Value::I32(x) => w.write_all(&x.to_le_bytes())?,
        Value::F32(x) => w.write_all(&x.to_le_bytes())?,
        Value::Bool(x) => w.write_all(&[*x as u8])?,
        Value::Str(s) => w_str(w, s)?,
        Value::Array(a) => {
            let ety = a.first().map(value_type).unwrap_or(4);
            w.write_all(&ety.to_le_bytes())?;
            w.write_all(&(a.len() as u64).to_le_bytes())?;
            for x in a {
                if value_type(x) != ety {
                    bail!("heterogeneous GGUF array");
                }
                w_value(w, x)?;
            }
        }
        Value::U64(x) => w.write_all(&x.to_le_bytes())?,
        Value::I64(x) => w.write_all(&x.to_le_bytes())?,
        Value::F64(x) => w.write_all(&x.to_le_bytes())?,
    }
    Ok(())
}

impl Default for GgufWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl GgufWriter {
    pub fn new() -> Self {
        GgufWriter {
            metadata: Vec::new(),
            tensors: Vec::new(),
        }
    }

    pub fn meta(&mut self, k: &str, v: Value) {
        self.metadata.push((k.to_string(), v));
    }

    pub fn add_tensor(&mut self, name: &str, dtype: DType, shape: &[usize]) {
        self.tensors.push((name.to_string(), dtype, shape.to_vec()));
    }

    /// Writes the header; returns a sink that must receive each tensor's bytes in declaration order.
    pub fn begin(self, path: &Path) -> Result<GgufDataSink> {
        let mut w = BufWriter::with_capacity(8 << 20, File::create(path)?);
        w.write_all(MAGIC)?;
        w.write_all(&VERSION.to_le_bytes())?;
        w.write_all(&(self.tensors.len() as u64).to_le_bytes())?;
        w.write_all(&(self.metadata.len() as u64).to_le_bytes())?;
        for (k, v) in &self.metadata {
            w_str(&mut w, k)?;
            w.write_all(&value_type(v).to_le_bytes())?;
            w_value(&mut w, v)?;
        }
        let mut off = 0usize;
        let mut sizes = Vec::new();
        for (name, dtype, shape) in &self.tensors {
            w_str(&mut w, name)?;
            w.write_all(&(shape.len() as u32).to_le_bytes())?;
            for d in shape.iter().rev() {
                w.write_all(&(*d as u64).to_le_bytes())?;
            }
            w.write_all(&dtype_to_ggml(*dtype).to_le_bytes())?;
            w.write_all(&(off as u64).to_le_bytes())?;
            let size = dtype.size_of_elems(shape.iter().product());
            sizes.push((name.clone(), size));
            off = (off + size).div_ceil(DEFAULT_ALIGNMENT) * DEFAULT_ALIGNMENT;
        }
        let pos = w.stream_position()? as usize;
        let pad = pos.div_ceil(DEFAULT_ALIGNMENT) * DEFAULT_ALIGNMENT - pos;
        w.write_all(&vec![0u8; pad])?;
        Ok(GgufDataSink {
            w,
            sizes,
            next: 0,
            written: 0,
        })
    }
}

pub struct GgufDataSink {
    w: BufWriter<File>,
    sizes: Vec<(String, usize)>,
    next: usize,
    written: usize,
}

impl GgufDataSink {
    /// Appends bytes to the current tensor; may be called several times per tensor.
    pub fn write(&mut self, data: &[u8]) -> Result<()> {
        let (name, size) = self.sizes.get(self.next).context("too many tensors written")?;
        if self.written + data.len() > *size {
            bail!("too much data for tensor {name}");
        }
        self.w.write_all(data)?;
        self.written += data.len();
        if self.written == *size {
            let pad = size.div_ceil(DEFAULT_ALIGNMENT) * DEFAULT_ALIGNMENT - size;
            self.w.write_all(&vec![0u8; pad])?;
            self.next += 1;
            self.written = 0;
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<()> {
        if self.next != self.sizes.len() {
            bail!("only {} of {} tensors written", self.next, self.sizes.len());
        }
        self.w.flush()?;
        self.w.get_ref().sync_all()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let dir = std::env::temp_dir().join(format!("malleus-gguf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.gguf");
        let mut w = GgufWriter::new();
        w.meta("general.architecture", Value::Str("qwen35".into()));
        w.meta("x.list", Value::Array(vec![Value::U32(1), Value::U32(2)]));
        w.add_tensor("a", DType::F32, &[2, 3]);
        w.add_tensor("b", DType::BF16, &[5]);
        let mut s = w.begin(&p).unwrap();
        let a: Vec<u8> = (0..6).flat_map(|i| (i as f32).to_le_bytes()).collect();
        s.write(&a[..10]).unwrap();
        s.write(&a[10..]).unwrap();
        s.write(&[7u8; 10]).unwrap();
        s.finish().unwrap();
        let g = Gguf::open(&p).unwrap();
        assert_eq!(g.meta_str("general.architecture").unwrap(), "qwen35");
        assert_eq!(g.info("a").unwrap().shape, vec![2, 3]);
        assert_eq!(g.read_all("a").unwrap(), a);
        assert_eq!(g.read_all("b").unwrap(), vec![7u8; 10]);
        assert!(g.malleus_arch().is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// A vocabulary larger than the old 16 MiB cap is still a file we must read: the tokenizer travels in the
    /// metadata, not in a tensor.
    #[test]
    fn a_tokenizer_sized_string_roundtrips() {
        let dir = std::env::temp_dir().join(format!("malleus-gguf-bigstr-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.gguf");
        let tok = "{\"model\":{\"vocab\":".to_string() + &"a".repeat(20 << 20) + "}}";
        let mut w = GgufWriter::new();
        w.meta("tokenizer.huggingface.json", Value::Str(tok.clone()));
        let s = w.begin(&p).unwrap();
        s.finish().unwrap();
        let g = Gguf::open(&p).unwrap();
        assert_eq!(g.meta_str("tokenizer.huggingface.json").unwrap(), tok);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn rewrite_keeping_leaves_the_kept_tensors_byte_for_byte() {
        let dir = std::env::temp_dir().join(format!("malleus-gguf-slim-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("full.malleus");
        let mut w = GgufWriter::new();
        w.meta("general.architecture", Value::Str("openmalleus.qwen4exp".into()));
        w.add_tensor("a", DType::Mls4, &[4, 128]);
        w.add_tensor("a.prefill", DType::Mls8, &[4, 128]);
        w.add_tensor("a.prefill.scales", DType::F32, &[4]);
        w.add_tensor("b", DType::F16, &[3]);
        let a: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let b: Vec<u8> = (0..6).map(|i| 200 + i as u8).collect();
        let mut s = w.begin(&p).unwrap();
        s.write(&a).unwrap();
        s.write(&[1u8; 512]).unwrap();
        s.write(&[2u8; 16]).unwrap();
        s.write(&b).unwrap();
        s.finish().unwrap();
        let g = Gguf::open(&p).unwrap();
        let out = dir.join("slim.malleus");
        let keep = |n: &str| !(n.ends_with(".prefill") || n.ends_with(".prefill.scales"));
        let (kept, dropped) = rewrite_keeping(&g, &g.metadata, &out, &keep).unwrap();
        assert_eq!((kept, dropped), (2, 2));
        let h = Gguf::open(&out).unwrap();
        assert_eq!(h.meta_str("general.architecture").unwrap(), "openmalleus.qwen4exp");
        assert_eq!(h.names(), vec!["a", "b"]);
        assert_eq!(h.read_all("a").unwrap(), a);
        assert_eq!(h.read_all("b").unwrap(), b);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn malleus_metadata() {
        let dir = std::env::temp_dir().join(format!("malleus-gguf-meta-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let open = |arch: &str, layout: u32| {
            let p = dir.join(format!("{arch}-{layout}.{EXTENSION}"));
            let mut w = GgufWriter::new();
            w.meta("general.architecture", Value::Str(arch.into()));
            w.meta("openmalleus.layout_version", Value::U32(layout));
            w.add_tensor("a", DType::F32, &[1]);
            let mut s = w.begin(&p).unwrap();
            s.write(&[0u8; 4]).unwrap();
            s.finish().unwrap();
            Gguf::open(&p).unwrap()
        };
        let v = crate::quant::LAYOUT_VERSION;
        assert_eq!(open("openmalleus.qwen35", v).malleus_arch().unwrap(), "qwen35");
        assert_eq!(open("qwen35", v).malleus_arch().unwrap(), "qwen35");
        assert!(open("openmalleus.qwen35", v + 1).malleus_arch().is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
