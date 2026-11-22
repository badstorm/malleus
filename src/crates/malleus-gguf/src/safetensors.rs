use crate::{DType, TensorInfo, TensorSource};
use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::os::unix::fs::FileExt;
use std::path::Path;

pub struct SafeTensors {
    files: Vec<File>,
    index: HashMap<String, (usize, TensorInfo)>,
}

fn parse_dtype(s: &str) -> Result<DType> {
    Ok(match s {
        "F32" => DType::F32,
        "F16" => DType::F16,
        "BF16" => DType::BF16,
        "I8" => DType::I8,
        "U8" => DType::U8,
        "F8_E4M3" => DType::F8E4M3,
        "F8_E8M0" => DType::F8E8M0,
        "F4_E2M1" => DType::F4E2M1,
        "I32" => DType::I32,
        "I64" => DType::I64,
        _ => bail!("unsupported safetensors dtype {s}"),
    })
}

impl SafeTensors {
    /// Opens every `*.safetensors` file in `dir`.
    pub fn open_dir(dir: &Path) -> Result<Self> {
        let mut files: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            bail!("no .safetensors files in {}", dir.display());
        }
        let mut st = SafeTensors {
            files: Vec::new(),
            index: HashMap::new(),
        };
        for f in files {
            st.add_file(&f).with_context(|| f.display().to_string())?;
        }
        Ok(st)
    }

    fn add_file(&mut self, path: &Path) -> Result<()> {
        let mut file = File::open(path)?;
        let len = file.metadata()?.len() as usize;
        let mut nb = [0u8; 8];
        file.read_exact(&mut nb)?;
        let n = u64::from_le_bytes(nb) as usize;
        if n > 100 << 20 || 8 + n > len {
            bail!("bad header length");
        }
        let mut hb = vec![0u8; n];
        file.read_exact(&mut hb)?;
        let header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&hb)?;
        let base = 8 + n;
        let fid = self.files.len();
        for (name, v) in header {
            if name == "__metadata__" {
                continue;
            }
            let dtype = parse_dtype(v["dtype"].as_str().context("dtype")?)?;
            let shape: Vec<usize> = v["shape"]
                .as_array()
                .context("shape")?
                .iter()
                .map(|x| x.as_u64().map(|x| x as usize).context("shape dim"))
                .collect::<Result<_>>()?;
            let off = v["data_offsets"].as_array().context("offsets")?;
            let a = off[0].as_u64().context("off")? as usize;
            let b = off[1].as_u64().context("off")? as usize;
            if base + b > len || a > b || dtype.size_of_elems(shape.iter().product()) != b - a {
                bail!("bad offsets for {name}");
            }
            let info = TensorInfo {
                name: name.clone(),
                dtype,
                shape,
                offset: base + a,
                size: b - a,
            };
            self.index.insert(name, (fid, info));
        }
        self.files.push(file);
        Ok(())
    }
}

impl TensorSource for SafeTensors {
    fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.index.get(name).map(|(_, i)| i)
    }

    fn names(&self) -> Vec<&str> {
        self.index.keys().map(|s| s.as_str()).collect()
    }

    fn read(&self, name: &str, start: usize, buf: &mut [u8]) -> std::io::Result<()> {
        let (fid, info) = self
            .index
            .get(name)
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, name.to_string()))?;
        if start + buf.len() > info.size {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "read past tensor end",
            ));
        }
        self.files[*fid].read_exact_at(buf, (info.offset + start) as u64)
    }
}
