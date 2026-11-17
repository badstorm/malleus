//! The models one server can serve: their API ids, their files and the stop tokens their tokenizer knows.
//!
//! The catalog is built once at startup from the files under the models directory. Requests only carry an id that
//! was found here: a path in a request body is never opened.

use anyhow::{Context, Result, bail, ensure};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

/// One `.malleus` file (or safetensors directory) the server can load.
#[derive(Debug)]
pub struct ModelSpec {
    /// The id the API answers to.
    pub id: Arc<str>,
    /// The file the GPU thread loads.
    pub path: PathBuf,
    /// Maximum context this model is served with.
    pub ctx: usize,
    /// End-of-sequence ids of the model's tokenizer.
    pub eos: Arc<[u32]>,
}

/// The models discovered at startup, with the optional default first.
#[derive(Debug)]
pub struct Catalog {
    entries: Vec<Arc<ModelSpec>>,
    by_id: HashMap<Arc<str>, Arc<ModelSpec>>,
    default: Option<Arc<ModelSpec>>,
}

impl Catalog {
    /// Builds the id map and orders `entries` with the default first. `default` names one id in the catalog.
    pub fn new(entries: Vec<ModelSpec>, default: Option<&str>) -> Result<Self> {
        ensure!(!entries.is_empty(), "no models to serve");
        let mut by_id: HashMap<Arc<str>, Arc<ModelSpec>> = HashMap::new();
        let mut arcs: Vec<Arc<ModelSpec>> = Vec::with_capacity(entries.len());
        for spec in entries {
            ensure!(!spec.id.is_empty(), "a model with an empty id");
            let spec = Arc::new(spec);
            if let Some(old) = by_id.insert(spec.id.clone(), spec.clone()) {
                bail!(
                    "model id `{}` is used by both {} and {}",
                    spec.id,
                    old.path.display(),
                    spec.path.display()
                );
            }
            arcs.push(spec);
        }
        let default = match default {
            Some(id) => Some(
                by_id
                    .get(id)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("default model `{id}` is not in the catalog"))?,
            ),
            None => None,
        };
        arcs.sort_by(|a, b| a.id.cmp(&b.id));
        if let Some(d) = &default {
            arcs.sort_by_key(|spec| spec.id != d.id);
        }
        Ok(Catalog {
            entries: arcs,
            by_id,
            default,
        })
    }

    pub fn entries(&self) -> &[Arc<ModelSpec>] {
        &self.entries
    }

    pub fn get(&self, id: &str) -> Option<Arc<ModelSpec>> {
        self.by_id.get(id).cloned()
    }

    pub fn default(&self) -> Option<Arc<ModelSpec>> {
        self.default.clone()
    }
}

/// Paths the server may serve: every `*.malleus` file directly under `dir` (symlinks that resolve outside it are
/// skipped), canonicalized and sorted.
pub fn files_in(dir: &std::path::Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("malleus") || !entry.file_type()?.is_file() {
            continue;
        }
        let canonical = path.canonicalize()?;
        if !canonical.starts_with(dir) {
            eprintln!(
                "warning: skipping {}: it resolves outside {}",
                path.display(),
                dir.display()
            );
            continue;
        }
        paths.push(canonical);
    }
    paths.sort();
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(id: &str) -> ModelSpec {
        ModelSpec {
            id: Arc::from(id),
            path: PathBuf::from(format!("/models/{id}.malleus")),
            ctx: 8192,
            eos: Arc::from([]),
        }
    }

    #[test]
    fn default_is_first() {
        let c = Catalog::new(vec![spec("b"), spec("a")], Some("a")).unwrap();
        assert_eq!(c.entries()[0].id.as_ref(), "a", "the default comes first");
        assert_eq!(c.entries()[1].id.as_ref(), "b");
        assert_eq!(c.get("a").unwrap().id.as_ref(), "a");
        assert!(c.get("missing").is_none());
        assert_eq!(c.default().unwrap().id.as_ref(), "a");
    }

    #[test]
    fn without_a_default_the_order_is_the_id_order() {
        let c = Catalog::new(vec![spec("b"), spec("a")], None).unwrap();
        assert!(c.default().is_none());
        let ids: Vec<&str> = c.entries().iter().map(|s| s.id.as_ref()).collect();
        assert_eq!(ids, ["a", "b"]);
    }

    #[test]
    fn duplicate_ids_are_rejected() {
        let e = Catalog::new(vec![spec("a"), spec("a")], None).unwrap_err();
        assert!(e.to_string().contains("model id `a`"));
        assert!(Catalog::new(vec![spec("a")], Some("missing")).is_err());
    }

    #[test]
    fn an_empty_catalog_is_rejected() {
        assert!(Catalog::new(vec![], None).is_err());
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "malleus-catalog-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn discovery_keeps_malleus_files_and_rejects_symlinks_that_escape() {
        let base = temp_dir("discover");
        let models = base.join("models");
        let outside = base.join("outside");
        std::fs::create_dir_all(&models).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(models.join("b.malleus"), b"").unwrap();
        std::fs::write(models.join("a.malleus"), b"").unwrap();
        std::fs::write(models.join("notes.txt"), b"").unwrap();
        std::fs::write(outside.join("hidden.malleus"), b"").unwrap();
        std::os::unix::fs::symlink(outside.join("hidden.malleus"), models.join("link.malleus")).unwrap();
        let got = files_in(&models).unwrap();
        let names: Vec<String> = got
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, ["a.malleus", "b.malleus"]);
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn discovery_is_empty_for_an_empty_directory() {
        let base = temp_dir("empty");
        let models = base.join("models");
        std::fs::create_dir_all(&models).unwrap();
        let file = models.join("a.malleus");
        std::fs::write(&file, b"").unwrap();
        assert_eq!(files_in(&models).unwrap().len(), 1);
        std::fs::remove_file(&file).unwrap();
        assert!(files_in(&models).unwrap().is_empty());
        std::fs::remove_dir_all(&base).unwrap();
    }
}
