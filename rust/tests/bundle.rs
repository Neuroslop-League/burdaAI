use burda_bundle::{Bundle, Profile};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
};
use tempfile::{tempdir, TempDir};

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures/c07-v1")
}

fn copied() -> TempDir {
    let dir = tempdir().unwrap();
    for entry in fs::read_dir(fixture()).unwrap() {
        let e = entry.unwrap();
        fs::copy(e.path(), dir.path().join(e.file_name())).unwrap();
    }
    dir
}

fn mutate_json(dir: &Path, file: &str, f: impl FnOnce(&mut Value)) {
    let path = dir.join(file);
    let mut json = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    f(&mut json);
    fs::write(path, serde_json::to_vec(&json).unwrap()).unwrap();
}

fn rehash(dir: &Path, name: &str) {
    let data = fs::read(dir.join(name)).unwrap();
    mutate_json(dir, "manifest.json", |m| {
        m["files"][name] =
            serde_json::json!({"bytes":data.len(),"sha256":format!("{:x}",Sha256::digest(&data))});
    });
}

#[test]
fn fixture_reads_every_value() {
    let bundle = Bundle::load(&fixture(), Profile::Fixture).unwrap();
    let tensors = bundle.tensors().unwrap();
    let mut names = tensors.names();
    names.sort();
    assert_eq!(names.len(), 11);
    for (i, name) in names.iter().enumerate() {
        let view = tensors.tensor(name).unwrap();
        for (j, chunk) in view.data().chunks_exact(4).enumerate() {
            assert_eq!(
                f32::from_le_bytes(chunk.try_into().unwrap()),
                (i * 1000 + j) as f32
            );
        }
    }
    assert!(Bundle::load(&fixture(), Profile::Reference).is_err());
}

#[test]
fn rejects_unknown_schema_and_corrupt_payload() {
    let dir = copied();
    mutate_json(dir.path(), "manifest.json", |m| {
        m["schema_version"] = 2.into()
    });
    assert!(Bundle::load(dir.path(), Profile::Fixture).is_err());
    let dir = copied();
    let path = dir.path().join("model.safetensors");
    let mut data = fs::read(&path).unwrap();
    *data.last_mut().unwrap() ^= 1;
    fs::write(path, data).unwrap();
    assert!(Bundle::load(dir.path(), Profile::Fixture).is_err());
}

#[test]
fn rejects_rehashed_transposed_axes_and_wrong_tokenizer() {
    let dir = copied();
    let path = dir.path().join("model.safetensors");
    let mut data = fs::read(&path).unwrap();
    let end = 8 + u64::from_le_bytes(data[..8].try_into().unwrap()) as usize;
    let header = std::str::from_utf8(&data[8..end]).unwrap().to_owned();
    assert!(header.contains("[4,8]"));
    data[8..end].copy_from_slice(header.replacen("[4,8]", "[8,4]", 1).as_bytes());
    fs::write(path, data).unwrap();
    rehash(dir.path(), "model.safetensors");
    assert!(Bundle::load(dir.path(), Profile::Fixture).is_err());
    let dir = copied();
    mutate_json(dir.path(), "config.json", |c| {
        c["tokenizer_id"] = "other".into()
    });
    rehash(dir.path(), "config.json");
    assert!(Bundle::load(dir.path(), Profile::Fixture).is_err());
}
