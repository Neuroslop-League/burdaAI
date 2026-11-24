use burda_bundle::tokenizer::TextTokenizer;
use serde::Deserialize;
use std::{fs, path::Path};

#[derive(Deserialize)]
struct Cases {
    tokenizer_id: String,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    text: String,
    ids: Vec<u32>,
}

#[test]
fn matches_python_golden() {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("../artifacts/c02-v1");
    let golden: Cases =
        serde_json::from_slice(&fs::read(directory.join("golden.json")).unwrap()).unwrap();
    assert_eq!(golden.cases.len(), 14);
    for _ in 0..2 {
        let tok = TextTokenizer::load_c02(&directory).unwrap();
        assert_eq!(tok.tokenizer_id, golden.tokenizer_id);
        for case in &golden.cases {
            let ids = tok.encode(&case.text).unwrap();
            assert_eq!(ids, case.ids, "{:?}", case.text);
            assert!(ids.iter().all(|&i| (3..32768).contains(&i)));
            assert_eq!(tok.decode(&ids, false).unwrap(), case.text);
            let mut framed = vec![1];
            framed.extend(ids);
            framed.extend([2, 0]);
            assert_eq!(tok.decode(&framed, true).unwrap(), case.text);
            assert_eq!(
                tok.decode(&framed, false).unwrap(),
                format!("<|bos|>{}<|eos|><|pad|>", case.text)
            );
        }
    }
}
