use anyhow::{bail, ensure, Context, Result};
use burda_bundle::{
    cpu::{generate, CpuDecoder},
    Bundle, Profile,
};
use std::{env, fs, path::PathBuf};

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let (mut bundle, mut prompt, mut trace) = (None, None, None);
    let mut max = 16;
    while let Some(flag) = args.next() {
        if flag == "--help" {
            println!("burda-generate --bundle DIR --text TEXT [--device cpu] [--placement cpu] [--max-new-tokens N] [--trace-dir NEW_DIR]");
            return Ok(());
        }
        let value = args
            .next()
            .with_context(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--bundle" => bundle = Some(PathBuf::from(value)),
            "--text" => prompt = Some(value),
            "--trace-dir" => trace = Some(PathBuf::from(value)),
            "--max-new-tokens" => max = value.parse().context("invalid max-new-tokens")?,
            "--device" | "--placement" => ensure!(
                value == "cpu",
                "CPU reference stage supports only cpu for {flag}"
            ),
            _ => bail!("unknown option {flag}"),
        }
    }
    let prompt = prompt.context("--text required")?;
    let bundle = Bundle::load(&bundle.context("--bundle required")?, Profile::Reference)?;
    let tokenizer = bundle.tokenizer.as_ref().context("missing tokenizer")?;
    let mut ids = vec![1];
    ids.extend(tokenizer.encode(&prompt)?);
    let decoder = CpuDecoder::new(&bundle)?;
    if let Some(dir) = &trace {
        fs::create_dir(dir).context("trace directory must be new")?;
    }
    let mut steps = Vec::new();
    let result = generate(ids, max, |prefix| {
        let logits = decoder.forward(prefix)?;
        ensure!(logits.iter().all(|n| n.is_finite()), "nonfinite logits");
        let row = logits.row(logits.nrows() - 1);
        let mut best = 0;
        for i in 1..row.len() {
            if row[i] > row[best] {
                best = i;
            }
        }
        if let Some(dir) = &trace {
            fs::write(
                dir.join(format!("{}.f32", steps.len())),
                bytemuck::cast_slice::<f32, u8>(logits.as_slice().context("logit layout")?),
            )?;
            steps.push(serde_json::json!({"id":best,"shape":logits.shape()}));
        }
        Ok(best as u32)
    })?;
    if let Some(dir) = trace {
        fs::write(dir.join("steps.json"), serde_json::to_vec(&steps)?)?;
    }
    println!("{}", tokenizer.decode(&result, true)?);
    Ok(())
}
