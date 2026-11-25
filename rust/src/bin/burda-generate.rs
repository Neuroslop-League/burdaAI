use anyhow::{bail, ensure, Context, Result};
use burda_bundle::{
    cpu::{generate, CpuDecoder},
    Bundle, Profile,
};
use std::{env, fs, path::PathBuf};

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let (mut bundle, mut prompt, mut trace) = (None, None, None);
    let mut kv = "on".to_string();
    let mut threads = 4usize;
    let mut benchmark = false;
    let mut max: usize = 16;
    let mut device = String::from("cpu");
    let mut placement = String::from("cpu");
    let mut gguf = None;
    let mut tokenizer_dir = None;
    let mut runner = PathBuf::from("burda-ggml");
    while let Some(flag) = args.next() {
        if flag == "--help" {
            println!("burda-generate (--bundle DIR | --gguf FILE --tokenizer DIR [--runner EXE]) --text TEXT [--device cpu|vulkan] [--placement cpu|last:N] [--max-new-tokens N] [--benchmark-256x64] [--kv on|off] [--threads N]");
            return Ok(());
        }
        if flag == "--benchmark-256x64" {
            benchmark = true;
            continue;
        }
        let value = args
            .next()
            .with_context(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--bundle" => bundle = Some(PathBuf::from(value)),
            "--text" => prompt = Some(value),
            "--trace-dir" => trace = Some(PathBuf::from(value)),
            "--max-new-tokens" => max = value.parse().context("invalid max-new-tokens")?,
            "--device" => device = value,
            "--placement" => placement = value,
            "--kv" => {
                ensure!(value == "on" || value == "off", "expected on/off");
                kv = value;
            }
            "--threads" => {
                threads = value.parse()?;
                ensure!((1..=64).contains(&threads), "invalid thread count");
            }
            "--gguf" => gguf = Some(PathBuf::from(value)),
            "--tokenizer" => tokenizer_dir = Some(PathBuf::from(value)),
            "--runner" => runner = PathBuf::from(value),
            _ => bail!("unknown option {flag}"),
        }
    }
    let prompt = prompt.context("--text required")?;
    ensure!(!benchmark || gguf.is_some(), "benchmark requires GGUF Q4_0");
    if let Some(path) = gguf {
        ensure!(
            bundle.is_none() && trace.is_none(),
            "GGUF mode takes --gguf and --tokenizer"
        );
        let layers = match (device.as_str(), placement.as_str()) {
            ("cpu", "cpu") => 0,
            ("vulkan", p) if p.starts_with("last:") => {
                let n = p[5..].parse::<usize>().context("invalid layer count")?;
                ensure!((1..=20).contains(&n), "expected 1..20 GPU layers");
                n
            }
            _ => bail!("use cpu/cpu or vulkan/last:N"),
        };
        let tokenizer = burda_bundle::tokenizer::TextTokenizer::load_c02(
            &tokenizer_dir.context("--tokenizer required for GGUF")?,
        )?;
        let mut ids = vec![1];
        ids.extend(tokenizer.encode(&prompt)?);
        ensure!(ids.len() <= 4096, "prompt exceeds context");
        ensure!(
            !benchmark || (ids.len() == 256 && max == 64),
            "benchmark requires 256 input IDs including BOS and 64 output steps"
        );
        let csv = ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
        let output = std::process::Command::new(runner)
            .env("BURDA_KV", &kv)
            .env("BURDA_THREADS", threads.to_string())
            .arg(path)
            .arg(layers.to_string())
            .arg(max.to_string())
            .arg(&tokenizer.tokenizer_id)
            .arg(csv)
            .arg(if benchmark { "benchmark" } else { "generate" })
            .stderr(std::process::Stdio::inherit())
            .output()
            .context("start burda-ggml")?;
        ensure!(output.status.success(), "GGUF backend failed");
        let generated = std::str::from_utf8(&output.stdout)?
            .lines()
            .map(str::parse::<u32>)
            .collect::<Result<Vec<_>, _>>()?;
        ensure!(
            !benchmark || generated.len() == 64,
            "benchmark returned fewer than 64 IDs"
        );
        println!("{}", tokenizer.decode(&generated, true)?);
        return Ok(());
    }
    ensure!(
        device == "cpu" && placement == "cpu" && tokenizer_dir.is_none(),
        "reference bundle requires CPU placement"
    );
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
