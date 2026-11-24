use anyhow::{bail, Result};
use burda_bundle::{Bundle, Profile};
use std::{env, path::PathBuf};

fn main() -> Result<()> {
    let mut args = env::args_os().skip(1);
    let directory = PathBuf::from(
        args.next()
            .ok_or_else(|| anyhow::anyhow!("usage: burda-bundle BUNDLE [--fixture]"))?,
    );
    let profile = match args.next() {
        None => Profile::Reference,
        Some(arg) if arg == "--fixture" => Profile::Fixture,
        _ => bail!("usage: burda-bundle BUNDLE [--fixture]"),
    };
    if args.next().is_some() {
        bail!("unexpected argument");
    }
    let bundle = Bundle::load(&directory, profile)?;
    let tensors = bundle.tensors()?;
    println!(
        "{} tensors, {} F32 elements, tokenizer {}",
        tensors.len(),
        tensors
            .tensors()
            .iter()
            .map(|(_, t)| t.data().len() / 4)
            .sum::<usize>(),
        bundle.config["tokenizer_id"]
    );
    Ok(())
}
