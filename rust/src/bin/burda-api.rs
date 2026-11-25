//! Local API sharing the GGUF worker and tokenizer with burda-generate.
use anyhow::{ensure, Context, Result};
use axum::{
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use burda_bundle::tokenizer::TextTokenizer;
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    env,
    io::{BufRead, BufReader, Write},
    path::Path,
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, SyncSender, TrySendError},
        Arc,
    },
};
use tokio::sync::oneshot;

type Response = (StatusCode, Json<Value>);
fn reply(status: StatusCode, value: Value) -> Response {
    (status, Json(value))
}

struct Job {
    prompt: Vec<u32>,
    limit: usize,
    ids: Vec<u32>,
    done: oneshot::Sender<Result<Vec<u32>, String>>,
}
#[derive(Clone)]
struct Service {
    tx: SyncSender<Job>,
    tok: Arc<TextTokenizer>,
    alive: Arc<AtomicBool>,
}

struct Worker {
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
}
impl Drop for Worker {
    fn drop(&mut self) {
        let _ = writeln!(self.input, "quit");
        let _ = self.child.wait();
    }
}
impl Worker {
    fn start(runner: &str, model: &str, id: &str) -> Result<Self> {
        let mut child = Command::new(runner)
            .args([model, "0", "1", id, "1", "serve"])
            .env("BURDA_THREADS", "4")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?;
        let input = child.stdin.take().unwrap();
        let output = BufReader::new(child.stdout.take().unwrap());
        let mut this = Self {
            child,
            input,
            output,
        };
        ensure!(this.read()? == "READY", "backend failed readiness");
        Ok(this)
    }
    fn read(&mut self) -> Result<String> {
        let mut line = String::new();
        ensure!(self.output.read_line(&mut line)? > 0, "backend exited");
        Ok(line.trim_end().to_owned())
    }
    fn call(&mut self, line: &str) -> Result<String> {
        writeln!(self.input, "{line}")?;
        self.input.flush()?;
        self.read()
    }
    fn schedule(&mut self, rx: Receiver<Job>) -> Result<()> {
        let mut active: [Option<Job>; 2] = [None, None];
        loop {
            for (slot, entry) in active.iter_mut().enumerate() {
                if entry.as_ref().is_some_and(|j| j.done.is_closed()) {
                    ensure!(
                        self.call(&format!("cancel {slot}"))? == "OK",
                        "cancel failed"
                    );
                    *entry = None;
                }
            }
            let idle = active.iter().all(Option::is_none);
            for (s, entry) in active.iter_mut().enumerate() {
                if entry.is_some() {
                    continue;
                }
                let next = if idle && s == 0 {
                    rx.recv().ok()
                } else {
                    rx.try_recv().ok()
                };
                let Some(job) = next else {
                    if idle && s == 0 {
                        return Ok(());
                    }
                    break;
                };
                if job.done.is_closed() {
                    continue;
                }
                let ids = job
                    .prompt
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(" ");
                if self.call(&format!("start {s} {} {ids}", job.limit))? == "OK" {
                    *entry = Some(job);
                } else {
                    let _ = job.done.send(Err("backend rejected request".into()));
                }
            }
            if active.iter().all(Option::is_none) {
                continue;
            }
            let response = self.call("tick")?;
            let mut events = response.split_whitespace();
            ensure!(events.next() == Some("OK"), "backend tick failed");
            for event in events {
                let mut fields = event.split(':');
                let s: usize = fields.next().context("missing slot")?.parse()?;
                let entry = active.get_mut(s).context("unknown slot")?;
                let token = fields.next().context("missing output")?;
                if token == "error" {
                    let _ = entry
                        .take()
                        .context("inactive slot")?
                        .done
                        .send(Err("decode failed".into()));
                    continue;
                }
                let id: u32 = token.parse()?;
                if id != 2 {
                    entry.as_mut().context("inactive slot")?.ids.push(id);
                }
                if fields.next() == Some("1") {
                    let job = entry.take().context("inactive slot")?;
                    let _ = job.done.send(Ok(job.ids));
                }
            }
        }
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    text: String,
    max_new_tokens: usize,
}

async fn ready(State(s): State<Service>) -> Response {
    let alive = s.alive.load(Ordering::Acquire);
    reply(
        if alive {
            StatusCode::OK
        } else {
            StatusCode::SERVICE_UNAVAILABLE
        },
        json!({"ready":alive}),
    )
}
async fn generate(
    State(s): State<Service>,
    body: Result<Json<Request>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Ok(Json(req)) = body else {
        return reply(
            StatusCode::BAD_REQUEST,
            json!({"error":"invalid JSON request"}),
        );
    };
    let mut prompt = vec![1];
    let Ok(tokens) = s.tok.encode(&req.text) else {
        return reply(StatusCode::BAD_REQUEST, json!({"error":"tokenization"}));
    };
    prompt.extend(tokens);
    if prompt.len().saturating_add(req.max_new_tokens) > 4096 {
        return reply(
            StatusCode::UNPROCESSABLE_ENTITY,
            json!({"error":"context limit 4096"}),
        );
    }
    if req.max_new_tokens == 0 {
        return reply(StatusCode::OK, json!({"ids":[],"text":""}));
    }
    let (tx, rx) = oneshot::channel();
    match s.tx.try_send(Job {
        prompt,
        limit: req.max_new_tokens,
        ids: vec![],
        done: tx,
    }) {
        Err(TrySendError::Full(_)) => {
            return reply(StatusCode::TOO_MANY_REQUESTS, json!({"error":"queue full"}))
        }
        Err(TrySendError::Disconnected(_)) => {
            return reply(
                StatusCode::SERVICE_UNAVAILABLE,
                json!({"error":"worker offline"}),
            )
        }
        Ok(()) => {}
    }
    match rx.await {
        Ok(Ok(ids)) => match s.tok.decode(&ids, true) {
            Ok(text) => reply(StatusCode::OK, json!({"text":text,"ids":ids})),
            Err(_) => reply(
                StatusCode::INTERNAL_SERVER_ERROR,
                json!({"error":"decode text"}),
            ),
        },
        _ => reply(
            StatusCode::SERVICE_UNAVAILABLE,
            json!({"error":"generation failed"}),
        ),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<_> = env::args().skip(1).collect();
    ensure!(
        args.len() == 4,
        "usage: burda-api RUNNER Q4_GGUF TOKENIZER_DIR PORT"
    );
    let port: u16 = args[3].parse()?;
    let tok = Arc::new(TextTokenizer::load_c02(Path::new(&args[2]))?);
    let mut worker = Worker::start(&args[0], &args[1], &tok.tokenizer_id)?;
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await?;
    let (tx, rx) = mpsc::sync_channel(2);
    let alive = Arc::new(AtomicBool::new(true));
    let flag = alive.clone();
    let thread = std::thread::spawn(move || {
        let result = worker.schedule(rx);
        flag.store(false, Ordering::Release);
        result
    });
    let app = Router::new()
        .route("/ready", get(ready))
        .route("/generate", post(generate))
        .layer(DefaultBodyLimit::max(256 * 1024))
        .with_state(Service { tx, tok, alive });
    eprintln!("ready http://{}", listener.local_addr()?);
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    thread
        .join()
        .map_err(|_| anyhow::anyhow!("worker panicked"))??;
    Ok(())
}
