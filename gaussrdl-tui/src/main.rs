//! Terminal UI for GaussRDL: configure model parameters with the keyboard and
//! watch training metrics update live.
//!
//! Keys: ↑/↓ select · ←/→ adjust · Enter train · q quit

use gaussrdl_rdl::models::{ModelConfig, ModelKind};
use gaussrdl_rdl::synthetic::SyntheticConfig;
use gaussrdl_rdl::{
    run_experiment_cb, DataSource, EpochMetrics, ExperimentConfig, ExperimentResult, LrSchedule,
    TaskType,
};

mod app;
use app::App;

use std::io::{self, Stdout};
use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

use ratatui::crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph, Sparkline, Wrap};

enum Msg {
    Epoch(EpochMetrics),
    Done(Box<ExperimentResult>),
    Err(String),
}

#[derive(Default)]
struct Run {
    losses: Vec<u64>,
    last: Option<EpochMetrics>,
    result: Option<ExperimentResult>,
    err: Option<String>,
    running: bool,
}

fn build_config(app: &App) -> Result<ExperimentConfig, String> {
    let model = ModelKind::parse(&app.choice("model")).map_err(|e| e.to_string())?;
    let task = TaskType::parse(&app.choice("task")).ok_or("bad task")?;
    let hidden = app.num("hidden_dim") as usize;
    let heads = app.num("num_heads") as usize;
    if hidden % heads != 0 {
        return Err(format!("hidden_dim {hidden} not divisible by heads {heads}"));
    }
    Ok(ExperimentConfig {
        model,
        task,
        model_cfg: ModelConfig {
            hidden_dim: hidden,
            num_layers: app.num("num_layers") as usize,
            num_heads: heads,
            dropout: app.num("dropout") as f32,
            ..Default::default()
        },
        channels: hidden,
        epochs: app.num("epochs") as usize,
        lr: app.num("lr"),
        schedule: LrSchedule::Cosine,
        early_stopping_patience: app.num("patience") as usize,
        data: DataSource::Synthetic(SyntheticConfig {
            num_users: app.num("num_users") as usize,
            ..Default::default()
        }),
        ..Default::default()
    })
}

fn start_training(app: &App) -> Result<Receiver<Msg>, String> {
    let cfg = build_config(app)?;
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut cb = |em: &EpochMetrics| {
            let _ = tx.send(Msg::Epoch(em.clone()));
        };
        match run_experiment_cb(&cfg, &mut cb) {
            Ok(r) => {
                let _ = tx.send(Msg::Done(Box::new(r)));
            }
            Err(e) => {
                let _ = tx.send(Msg::Err(e.to_string()));
            }
        }
    });
    Ok(rx)
}

fn main() -> io::Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let res = run_loop(&mut terminal);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    res
}

fn run_loop(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> io::Result<()> {
    let mut app = App::new();
    let mut run = Run::default();
    let mut rx: Option<Receiver<Msg>> = None;

    loop {
        // Drain training messages.
        if let Some(r) = &rx {
            while let Ok(msg) = r.try_recv() {
                match msg {
                    Msg::Epoch(em) => {
                        run.losses.push((em.train_loss.max(0.0) * 1000.0) as u64);
                        run.last = Some(em);
                    }
                    Msg::Done(res) => {
                        run.result = Some(*res);
                        run.running = false;
                    }
                    Msg::Err(e) => {
                        run.err = Some(e);
                        run.running = false;
                    }
                }
            }
        }

        terminal.draw(|f| ui(f, &app, &run))?;

        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(k) = event::read()? {
                if k.kind != KeyEventKind::Press {
                    continue;
                }
                match k.code {
                    KeyCode::Char('q') | KeyCode::Esc => break,
                    KeyCode::Up => app.prev(),
                    KeyCode::Down => app.next(),
                    KeyCode::Left => app.dec(),
                    KeyCode::Right => app.inc(),
                    KeyCode::Enter if !run.running => {
                        run = Run::default();
                        match start_training(&app) {
                            Ok(r) => {
                                run.running = true;
                                rx = Some(r);
                            }
                            Err(e) => run.err = Some(e),
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    Ok(())
}

fn ui(f: &mut Frame, app: &App, run: &Run) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(0), Constraint::Length(3)])
        .split(f.area());

    let title = Paragraph::new("GaussRDL · Relational Deep Learning · Gaussian Technologies")
        .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(title, chunks[0]);

    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(40), Constraint::Min(0)])
        .split(chunks[1]);

    // Settings panel.
    let mut lines: Vec<Line> = Vec::new();
    for (i, s) in app.settings.iter().enumerate() {
        let style = if i == app.cursor {
            Style::default().fg(Color::Black).bg(Color::Cyan)
        } else {
            Style::default().fg(Color::White)
        };
        lines.push(Line::styled(format!("  {:<14} {}", s.name(), s.display()), style));
    }
    let settings = Paragraph::new(lines)
        .block(Block::default().title(" parameters ").borders(Borders::ALL));
    f.render_widget(settings, body[0]);

    // Monitor panel: sparkline + metrics.
    let mon = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(9), Constraint::Min(0)])
        .split(body[1]);

    let spark = Sparkline::default()
        .block(Block::default().title(" training loss ").borders(Borders::ALL))
        .data(&run.losses)
        .style(Style::default().fg(Color::LightBlue));
    f.render_widget(spark, mon[0]);

    let text = if let Some(err) = &run.err {
        format!("error: {err}")
    } else if let Some(r) = &run.result {
        match r.task {
            TaskType::BinaryClassification => format!(
                "DONE  {}\n\nval ROC-AUC : {:.4}\ntest ROC-AUC: {:.4}\naccuracy    : {:.4}\nBrier       : {:.4}\n\nthroughput  : {:.0} /s\nlatency     : {:.1} ms\nepochs run  : {}  (best {})",
                r.model, r.val.auroc, r.test.auroc, r.test.accuracy, r.inference.brier,
                r.inference.throughput_per_s, r.inference.latency_ms,
                r.history.epochs.len(), r.history.best_epoch
            ),
            TaskType::Regression => format!(
                "DONE  {}\n\nval MAE : {:.2}\ntest MAE: {:.2}\nRMSE    : {:.2}\nR²      : {:.3}\n\nthroughput: {:.0} /s\nlatency   : {:.1} ms\nepochs run: {}  (best {})",
                r.model, r.val.mae, r.test.mae, r.test.rmse, r.inference.r2,
                r.inference.throughput_per_s, r.inference.latency_ms,
                r.history.epochs.len(), r.history.best_epoch
            ),
        }
    } else if let Some(em) = &run.last {
        format!(
            "TRAINING…\n\nepoch     : {}\ntrain loss: {:.4}\nval metric: {:.4}\nval loss  : {:.4}\nlearn rate: {:.5}\ngrad norm : {:.3}",
            em.epoch, em.train_loss, em.val_metric, em.val_loss, em.learning_rate, em.grad_norm
        )
    } else {
        "Configure parameters on the left.\nPress Enter to train, q to quit.".to_string()
    };
    let metrics = Paragraph::new(text)
        .wrap(Wrap { trim: true })
        .block(Block::default().title(" metrics & KPIs ").borders(Borders::ALL));
    f.render_widget(metrics, mon[1]);

    let footer = Paragraph::new("↑/↓ select   ←/→ adjust   Enter train   q quit")
        .style(Style::default().fg(Color::DarkGray))
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(footer, chunks[2]);
}
