//! Terminal UI for GaussRDL — Gaussian Technologies' Relational Deep Learning
//! research suite. Two tabs:
//!   * Datasets  — browse the benchmark catalog; download/prepare/delete.
//!   * Benchmark — configure a model, run it on a dataset/task, watch training
//!     live, and read paper-style results (mean ± std over seeds) + KPIs.
//!
//! Keys: Tab switch · ↑/↓ select · ←/→ adjust · Enter run · d download · x
//! delete · r refresh · q quit

use gaussrdl_bench::{find, run_benchmark, BenchmarkReport, DatasetManager};
use gaussrdl_rdl::models::{ModelConfig, ModelKind};
use gaussrdl_rdl::EpochMetrics;

mod app;
use app::{App, Tab};

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
    BenchDone(Box<BenchmarkReport>),
    DsProgress(String),
    DsDone,
    Err(String),
}

#[derive(Default)]
struct State {
    losses: Vec<u64>,
    last: Option<EpochMetrics>,
    report: Option<BenchmarkReport>,
    status: String,
    err: Option<String>,
    busy: bool,
}

fn start_download(id: String) -> Receiver<Msg> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mgr = DatasetManager::with_default_root();
        match find(&id) {
            Some(info) => {
                let r = mgr.ensure(&info, &mut |p| {
                    let _ = tx.send(Msg::DsProgress(p.message));
                });
                match r {
                    Ok(_) => {
                        let _ = tx.send(Msg::DsDone);
                    }
                    Err(e) => {
                        let _ = tx.send(Msg::Err(e.to_string()));
                    }
                }
            }
            None => {
                let _ = tx.send(Msg::Err("unknown dataset".into()));
            }
        }
    });
    rx
}

fn start_benchmark(app: &App) -> Result<Receiver<Msg>, String> {
    let info = find(&app.choice("dataset")).ok_or("unknown dataset")?;
    let task = app.choice("task");
    let model = ModelKind::parse(&app.choice("model")).map_err(|e| e.to_string())?;
    let hidden = app.num("hidden_dim") as usize;
    let heads = app.num("num_heads") as usize;
    if hidden % heads != 0 {
        return Err(format!("hidden_dim {hidden} not divisible by heads {heads}"));
    }
    let cfg = ModelConfig {
        hidden_dim: hidden,
        num_layers: app.num("num_layers") as usize,
        num_heads: heads,
        dropout: app.num("dropout") as f32,
        ..Default::default()
    };
    let epochs = app.num("epochs") as usize;
    let seeds: Vec<u64> = (0..app.num("seeds") as u64).collect();
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mgr = DatasetManager::with_default_root();
        if let Err(e) = mgr.ensure(&info, &mut |p| {
            let _ = tx.send(Msg::DsProgress(p.message));
        }) {
            let _ = tx.send(Msg::Err(e.to_string()));
            return;
        }
        let mut cb = |em: &EpochMetrics| {
            let _ = tx.send(Msg::Epoch(em.clone()));
        };
        match run_benchmark(&mgr, &info, &task, model, cfg, hidden, epochs, &seeds, &mut cb) {
            Ok(r) => {
                let _ = tx.send(Msg::BenchDone(Box::new(r)));
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
    let mgr = DatasetManager::with_default_root();
    let mut app = App::new();
    app.refresh_datasets(&mgr);
    let mut st = State::default();
    let mut rx: Option<Receiver<Msg>> = None;

    loop {
        if let Some(r) = &rx {
            while let Ok(msg) = r.try_recv() {
                match msg {
                    Msg::Epoch(em) => {
                        st.losses.push((em.train_loss.max(0.0) * 1000.0) as u64);
                        st.last = Some(em);
                    }
                    Msg::BenchDone(r) => {
                        st.report = Some(*r);
                        st.busy = false;
                        st.status = "benchmark complete".into();
                    }
                    Msg::DsProgress(m) => st.status = m,
                    Msg::DsDone => {
                        st.busy = false;
                        st.status = "dataset ready".into();
                        app.refresh_datasets(&mgr);
                    }
                    Msg::Err(e) => {
                        st.err = Some(e);
                        st.busy = false;
                    }
                }
            }
        }

        terminal.draw(|f| ui(f, &app, &st))?;

        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(k) = event::read()? {
                if k.kind != KeyEventKind::Press {
                    continue;
                }
                match k.code {
                    KeyCode::Char('q') | KeyCode::Esc => break,
                    KeyCode::Tab => app.toggle_tab(),
                    KeyCode::Up => app.prev(),
                    KeyCode::Down => app.next(),
                    KeyCode::Left => app.dec(),
                    KeyCode::Right => app.inc(),
                    KeyCode::Char('r') => app.refresh_datasets(&mgr),
                    KeyCode::Char('d') if app.tab == Tab::Datasets && !st.busy => {
                        if let Some(id) = app.selected_dataset_id() {
                            st.err = None;
                            st.busy = true;
                            st.status = format!("preparing {id}…");
                            rx = Some(start_download(id));
                        }
                    }
                    KeyCode::Char('x') if app.tab == Tab::Datasets && !st.busy => {
                        if let Some(id) = app.selected_dataset_id() {
                            let _ = mgr.delete(&id);
                            app.refresh_datasets(&mgr);
                            st.status = format!("deleted {id}");
                        }
                    }
                    KeyCode::Enter if app.tab == Tab::Benchmark && !st.busy => {
                        st = State { busy: true, status: "starting…".into(), ..Default::default() };
                        match start_benchmark(&app) {
                            Ok(r) => rx = Some(r),
                            Err(e) => {
                                st.err = Some(e);
                                st.busy = false;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    Ok(())
}

fn ui(f: &mut Frame, app: &App, st: &State) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(0), Constraint::Length(3)])
        .split(f.area());

    // Title + tab indicator.
    let tab_label = match app.tab {
        Tab::Datasets => "[ Datasets ]  Benchmark ",
        Tab::Benchmark => " Datasets  [ Benchmark ]",
    };
    let title = Paragraph::new(Line::from(vec![
        Span::styled("GaussRDL ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::styled("· Gaussian Technologies   ", Style::default().fg(Color::DarkGray)),
        Span::styled(tab_label, Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
    ]))
    .block(Block::default().borders(Borders::ALL));
    f.render_widget(title, chunks[0]);

    match app.tab {
        Tab::Datasets => render_datasets(f, chunks[1], app, st),
        Tab::Benchmark => render_benchmark(f, chunks[1], app, st),
    }

    let help = match app.tab {
        Tab::Datasets => "Tab switch · ↑/↓ select · d download/prepare · x delete · r refresh · q quit",
        Tab::Benchmark => "Tab switch · ↑/↓ select · ←/→ adjust · Enter run · q quit",
    };
    let footer = Paragraph::new(help)
        .style(Style::default().fg(Color::DarkGray))
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(footer, chunks[2]);
}

fn render_datasets(f: &mut Frame, area: Rect, app: &App, st: &State) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(area);

    let mut lines: Vec<Line> = Vec::new();
    for (i, row) in app.datasets.iter().enumerate() {
        let sel = i == app.ds_cursor;
        let kind = if row.official { "RelBench" } else { "prepared" };
        let st_color = if row.status == "ready" {
            Color::Green
        } else if row.status.starts_with("error") {
            Color::Red
        } else {
            Color::DarkGray
        };
        let base = if sel {
            Style::default().fg(Color::Black).bg(Color::Cyan)
        } else {
            Style::default().fg(Color::White)
        };
        lines.push(Line::from(vec![
            Span::styled(format!("{:<22}", row.display), base),
            Span::styled(format!(" {:<9}", kind), base.fg(if sel { Color::Black } else { Color::Magenta })),
            Span::styled(format!(" {}", row.status), base.fg(if sel { Color::Black } else { st_color })),
        ]));
    }
    let list = Paragraph::new(lines)
        .block(Block::default().title(" dataset catalog ").borders(Borders::ALL));
    f.render_widget(list, cols[0]);

    let detail = app
        .selected_dataset_id()
        .and_then(|id| find(&id))
        .map(|d| {
            let tasks = d
                .tasks
                .iter()
                .map(|t| format!("  • {} ({})", t.name, t.kind.metric_name()))
                .collect::<Vec<_>>()
                .join("\n");
            format!(
                "{}\n\ndomain: {}\nsize: {}\nsource: {}\n\ntasks:\n{}\n\n{}\n\ncite: {}",
                d.display_name,
                d.domain,
                d.approx_size,
                if d.official { "RelBench (network)" } else { "prepared (offline)" },
                tasks,
                d.description,
                d.citation,
            )
        })
        .unwrap_or_default();
    let info = Paragraph::new(detail)
        .wrap(Wrap { trim: true })
        .block(Block::default().title(format!(" details {}", status_suffix(st))).borders(Borders::ALL));
    f.render_widget(info, cols[1]);
}

fn status_suffix(st: &State) -> String {
    if let Some(e) = &st.err {
        format!("· error: {e}")
    } else if st.busy {
        format!("· {}", st.status)
    } else if !st.status.is_empty() {
        format!("· {}", st.status)
    } else {
        String::new()
    }
}

fn render_benchmark(f: &mut Frame, area: Rect, app: &App, st: &State) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(40), Constraint::Min(0)])
        .split(area);

    let mut lines: Vec<Line> = Vec::new();
    for (i, s) in app.settings.iter().enumerate() {
        let style = if i == app.cursor {
            Style::default().fg(Color::Black).bg(Color::Cyan)
        } else {
            Style::default().fg(Color::White)
        };
        lines.push(Line::styled(format!("  {:<12} {}", s.name(), s.display()), style));
    }
    let settings =
        Paragraph::new(lines).block(Block::default().title(" configuration ").borders(Borders::ALL));
    f.render_widget(settings, cols[0]);

    let mon = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(9), Constraint::Min(0)])
        .split(cols[1]);
    let spark = Sparkline::default()
        .block(Block::default().title(" training loss ").borders(Borders::ALL))
        .data(&st.losses)
        .style(Style::default().fg(Color::LightBlue));
    f.render_widget(spark, mon[0]);

    let text = if let Some(e) = &st.err {
        format!("error: {e}")
    } else if let Some(r) = &st.report {
        let seeds = r
            .seeds
            .iter()
            .map(|s| format!("  seed {}: val {:.4}  test {:.4}", s.seed, s.val, s.test))
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "RESULT  {} on {}/{}\n\n{:.4} ± {:.4}  {}  ({})\nval {}: {:.4}\n\ngraph: {} nodes / {} edges / {} relations\nthroughput: {:.0}/s   latency: {:.1} ms\n\nper-seed:\n{}",
            r.model, r.dataset, r.task,
            r.test_mean, r.test_std, r.metric_name,
            if r.higher_is_better { "higher better" } else { "lower better" },
            r.metric_name, r.val_mean,
            r.num_nodes, r.num_edges, r.num_relations,
            r.inference.throughput_per_s, r.inference.latency_ms,
            seeds,
        )
    } else if let Some(em) = &st.last {
        format!(
            "TRAINING (seed 0)…  {}\n\nepoch: {}\ntrain loss: {:.4}\nval metric: {:.4}\nlearning rate: {:.5}\ngrad norm: {:.3}",
            st.status, em.epoch, em.train_loss, em.val_metric, em.learning_rate, em.grad_norm
        )
    } else {
        format!(
            "Select dataset & task, tune the model, press Enter.\n\ndataset: {}\ntask: {}\nmodel: {}\n\n{}",
            app.choice("dataset"),
            app.choice("task"),
            app.choice("model"),
            st.status
        )
    };
    let metrics = Paragraph::new(text)
        .wrap(Wrap { trim: true })
        .block(Block::default().title(" results & KPIs ").borders(Borders::ALL));
    f.render_widget(metrics, mon[1]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use gaussrdl_bench::{BenchmarkReport, SeedScore};
    use gaussrdl_rdl::{InferenceReport, TrainingHistory};
    use ratatui::backend::TestBackend;

    fn buffer_text(t: &Terminal<TestBackend>) -> String {
        t.backend().buffer().content().iter().map(|c| c.symbol()).collect()
    }

    #[test]
    fn renders_datasets_tab() {
        let app = App::new();
        let st = State::default();
        let mut term = Terminal::new(TestBackend::new(120, 40)).unwrap();
        term.draw(|f| ui(f, &app, &st)).unwrap();
        let text = buffer_text(&term);
        assert!(text.contains("GaussRDL"));
        assert!(text.contains("dataset catalog"));
        assert!(text.contains("Gauss e-commerce"));
    }

    #[test]
    fn renders_benchmark_tab_with_report() {
        let mut app = App::new();
        app.toggle_tab();
        let mut st = State::default();
        st.report = Some(BenchmarkReport {
            dataset: "gauss-ecom-small".into(),
            task: "user-churn".into(),
            model: "RelGT".into(),
            metric_name: "ROC-AUC".into(),
            higher_is_better: true,
            test_mean: 0.73,
            test_std: 0.02,
            val_mean: 0.71,
            seeds: vec![SeedScore { seed: 0, val: 0.71, test: 0.73 }],
            num_nodes: 1707,
            num_edges: 4748,
            num_relations: 4,
            inference: InferenceReport { throughput_per_s: 400.0, ..Default::default() },
            history: TrainingHistory::default(),
        });
        let mut term = Terminal::new(TestBackend::new(120, 40)).unwrap();
        term.draw(|f| ui(f, &app, &st)).unwrap();
        let text = buffer_text(&term);
        assert!(text.contains("configuration"));
        assert!(text.contains("RESULT"));
        assert!(text.contains("ROC-AUC"));
    }

    #[test]
    fn renders_benchmark_training_progress() {
        let mut app = App::new();
        app.toggle_tab();
        let mut st = State::default();
        st.busy = true;
        st.last = Some(EpochMetrics { epoch: 5, train_loss: 0.4, val_metric: 0.66, learning_rate: 0.009, grad_norm: 1.2, ..Default::default() });
        st.losses = vec![900, 700, 500, 400];
        let mut term = Terminal::new(TestBackend::new(120, 40)).unwrap();
        term.draw(|f| ui(f, &app, &st)).unwrap();
        let text = buffer_text(&term);
        assert!(text.contains("TRAINING"));
        assert!(text.contains("training loss"));
    }
}
