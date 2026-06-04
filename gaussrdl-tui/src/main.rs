//! Terminal UI for GaussRDL — Gaussian Technologies' Relational Deep Learning
//! research suite. Two tabs:
//!   * Datasets  — browse the benchmark catalog; download/prepare/delete.
//!   * Benchmark — configure a model, run it on a dataset/task, watch training
//!     live (loss + validation charts), and read paper-style results
//!     (mean ± std over seeds) + inference KPIs.
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
use ratatui::symbols;
use ratatui::widgets::{
    Axis, Block, BorderType, Borders, Chart, Dataset, GraphType, Paragraph, Row, Table, Tabs, Wrap,
};

const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const C_ACC: Color = Color::Rgb(108, 140, 255);
const C_OK: Color = Color::Rgb(54, 211, 155);
const C_MUT: Color = Color::Rgb(132, 147, 184);
const C_VIO: Color = Color::Rgb(160, 123, 255);

enum Msg {
    Epoch(EpochMetrics),
    BenchDone(Box<BenchmarkReport>),
    DsProgress(String),
    DsDone,
    Err(String),
}

#[derive(Default)]
struct State {
    loss_pts: Vec<(f64, f64)>,
    val_pts: Vec<(f64, f64)>,
    last: Option<EpochMetrics>,
    report: Option<BenchmarkReport>,
    status: String,
    err: Option<String>,
    busy: bool,
    tick: usize,
}

impl State {
    fn spinner(&self) -> &'static str {
        SPINNER[self.tick % SPINNER.len()]
    }
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
        st.tick = st.tick.wrapping_add(1);
        if let Some(r) = &rx {
            while let Ok(msg) = r.try_recv() {
                match msg {
                    Msg::Epoch(em) => {
                        st.loss_pts.push((em.epoch as f64, em.train_loss as f64));
                        st.val_pts.push((em.epoch as f64, em.val_metric as f64));
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

        if event::poll(Duration::from_millis(90))? {
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
                        st = State { busy: true, status: "starting…".into(), tick: st.tick, ..Default::default() };
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

    // Header: brand + tab bar.
    let head = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(0), Constraint::Length(34)])
        .split(chunks[0]);
    let brand = Paragraph::new(Line::from(vec![
        Span::styled("◆ GaussRDL ", Style::default().fg(C_ACC).add_modifier(Modifier::BOLD)),
        Span::styled("· Gaussian Technologies", Style::default().fg(C_MUT)),
    ]))
    .block(Block::default().borders(Borders::ALL).border_type(BorderType::Rounded).border_style(Style::default().fg(Color::Rgb(34, 48, 79))));
    f.render_widget(brand, head[0]);
    let tabs = Tabs::new(vec![" Datasets ", " Benchmark "])
        .select(if app.tab == Tab::Datasets { 0 } else { 1 })
        .style(Style::default().fg(C_MUT))
        .highlight_style(Style::default().fg(Color::Black).bg(C_VIO).add_modifier(Modifier::BOLD))
        .divider(symbols::DOT)
        .block(Block::default().borders(Borders::ALL).border_type(BorderType::Rounded).border_style(Style::default().fg(Color::Rgb(34, 48, 79))));
    f.render_widget(tabs, head[1]);

    match app.tab {
        Tab::Datasets => render_datasets(f, chunks[1], app, st),
        Tab::Benchmark => render_benchmark(f, chunks[1], app, st),
    }

    let help = match app.tab {
        Tab::Datasets => "Tab switch  ·  ↑/↓ select  ·  d prepare/download  ·  x delete  ·  r refresh  ·  q quit",
        Tab::Benchmark => "Tab switch  ·  ↑/↓ select  ·  ←/→ adjust  ·  Enter run  ·  q quit",
    };
    let footer = Paragraph::new(help)
        .style(Style::default().fg(C_MUT))
        .block(Block::default().borders(Borders::ALL).border_type(BorderType::Rounded).border_style(Style::default().fg(Color::Rgb(34, 48, 79))));
    f.render_widget(footer, chunks[2]);
}

fn rblock(title: String) -> Block<'static> {
    Block::default()
        .title(Span::styled(title, Style::default().fg(C_ACC).add_modifier(Modifier::BOLD)))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(Color::Rgb(34, 48, 79)))
}

fn render_datasets(f: &mut Frame, area: Rect, app: &App, st: &State) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(58), Constraint::Percentage(42)])
        .split(area);

    let header = Row::new(vec!["Dataset", "Type", "Status"])
        .style(Style::default().fg(C_MUT).add_modifier(Modifier::BOLD));
    let rows: Vec<Row> = app
        .datasets
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let kind = if r.official { "RelBench" } else { "prepared" };
            let st_color = if r.status == "ready" {
                C_OK
            } else if r.status.starts_with("error") {
                Color::Rgb(255, 107, 129)
            } else {
                C_MUT
            };
            let row = Row::new(vec![
                Span::raw(r.display.clone()),
                Span::styled(kind, Style::default().fg(if r.official { C_VIO } else { C_ACC })),
                Span::styled(r.status.clone(), Style::default().fg(st_color)),
            ]);
            if i == app.ds_cursor {
                row.style(Style::default().fg(Color::Black).bg(C_ACC).add_modifier(Modifier::BOLD))
            } else {
                row
            }
        })
        .collect();
    let busy_tag = if st.busy { format!(" {} {} ", st.spinner(), st.status) } else { " Dataset catalog ".to_string() };
    let table = Table::new(
        rows,
        [Constraint::Percentage(55), Constraint::Percentage(20), Constraint::Percentage(25)],
    )
    .header(header)
    .block(rblock(busy_tag));
    f.render_widget(table, cols[0]);

    let detail = app
        .selected_dataset_id()
        .and_then(|id| find(&id))
        .map(|d| {
            let tasks = d
                .tasks
                .iter()
                .map(|t| format!("  • {} — {} ({})", t.name, t.description, t.kind.metric_name()))
                .collect::<Vec<_>>()
                .join("\n");
            (d.display_name.clone(), format!(
                "domain: {}\nsize: {}\nsource: {}\n\ntasks:\n{}\n\n{}\n\ncite: {}",
                d.domain,
                d.approx_size,
                if d.official { "RelBench (network)" } else { "prepared (offline)" },
                tasks,
                d.description,
                d.citation,
            ))
        })
        .unwrap_or_default();
    let info = Paragraph::new(detail.1)
        .style(Style::default().fg(Color::Rgb(195, 205, 234)))
        .wrap(Wrap { trim: true })
        .block(rblock(format!(" {} ", if detail.0.is_empty() { "details".into() } else { detail.0 })));
    f.render_widget(info, cols[1]);
}

fn line_chart<'a>(title: String, pts: &'a [(f64, f64)], color: Color, fmt: fn(f64) -> String) -> Chart<'a> {
    let (mut y0, mut y1) = (f64::INFINITY, f64::NEG_INFINITY);
    for &(_, v) in pts {
        y0 = y0.min(v);
        y1 = y1.max(v);
    }
    if !y0.is_finite() {
        y0 = 0.0;
        y1 = 1.0;
    }
    if (y1 - y0).abs() < 1e-9 {
        y0 -= 0.5;
        y1 += 0.5;
    }
    let xmax = pts.last().map(|&(x, _)| x).unwrap_or(1.0).max(1.0);
    let ds = Dataset::default()
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(color))
        .data(pts);
    Chart::new(vec![ds])
        .block(rblock(title))
        .x_axis(
            Axis::default()
                .style(Style::default().fg(C_MUT))
                .bounds([0.0, xmax])
                .labels(vec![Span::raw("0"), Span::raw(format!("e{}", xmax as usize))]),
        )
        .y_axis(
            Axis::default()
                .style(Style::default().fg(C_MUT))
                .bounds([y0, y1])
                .labels(vec![Span::raw(fmt(y0)), Span::raw(fmt(y1))]),
        )
}

fn render_benchmark(f: &mut Frame, area: Rect, app: &App, st: &State) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(38), Constraint::Min(0)])
        .split(area);

    // Configuration list.
    let mut lines: Vec<Line> = Vec::new();
    for (i, s) in app.settings.iter().enumerate() {
        let style = if i == app.cursor {
            Style::default().fg(Color::Black).bg(C_ACC).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::White)
        };
        lines.push(Line::styled(format!(" {:<11} {}", s.name(), s.display()), style));
    }
    let cfg = Paragraph::new(lines).block(rblock(" Configuration ".into()));
    f.render_widget(cfg, cols[0]);

    // Monitor: loss chart, validation chart, results.
    let mon = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(9), Constraint::Length(8), Constraint::Min(0)])
        .split(cols[1]);

    f.render_widget(line_chart(" Training loss ".into(), &st.loss_pts, C_ACC, |v| format!("{v:.2}")), mon[0]);
    f.render_widget(line_chart(" Validation metric ".into(), &st.val_pts, C_OK, |v| format!("{v:.3}")), mon[1]);

    let text = if let Some(e) = &st.err {
        Text::from(vec![Line::styled(format!("error: {e}"), Style::default().fg(Color::Rgb(255, 107, 129)))])
    } else if let Some(r) = &st.report {
        let seeds = r
            .seeds
            .iter()
            .map(|s| format!("  seed {}: val {:.4}  test {:.4}", s.seed, s.val, s.test))
            .collect::<Vec<_>>()
            .join("\n");
        Text::from(vec![
            Line::styled(
                format!("RESULT  {} on {}/{}", r.model, r.dataset, r.task),
                Style::default().fg(C_OK).add_modifier(Modifier::BOLD),
            ),
            Line::from(""),
            Line::styled(
                format!("{:.4} ± {:.4}  {}  ({})", r.test_mean, r.test_std, r.metric_name,
                    if r.higher_is_better { "↑ better" } else { "↓ better" }),
                Style::default().fg(C_ACC).add_modifier(Modifier::BOLD),
            ),
            Line::raw(format!("val {}: {:.4}", r.metric_name, r.val_mean)),
            Line::from(""),
            Line::raw(format!("graph: {} nodes · {} edges · {} relations", r.num_nodes, r.num_edges, r.num_relations)),
            Line::raw(format!("throughput: {:.0}/s   latency: {:.1} ms", r.inference.throughput_per_s, r.inference.latency_ms)),
            Line::from(""),
            Line::styled("per-seed:", Style::default().fg(C_MUT)),
            Line::raw(seeds),
        ])
    } else if let Some(em) = &st.last {
        Text::from(vec![
            Line::styled(format!("{} TRAINING (seed 0) · {}", st.spinner(), st.status), Style::default().fg(C_VIO).add_modifier(Modifier::BOLD)),
            Line::from(""),
            Line::raw(format!("epoch       {}", em.epoch)),
            Line::raw(format!("train loss  {:.4}", em.train_loss)),
            Line::raw(format!("val metric  {:.4}", em.val_metric)),
            Line::raw(format!("learn rate  {:.5}", em.learning_rate)),
            Line::raw(format!("grad norm   {:.3}", em.grad_norm)),
        ])
    } else {
        Text::from(vec![
            Line::raw("Select dataset & task, tune the model, press Enter."),
            Line::from(""),
            Line::raw(format!("dataset:  {}", app.choice("dataset"))),
            Line::raw(format!("task:     {}", app.choice("task"))),
            Line::raw(format!("model:    {}", app.choice("model"))),
            Line::styled(format!("\n{}", st.status), Style::default().fg(C_MUT)),
        ])
    };
    let results = Paragraph::new(text).wrap(Wrap { trim: true }).block(rblock(" Results & KPIs ".into()));
    f.render_widget(results, mon[2]);
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
        let mut term = Terminal::new(TestBackend::new(140, 40)).unwrap();
        term.draw(|f| ui(f, &app, &st)).unwrap();
        let text = buffer_text(&term);
        assert!(text.contains("GaussRDL"));
        assert!(text.contains("Datasets"));
        assert!(text.contains("Dataset catalog"));
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
        let mut term = Terminal::new(TestBackend::new(140, 40)).unwrap();
        term.draw(|f| ui(f, &app, &st)).unwrap();
        let text = buffer_text(&term);
        assert!(text.contains("Configuration"));
        assert!(text.contains("RESULT"));
        assert!(text.contains("ROC-AUC"));
        assert!(text.contains("Results & KPIs"));
    }

    #[test]
    fn renders_benchmark_training_progress() {
        let mut app = App::new();
        app.toggle_tab();
        let mut st = State::default();
        st.busy = true;
        st.last = Some(EpochMetrics { epoch: 5, train_loss: 0.4, val_metric: 0.66, learning_rate: 0.009, grad_norm: 1.2, ..Default::default() });
        st.loss_pts = vec![(0.0, 0.9), (1.0, 0.7), (2.0, 0.5), (3.0, 0.4)];
        st.val_pts = vec![(0.0, 0.5), (1.0, 0.6), (2.0, 0.63), (3.0, 0.66)];
        let mut term = Terminal::new(TestBackend::new(140, 40)).unwrap();
        term.draw(|f| ui(f, &app, &st)).unwrap();
        let text = buffer_text(&term);
        assert!(text.contains("TRAINING"));
        assert!(text.contains("Training loss"));
        assert!(text.contains("Validation metric"));
    }
}
