//! TUI application state: a two-tab editor (Datasets / Benchmark) kept free of
//! rendering so the navigation and configuration logic can be unit-tested.

use gaussrdl_bench::{catalog, find, DatasetManager, DatasetStatus};

/// Which workspace tab is active.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Datasets,
    Benchmark,
}

/// A configurable parameter: cycled choice or bounded number.
pub enum Setting {
    Choice { name: &'static str, options: Vec<String>, idx: usize },
    Number { name: &'static str, value: f64, step: f64, min: f64, max: f64, int: bool },
}

impl Setting {
    pub fn name(&self) -> &'static str {
        match self {
            Setting::Choice { name, .. } => name,
            Setting::Number { name, .. } => name,
        }
    }
    pub fn display(&self) -> String {
        match self {
            Setting::Choice { options, idx, .. } => options.get(*idx).cloned().unwrap_or_default(),
            Setting::Number { value, int, .. } => {
                if *int { format!("{}", *value as i64) } else { format!("{value:.4}") }
            }
        }
    }
    fn inc(&mut self) {
        match self {
            Setting::Choice { options, idx, .. } => {
                if !options.is_empty() { *idx = (*idx + 1) % options.len() }
            }
            Setting::Number { value, step, max, .. } => *value = (*value + *step).min(*max),
        }
    }
    fn dec(&mut self) {
        match self {
            Setting::Choice { options, idx, .. } => {
                if !options.is_empty() { *idx = (*idx + options.len() - 1) % options.len() }
            }
            Setting::Number { value, step, min, .. } => *value = (*value - *step).max(*min),
        }
    }
}

/// A catalog row shown in the Datasets tab.
pub struct DatasetRow {
    pub id: String,
    pub display: String,
    pub official: bool,
    pub status: String,
}

pub struct App {
    pub tab: Tab,
    pub settings: Vec<Setting>,
    pub cursor: usize,
    pub datasets: Vec<DatasetRow>,
    pub ds_cursor: usize,
}

impl App {
    pub fn new() -> Self {
        let ids: Vec<String> = catalog().iter().map(|d| d.id.clone()).collect();
        let first_tasks = catalog()
            .first()
            .map(|d| d.tasks.iter().map(|t| t.name.clone()).collect::<Vec<_>>())
            .unwrap_or_default();
        let settings = vec![
            Setting::Choice { name: "dataset", options: ids, idx: 0 },
            Setting::Choice { name: "task", options: first_tasks, idx: 0 },
            Setting::Choice {
                name: "model",
                options: vec!["HeteroSAGE".into(), "RGCN".into(), "GAT".into(), "RelGT".into()],
                idx: 3,
            },
            Setting::Number { name: "hidden_dim", value: 64.0, step: 8.0, min: 8.0, max: 512.0, int: true },
            Setting::Number { name: "num_layers", value: 2.0, step: 1.0, min: 1.0, max: 8.0, int: true },
            Setting::Number { name: "num_heads", value: 4.0, step: 1.0, min: 1.0, max: 16.0, int: true },
            Setting::Number { name: "dropout", value: 0.1, step: 0.05, min: 0.0, max: 0.9, int: false },
            Setting::Number { name: "epochs", value: 40.0, step: 10.0, min: 1.0, max: 1000.0, int: true },
            Setting::Number { name: "seeds", value: 3.0, step: 1.0, min: 1.0, max: 5.0, int: true },
        ];
        let datasets = catalog()
            .iter()
            .map(|d| DatasetRow {
                id: d.id.clone(),
                display: d.display_name.clone(),
                official: d.official,
                status: "…".into(),
            })
            .collect();
        Self { tab: Tab::Datasets, settings, cursor: 0, datasets, ds_cursor: 0 }
    }

    pub fn toggle_tab(&mut self) {
        self.tab = match self.tab {
            Tab::Datasets => Tab::Benchmark,
            Tab::Benchmark => Tab::Datasets,
        };
    }

    pub fn next(&mut self) {
        match self.tab {
            Tab::Datasets => {
                if !self.datasets.is_empty() {
                    self.ds_cursor = (self.ds_cursor + 1) % self.datasets.len();
                }
            }
            Tab::Benchmark => self.cursor = (self.cursor + 1) % self.settings.len(),
        }
    }
    pub fn prev(&mut self) {
        match self.tab {
            Tab::Datasets => {
                if !self.datasets.is_empty() {
                    self.ds_cursor = (self.ds_cursor + self.datasets.len() - 1) % self.datasets.len();
                }
            }
            Tab::Benchmark => self.cursor = (self.cursor + self.settings.len() - 1) % self.settings.len(),
        }
    }
    pub fn inc(&mut self) {
        if self.tab == Tab::Benchmark {
            self.settings[self.cursor].inc();
            if self.settings[self.cursor].name() == "dataset" {
                self.rebuild_tasks();
            }
        }
    }
    pub fn dec(&mut self) {
        if self.tab == Tab::Benchmark {
            self.settings[self.cursor].dec();
            if self.settings[self.cursor].name() == "dataset" {
                self.rebuild_tasks();
            }
        }
    }

    /// Rebuild the task choice for the currently-selected dataset.
    fn rebuild_tasks(&mut self) {
        let id = self.choice("dataset");
        if let Some(info) = find(&id) {
            let tasks: Vec<String> = info.tasks.iter().map(|t| t.name.clone()).collect();
            for s in self.settings.iter_mut() {
                if let Setting::Choice { name, options, idx } = s {
                    if *name == "task" {
                        *options = tasks.clone();
                        *idx = 0;
                    }
                }
            }
        }
    }

    /// Refresh dataset statuses from the on-disk cache.
    pub fn refresh_datasets(&mut self, mgr: &DatasetManager) {
        for row in self.datasets.iter_mut() {
            if let Some(info) = find(&row.id) {
                row.status = match mgr.status(&info) {
                    DatasetStatus::Ready => "ready".into(),
                    DatasetStatus::NotDownloaded => {
                        if info.offline_capable() { "not prepared".into() } else { "not downloaded".into() }
                    }
                    DatasetStatus::Error(e) => format!("error: {e}"),
                };
            }
        }
    }

    pub fn selected_dataset_id(&self) -> Option<String> {
        self.datasets.get(self.ds_cursor).map(|r| r.id.clone())
    }

    pub fn num(&self, name: &str) -> f64 {
        self.settings.iter().find_map(|s| match s {
            Setting::Number { name: n, value, .. } if *n == name => Some(*value),
            _ => None,
        }).unwrap_or(0.0)
    }
    pub fn choice(&self, name: &str) -> String {
        self.settings.iter().find_map(|s| match s {
            Setting::Choice { name: n, options, idx } if *n == name => options.get(*idx).cloned(),
            _ => None,
        }).unwrap_or_default()
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults() {
        let app = App::new();
        assert_eq!(app.tab, Tab::Datasets);
        assert_eq!(app.choice("model"), "RelGT");
        assert!(!app.choice("dataset").is_empty());
        assert!(!app.datasets.is_empty());
    }

    #[test]
    fn tab_toggle_and_nav() {
        let mut app = App::new();
        app.toggle_tab();
        assert_eq!(app.tab, Tab::Benchmark);
        // Benchmark nav moves the settings cursor.
        let c0 = app.cursor;
        app.next();
        assert_ne!(app.cursor, c0);
    }

    #[test]
    fn changing_dataset_rebuilds_tasks() {
        let mut app = App::new();
        app.toggle_tab(); // Benchmark
        // cursor at dataset (0)
        assert_eq!(app.settings[app.cursor].name(), "dataset");
        let t0 = app.choice("task");
        // cycle dataset until the id changes to one with different tasks
        let d0 = app.choice("dataset");
        app.inc();
        assert_ne!(app.choice("dataset"), d0);
        // task is always valid for the selected dataset
        let info = find(&app.choice("dataset")).unwrap();
        assert!(info.tasks.iter().any(|t| t.name == app.choice("task")));
        let _ = t0;
    }

    #[test]
    fn refresh_marks_status() {
        let root = std::env::temp_dir().join(format!("gaussrdl_tui_test_{}", std::process::id()));
        let mgr = DatasetManager::new(&root);
        let mut app = App::new();
        app.refresh_datasets(&mgr);
        assert!(app.datasets.iter().all(|r| !r.status.is_empty()));
    }
}
