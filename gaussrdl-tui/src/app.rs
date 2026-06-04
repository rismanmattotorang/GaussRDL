//! TUI application state and the keyboard-driven parameter editor. Kept free of
//! rendering so the adjustment logic can be unit-tested.

/// A single configurable parameter: either a cycled choice or a bounded number.
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
            Setting::Choice { options, idx, .. } => options[*idx].clone(),
            Setting::Number { value, int, .. } => {
                if *int {
                    format!("{}", *value as i64)
                } else {
                    format!("{value:.4}")
                }
            }
        }
    }

    fn inc(&mut self) {
        match self {
            Setting::Choice { options, idx, .. } => *idx = (*idx + 1) % options.len(),
            Setting::Number { value, step, max, .. } => *value = (*value + *step).min(*max),
        }
    }

    fn dec(&mut self) {
        match self {
            Setting::Choice { options, idx, .. } => {
                *idx = (*idx + options.len() - 1) % options.len()
            }
            Setting::Number { value, step, min, .. } => *value = (*value - *step).max(*min),
        }
    }
}

/// The full editor state.
pub struct App {
    pub settings: Vec<Setting>,
    pub cursor: usize,
}

impl App {
    pub fn new() -> Self {
        let settings = vec![
            Setting::Choice {
                name: "model",
                options: vec!["HeteroSAGE".into(), "RGCN".into(), "GAT".into(), "RelGT".into()],
                idx: 3,
            },
            Setting::Choice { name: "task", options: vec!["churn".into(), "ltv".into()], idx: 0 },
            Setting::Number { name: "hidden_dim", value: 64.0, step: 8.0, min: 8.0, max: 512.0, int: true },
            Setting::Number { name: "num_layers", value: 2.0, step: 1.0, min: 1.0, max: 8.0, int: true },
            Setting::Number { name: "num_heads", value: 4.0, step: 1.0, min: 1.0, max: 16.0, int: true },
            Setting::Number { name: "dropout", value: 0.1, step: 0.05, min: 0.0, max: 0.9, int: false },
            Setting::Number { name: "lr", value: 0.01, step: 0.005, min: 0.0005, max: 0.5, int: false },
            Setting::Number { name: "epochs", value: 50.0, step: 10.0, min: 1.0, max: 1000.0, int: true },
            Setting::Number { name: "num_users", value: 300.0, step: 50.0, min: 50.0, max: 5000.0, int: true },
            Setting::Number { name: "patience", value: 0.0, step: 1.0, min: 0.0, max: 100.0, int: true },
        ];
        Self { settings, cursor: 0 }
    }

    pub fn next(&mut self) {
        self.cursor = (self.cursor + 1) % self.settings.len();
    }
    pub fn prev(&mut self) {
        self.cursor = (self.cursor + self.settings.len() - 1) % self.settings.len();
    }
    pub fn inc(&mut self) {
        self.settings[self.cursor].inc();
    }
    pub fn dec(&mut self) {
        self.settings[self.cursor].dec();
    }

    /// Current numeric value of a setting by name (0.0 if absent/not numeric).
    pub fn num(&self, name: &str) -> f64 {
        self.settings.iter().find_map(|s| match s {
            Setting::Number { name: n, value, .. } if *n == name => Some(*value),
            _ => None,
        }).unwrap_or(0.0)
    }

    /// Current selected choice of a setting by name.
    pub fn choice(&self, name: &str) -> String {
        self.settings.iter().find_map(|s| match s {
            Setting::Choice { name: n, options, idx } if *n == name => Some(options[*idx].clone()),
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
    fn defaults_and_lookup() {
        let app = App::new();
        assert_eq!(app.choice("model"), "RelGT");
        assert_eq!(app.choice("task"), "churn");
        assert_eq!(app.num("hidden_dim"), 64.0);
        assert_eq!(app.num("num_heads"), 4.0);
    }

    #[test]
    fn number_adjust_is_bounded() {
        let mut app = App::new();
        // cursor at model; move to dropout and clamp at min.
        while app.settings[app.cursor].name() != "dropout" {
            app.next();
        }
        for _ in 0..50 {
            app.dec();
        }
        assert_eq!(app.num("dropout"), 0.0);
        for _ in 0..100 {
            app.inc();
        }
        assert!((app.num("dropout") - 0.9).abs() < 1e-9);
    }

    #[test]
    fn choice_cycles() {
        let mut app = App::new();
        // cursor at model (index 0)
        assert_eq!(app.cursor, 0);
        app.inc(); // RelGT -> HeteroSAGE (wraps)
        assert_eq!(app.choice("model"), "HeteroSAGE");
        app.dec();
        assert_eq!(app.choice("model"), "RelGT");
    }
}
