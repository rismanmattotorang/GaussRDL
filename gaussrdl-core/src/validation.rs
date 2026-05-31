//! Validation utilities for GaussRDL
//! 
//! This module provides comprehensive validation functionality for datasets,
//! tables, and other data structures.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::error::{ValidationError, ValidationResult};
use crate::types::Table;

/// Validation rule for data validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationRule {
    /// Check if value is not null
    NotNull,
    /// Check if value is within range
    Range { min: f64, max: f64 },
    /// Check if value matches pattern
    Pattern { regex: String },
    /// Check if value is in allowed values
    In { values: Vec<String> },
    /// Check if value is unique
    Unique,
    /// Check if value is positive
    Positive,
    /// Check if value is non-negative
    NonNegative,
    /// Check if value is integer
    Integer,
    /// Check if value is float
    Float,
    /// Check if value is boolean
    Boolean,
    /// Check if value is valid timestamp
    Timestamp,
    /// Check if value is valid UUID
    Uuid,
    /// Check if value is valid email
    Email,
    /// Check if value is valid URL
    Url,
    /// Check if value is valid JSON
    Json,
    /// Custom validation rule
    Custom { name: String, validator: String },
}

impl ValidationRule {
    /// Get rule name
    pub fn name(&self) -> &str {
        match self {
            Self::NotNull => "not_null",
            Self::Range { .. } => "range",
            Self::Pattern { .. } => "pattern",
            Self::In { .. } => "in",
            Self::Unique => "unique",
            Self::Positive => "positive",
            Self::NonNegative => "non_negative",
            Self::Integer => "integer",
            Self::Float => "float",
            Self::Boolean => "boolean",
            Self::Timestamp => "timestamp",
            Self::Uuid => "uuid",
            Self::Email => "email",
            Self::Url => "url",
            Self::Json => "json",
            Self::Custom { name, .. } => name,
        }
    }
    
    /// Get rule description
    pub fn description(&self) -> String {
        match self {
            Self::NotNull => "Value must not be null".to_string(),
            Self::Range { min, max } => format!("Value must be between {} and {}", min, max),
            Self::Pattern { regex } => format!("Value must match pattern: {}", regex),
            Self::In { values } => format!("Value must be one of: {}", values.join(", ")),
            Self::Unique => "Value must be unique".to_string(),
            Self::Positive => "Value must be positive".to_string(),
            Self::NonNegative => "Value must be non-negative".to_string(),
            Self::Integer => "Value must be an integer".to_string(),
            Self::Float => "Value must be a float".to_string(),
            Self::Boolean => "Value must be a boolean".to_string(),
            Self::Timestamp => "Value must be a valid timestamp".to_string(),
            Self::Uuid => "Value must be a valid UUID".to_string(),
            Self::Email => "Value must be a valid email".to_string(),
            Self::Url => "Value must be a valid URL".to_string(),
            Self::Json => "Value must be valid JSON".to_string(),
            Self::Custom { name, .. } => format!("Custom validation: {}", name),
        }
    }
}

/// Validation context for tracking validation state
#[derive(Debug, Clone)]
pub struct ValidationContext {
    /// Current table being validated
    pub table_name: Option<String>,
    /// Current column being validated
    pub column_name: Option<String>,
    /// Current row being validated
    pub row_index: Option<usize>,
    /// Validation errors collected
    pub errors: Vec<ValidationError>,
    /// Validation warnings collected
    pub warnings: Vec<String>,
    /// Custom validation data
    pub metadata: HashMap<String, String>,
}

impl ValidationContext {
    /// Create a new validation context
    pub fn new() -> Self {
        Self {
            table_name: None,
            column_name: None,
            row_index: None,
            errors: Vec::new(),
            warnings: Vec::new(),
            metadata: HashMap::new(),
        }
    }
    
    /// Set table name
    pub fn with_table(mut self, table_name: impl Into<String>) -> Self {
        self.table_name = Some(table_name.into());
        self
    }
    
    /// Set column name
    pub fn with_column(mut self, column_name: impl Into<String>) -> Self {
        self.column_name = Some(column_name.into());
        self
    }
    
    /// Set row index
    pub fn with_row(mut self, row_index: usize) -> Self {
        self.row_index = Some(row_index);
        self
    }
    
    /// Add validation error
    pub fn add_error(&mut self, error: ValidationError) {
        self.errors.push(error);
    }
    
    /// Add validation warning
    pub fn add_warning(&mut self, warning: impl Into<String>) {
        self.warnings.push(warning.into());
    }
    
    /// Set metadata
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }
    
    /// Get metadata
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }
    
    /// Check if validation has errors
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }
    
    /// Check if validation has warnings
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }
    
    /// Get error count
    pub fn error_count(&self) -> usize {
        self.errors.len()
    }
    
    /// Get warning count
    pub fn warning_count(&self) -> usize {
        self.warnings.len()
    }
    
    /// Clear all errors and warnings
    pub fn clear(&mut self) {
        self.errors.clear();
        self.warnings.clear();
    }
    
    /// Convert to result
    pub fn into_result<T>(self, value: T) -> ValidationResult<T> {
        if self.has_errors() {
            Err(self.errors.into_iter().next().unwrap())
        } else {
            Ok(value)
        }
    }
}

/// Validator trait for implementing custom validators
pub trait Validator: Send + Sync {
    /// Validate a value
    fn validate(&self, value: &str, context: &mut ValidationContext) -> ValidationResult<()>;
    
    /// Get validator name
    fn name(&self) -> &str;
    
    /// Get validator description
    fn description(&self) -> &str;
}

/// Built-in validators
pub mod validators {
    use super::*;
    use regex::Regex;
    use uuid::Uuid;
    use chrono::DateTime;
    use serde_json::Value;
    
    /// Not null validator
    #[derive(Debug)]
    pub struct NotNullValidator;
    
    impl Validator for NotNullValidator {
        fn validate(&self, value: &str, context: &mut ValidationContext) -> ValidationResult<()> {
            if value.is_empty() || value == "null" || value == "NULL" {
                context.add_error(ValidationError::field(
                    context.column_name.clone().unwrap_or_default(),
                    "Value cannot be null or empty"
                ));
                return Err(ValidationError::field(
                    context.column_name.clone().unwrap_or_default(),
                    "Value cannot be null or empty"
                ));
            }
            Ok(())
        }
        
        fn name(&self) -> &str {
            "not_null"
        }
        
        fn description(&self) -> &str {
            "Validates that value is not null or empty"
        }
    }
    
    /// Range validator
    #[derive(Debug)]
    pub struct RangeValidator {
        min: f64,
        max: f64,
    }
    
    impl RangeValidator {
        pub fn new(min: f64, max: f64) -> Self {
            Self { min, max }
        }
    }
    
    impl Validator for RangeValidator {
        fn validate(&self, value: &str, context: &mut ValidationContext) -> ValidationResult<()> {
            match value.parse::<f64>() {
                Ok(num) => {
                    if num < self.min || num > self.max {
                        let error = ValidationError::field_with_value(
                            context.column_name.clone().unwrap_or_default(),
                            format!("Value must be between {} and {}", self.min, self.max),
                            value.to_string()
                        );
                        context.add_error(error.clone());
                        return Err(error);
                    }
                }
                Err(_) => {
                    let error = ValidationError::field_with_value(
                        context.column_name.clone().unwrap_or_default(),
                        "Value must be a valid number",
                        value.to_string()
                    );
                    context.add_error(error.clone());
                    return Err(error);
                }
            }
            Ok(())
        }
        
        fn name(&self) -> &str {
            "range"
        }
        
        fn description(&self) -> &str {
            "Validates that value is within a range"
        }
    }
    
    /// Pattern validator
    #[derive(Debug)]
    pub struct PatternValidator {
        regex: Regex,
    }
    
    impl PatternValidator {
        pub fn new(pattern: &str) -> Result<Self, regex::Error> {
            Ok(Self {
                regex: Regex::new(pattern)?,
            })
        }
    }
    
    impl Validator for PatternValidator {
        fn validate(&self, value: &str, context: &mut ValidationContext) -> ValidationResult<()> {
            if !self.regex.is_match(value) {
                let error = ValidationError::field_with_value(
                    context.column_name.clone().unwrap_or_default(),
                    "Value does not match required pattern",
                    value.to_string()
                );
                context.add_error(error.clone());
                return Err(error);
            }
            Ok(())
        }
        
        fn name(&self) -> &str {
            "pattern"
        }
        
        fn description(&self) -> &str {
            "Validates that value matches a regular expression pattern"
        }
    }
    
    /// Email validator
    #[derive(Debug)]
    pub struct EmailValidator;
    
    impl Validator for EmailValidator {
        fn validate(&self, value: &str, context: &mut ValidationContext) -> ValidationResult<()> {
            // Simple email validation
            let email_regex = Regex::new(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$").unwrap();
            if !email_regex.is_match(value) {
                let error = ValidationError::field_with_value(
                    context.column_name.clone().unwrap_or_default(),
                    "Value must be a valid email address",
                    value.to_string()
                );
                context.add_error(error.clone());
                return Err(error);
            }
            Ok(())
        }
        
        fn name(&self) -> &str {
            "email"
        }
        
        fn description(&self) -> &str {
            "Validates that value is a valid email address"
        }
    }
    
    /// UUID validator
    #[derive(Debug)]
    pub struct UuidValidator;
    
    impl Validator for UuidValidator {
        fn validate(&self, value: &str, context: &mut ValidationContext) -> ValidationResult<()> {
            if Uuid::parse_str(value).is_err() {
                let error = ValidationError::field_with_value(
                    context.column_name.clone().unwrap_or_default(),
                    "Value must be a valid UUID",
                    value.to_string()
                );
                context.add_error(error.clone());
                return Err(error);
            }
            Ok(())
        }
        
        fn name(&self) -> &str {
            "uuid"
        }
        
        fn description(&self) -> &str {
            "Validates that value is a valid UUID"
        }
    }
    
    /// Timestamp validator
    #[derive(Debug)]
    pub struct TimestampValidator;
    
    impl Validator for TimestampValidator {
        fn validate(&self, value: &str, context: &mut ValidationContext) -> ValidationResult<()> {
            if DateTime::parse_from_rfc3339(value).is_err() {
                let error = ValidationError::field_with_value(
                    context.column_name.clone().unwrap_or_default(),
                    "Value must be a valid RFC3339 timestamp",
                    value.to_string()
                );
                context.add_error(error.clone());
                return Err(error);
            }
            Ok(())
        }
        
        fn name(&self) -> &str {
            "timestamp"
        }
        
        fn description(&self) -> &str {
            "Validates that value is a valid RFC3339 timestamp"
        }
    }
    
    /// JSON validator
    #[derive(Debug)]
    pub struct JsonValidator;
    
    impl Validator for JsonValidator {
        fn validate(&self, value: &str, context: &mut ValidationContext) -> ValidationResult<()> {
            if serde_json::from_str::<Value>(value).is_err() {
                let error = ValidationError::field_with_value(
                    context.column_name.clone().unwrap_or_default(),
                    "Value must be valid JSON",
                    value.to_string()
                );
                context.add_error(error.clone());
                return Err(error);
            }
            Ok(())
        }
        
        fn name(&self) -> &str {
            "json"
        }
        
        fn description(&self) -> &str {
            "Validates that value is valid JSON"
        }
    }
}

/// Schema validator for validating table schemas
pub struct SchemaValidator {
    rules: HashMap<String, Vec<ValidationRule>>,
}

impl SchemaValidator {
    /// Create a new schema validator
    pub fn new() -> Self {
        Self {
            rules: HashMap::new(),
        }
    }
    
    /// Add validation rule for a column
    pub fn add_rule(&mut self, column: impl Into<String>, rule: ValidationRule) {
        self.rules.entry(column.into()).or_insert_with(Vec::new).push(rule);
    }
    
    /// Validate a table schema
    pub fn validate_table(&self, table: &Table, context: &mut ValidationContext) -> ValidationResult<()> {
        // Validate table structure
        if table.data.height() == 0 {
            context.add_error(ValidationError::schema("Table is empty"));
        }
        
        // Validate each column
        for (column_name, rules) in &self.rules {
            let series = table.data.column(column_name)
                .map_err(|e| ValidationError::schema(format!("Column '{}' not found: {}", column_name, e)))?;
            for rule in rules {
                self.validate_column_rule(series, rule, context)?;
            }
        }
        
        if context.has_errors() {
            Err(context.errors.clone().into_iter().next().unwrap())
        } else {
            Ok(())
        }
    }
    
    /// Validate a column rule
    fn validate_column_rule(
        &self,
        series: &polars::prelude::Series,
        rule: &ValidationRule,
        context: &mut ValidationContext,
    ) -> ValidationResult<()> {
        match rule {
            ValidationRule::NotNull => {
                if series.null_count() > 0 {
                    context.add_error(ValidationError::field(
                        context.column_name.clone().unwrap_or_default(),
                        "Column contains null values"
                    ));
                }
            }
            ValidationRule::Unique => {
                if series.n_unique().unwrap_or(0) != series.len() {
                    context.add_error(ValidationError::field(
                        context.column_name.clone().unwrap_or_default(),
                        "Column values are not unique"
                    ));
                }
            }
            ValidationRule::Positive => {
                if let Ok(numeric_series) = series.f64() {
                    if numeric_series.into_iter().any(|v| v.map_or(false, |x| x <= 0.0)) {
                        context.add_error(ValidationError::field(
                            context.column_name.clone().unwrap_or_default(),
                            "Column contains non-positive values"
                        ));
                    }
                }
            }
            ValidationRule::NonNegative => {
                if let Ok(numeric_series) = series.f64() {
                    if numeric_series.into_iter().any(|v| v.map_or(false, |x| x < 0.0)) {
                        context.add_error(ValidationError::field(
                            context.column_name.clone().unwrap_or_default(),
                            "Column contains negative values"
                        ));
                    }
                }
            }
            _ => {
                // For other rules, we would need to implement specific validation logic
                context.add_warning(format!("Rule '{}' not implemented for column validation", rule.name()));
            }
        }
        
        Ok(())
    }
}

/// Data validator for validating data values
pub struct DataValidator {
    validators: HashMap<String, Box<dyn Validator>>,
}

impl DataValidator {
    /// Create a new data validator
    pub fn new() -> Self {
        Self {
            validators: HashMap::new(),
        }
    }
    
    /// Add a validator
    pub fn add_validator(&mut self, name: impl Into<String>, validator: Box<dyn Validator>) {
        self.validators.insert(name.into(), validator);
    }
    
    /// Validate a value
    pub fn validate_value(
        &self,
        value: &str,
        validator_name: &str,
        context: &mut ValidationContext,
    ) -> ValidationResult<()> {
        if let Some(validator) = self.validators.get(validator_name) {
            validator.validate(value, context)
        } else {
            Err(ValidationError::field(
                context.column_name.clone().unwrap_or_default(),
                format!("Validator '{}' not found", validator_name)
            ))
        }
    }
    
    /// Get built-in validators
    pub fn with_builtin_validators() -> Self {
        let mut validator = Self::new();
        
        validator.add_validator("not_null".to_string(), Box::new(validators::NotNullValidator));
        validator.add_validator("email".to_string(), Box::new(validators::EmailValidator));
        validator.add_validator("uuid".to_string(), Box::new(validators::UuidValidator));
        validator.add_validator("timestamp".to_string(), Box::new(validators::TimestampValidator));
        validator.add_validator("json".to_string(), Box::new(validators::JsonValidator));
        
        validator
    }
}

/// Validation result with detailed information
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// Whether validation passed
    pub is_valid: bool,
    /// Number of errors
    pub error_count: usize,
    /// Number of warnings
    pub warning_count: usize,
    /// Validation errors
    pub errors: Vec<ValidationError>,
    /// Validation warnings
    pub warnings: Vec<String>,
    /// Validation metadata
    pub metadata: HashMap<String, String>,
}

impl ValidationReport {
    /// Create a new validation report
    pub fn new() -> Self {
        Self {
            is_valid: true,
            error_count: 0,
            warning_count: 0,
            errors: Vec::new(),
            warnings: Vec::new(),
            metadata: HashMap::new(),
        }
    }
    
    /// Add error
    pub fn add_error(&mut self, error: ValidationError) {
        self.errors.push(error);
        self.error_count = self.errors.len();
        self.is_valid = false;
    }
    
    /// Add warning
    pub fn add_warning(&mut self, warning: impl Into<String>) {
        self.warnings.push(warning.into());
        self.warning_count = self.warnings.len();
    }
    
    /// Set metadata
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }
    
    /// Convert to result
    pub fn into_result<T>(self, value: T) -> ValidationResult<T> {
        if self.is_valid {
            Ok(value)
        } else {
            Err(self.errors.into_iter().next().unwrap())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Table;
    use polars::prelude::*;

    #[test]
    fn test_validation_rule() {
        let rule = ValidationRule::Range { min: 0.0, max: 100.0 };
        assert_eq!(rule.name(), "range");
        assert!(rule.description().contains("between 0 and 100"));
    }

    #[test]
    fn test_validation_context() {
        let mut context = ValidationContext::new()
            .with_table("test_table")
            .with_column("test_column");
        
        context.add_error(ValidationError::field("test_column".to_string(), "Test error".to_string()));
        assert!(context.has_errors());
        assert_eq!(context.error_count(), 1);
    }

    #[test]
    fn test_not_null_validator() {
        let validator = validators::NotNullValidator;
        let mut context = ValidationContext::new().with_column("test");
        
        assert!(validator.validate("", &mut context).is_err());
        assert!(validator.validate("null", &mut context).is_err());
        assert!(validator.validate("valid", &mut context).is_ok());
    }

    #[test]
    fn test_range_validator() {
        let validator = RangeValidator::new(0.0, 100.0);
        let mut context = ValidationContext::new().with_column("test");
        
        assert!(validator.validate("50", &mut context).is_ok());
        assert!(validator.validate("-1", &mut context).is_err());
        assert!(validator.validate("101", &mut context).is_err());
        assert!(validator.validate("invalid", &mut context).is_err());
    }

    #[test]
    fn test_email_validator() {
        let validator = validators::EmailValidator;
        let mut context = ValidationContext::new().with_column("email");
        
        assert!(validator.validate("test@example.com", &mut context).is_ok());
        assert!(validator.validate("invalid-email", &mut context).is_err());
    }

    #[test]
    fn test_schema_validator() {
        let mut validator = SchemaValidator::new();
        validator.add_rule("value".to_string(), ValidationRule::NotNull);
        
        let df = DataFrame::new(vec![
            Series::new("value", &[1, 2, 3]),
        ]).unwrap();
        let table = Table::new(df);
        
        let mut context = ValidationContext::new().with_table("test");
        assert!(validator.validate_table(&table, &mut context).is_ok());
    }

    #[test]
    fn test_data_validator() {
        let mut validator = DataValidator::with_builtin_validators();
        let mut context = ValidationContext::new().with_column("email");
        
        assert!(validator.validate_value("test@example.com", "email", &mut context).is_ok());
        assert!(validator.validate_value("invalid", "email", &mut context).is_err());
    }

    #[test]
    fn test_validation_report() {
        let mut report = ValidationReport::new();
        report.add_error(ValidationError::field("test".to_string(), "Test error".to_string()));
        report.add_warning("Test warning");
        
        assert!(!report.is_valid);
        assert_eq!(report.error_count, 1);
        assert_eq!(report.warning_count, 1);
    }
} 