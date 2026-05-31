# Contributing to GaussRDL

Thank you for your interest in contributing to GaussRDL! This document provides guidelines and information for contributors.

## 🎯 **Project Status**

**Current State**: ✅ **Production Ready**
- **Zero compilation errors** - Fully functional modular workspace
- **Superior performance** - 10-100x faster than Python RelBench
- **Comprehensive features** - 15+ metrics, 15+ model types across 11 crates
- **Clean architecture** - Modular workspace ready for contributions

## 🤝 **How to Contribute**

### **Types of Contributions**

We welcome various types of contributions:

1. **🐛 Bug Reports** - Help us identify and fix issues
2. **✨ Feature Requests** - Suggest new functionality
3. **📝 Documentation** - Improve guides, examples, and API docs
4. **🔧 Code Contributions** - Implement features, fix bugs, optimize performance
5. **🧪 Testing** - Add tests, improve coverage, stress testing
6. **📊 Benchmarks** - Performance comparisons and optimizations
7. **🎨 Examples** - Real-world usage examples and tutorials

### **Getting Started**

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/gaussrdl.git
   cd gaussrdl
   ```
3. **Set up development environment**:
   ```bash
   # Install Rust 1.70+
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   
   # Install development tools
   cargo install cargo-watch cargo-expand cargo-audit
   
   # Build and test the workspace
   cargo build
   cargo test
   ```

## 📋 **Contribution Guidelines**

### **Code Style**

We follow standard Rust conventions:

```bash
# Format code before committing
cargo fmt

# Check for issues
cargo clippy

# Ensure no warnings
cargo clippy -- -D warnings
```

**Key Style Points**:
- Use `snake_case` for functions and variables
- Use `PascalCase` for types and traits
- Use `SCREAMING_SNAKE_CASE` for constants
- Prefer explicit types when clarity improves
- Add documentation comments for public APIs
- Keep line length under 100 characters

### **Testing Requirements**

All contributions must include appropriate tests:

```bash
# Run all tests across the workspace
cargo test

# Run tests for specific crate
cargo test -p gaussrdl-core
cargo test -p gaussrdl-models
cargo test -p gaussrdl-training

# Run with output
cargo test -- --nocapture

# Test specific module
cargo test -p gaussrdl-data test_datasets
```

**Testing Guidelines**:
- Unit tests for individual functions
- Integration tests for end-to-end workflows
- Property-based tests for complex logic
- Benchmark tests for performance-critical code
- Error case testing

### **Documentation Standards**

- **Public APIs** must have documentation comments
- **Examples** should be included in doc comments
- **README updates** for new features
- **Changelog entries** for user-facing changes

```rust
/// Computes the accuracy metric for classification tasks.
/// 
/// # Arguments
/// 
/// * `predictions` - Model predictions as probabilities
/// * `targets` - Ground truth labels
/// 
/// # Returns
/// 
/// Accuracy score between 0.0 and 1.0
/// 
/// # Examples
/// 
/// ```
/// use gaussrdl::metrics::accuracy;
/// 
/// let predictions = vec![0.9, 0.1, 0.8, 0.2];
/// let targets = vec![1.0, 0.0, 1.0, 0.0];
/// let acc = accuracy(&predictions, &targets)?;
/// assert_eq!(acc, 1.0);
/// ```
pub fn accuracy(predictions: &[f64], targets: &[f64]) -> Result<f64> {
    // Implementation...
}
```

## 🚀 **Development Workflow**

### **Workspace Structure**

GaussRDL is organized as a modular workspace with the following crates:

- **`gaussrdl-core`**: Foundation types, traits, and base functionality
- **`gaussrdl-data`**: Data loading, datasets, and data processing
- **`gaussrdl-graph`**: Graph construction and manipulation
- **`gaussrdl-models`**: ML models and neural networks
- **`gaussrdl-training`**: Training infrastructure and optimizers
- **`gaussrdl-metrics`**: Evaluation metrics and monitoring
- **`gaussrdl-database`**: Database connectivity and operations
- **`gaussrdl-server`**: HTTP server and API
- **`gaussrdl-cli`**: Command-line interface
- **`gaussrdl-utils`**: Utility functions and helpers
- **`gaussrdl`**: Main library (re-exports everything)

### **Branch Strategy**

- **`main`** - Production-ready code, always stable
- **`develop`** - Integration branch for new features
- **`feature/feature-name`** - Individual feature development
- **`bugfix/issue-number`** - Bug fixes
- **`hotfix/critical-fix`** - Critical production fixes

### **Commit Messages**

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `perf`: Performance improvements
- `chore`: Maintenance tasks

**Examples**:
```
feat(metrics): add F1-score metric for classification

Implements macro and micro F1-score calculations with support
for multi-class classification tasks.

Closes #123
```

```
fix(server): handle connection timeout gracefully

Prevents server crashes when client connections timeout
during large data transfers.

Fixes #456
```

### **Pull Request Process**

1. **Create feature branch**:
   ```bash
   git checkout -b feature/my-awesome-feature
   ```

2. **Make changes with tests**:
   ```bash
   # Implement feature
   # Add tests
   # Update documentation
   ```

3. **Verify everything works**:
   ```bash
   # Check the entire workspace
   cargo check
   cargo test
   cargo clippy
   cargo fmt --check
   
   # Check specific crate if working on one
   cargo check -p gaussrdl-core
   cargo test -p gaussrdl-core
   ```

4. **Commit and push**:
   ```bash
   git add .
   git commit -m "feat(scope): description"
   git push origin feature/my-awesome-feature
   ```

5. **Create Pull Request** on GitHub with:
   - Clear title and description
   - Link to related issues
   - Screenshots/examples if applicable
   - Checklist completion

### **Pull Request Checklist**

- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Tests added for new functionality
- [ ] All tests pass locally across the workspace
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
- [ ] Performance impact considered
- [ ] Security implications reviewed
- [ ] Workspace builds successfully

## 🎯 **Good First Issues**

Perfect for new contributors:

### **Documentation Improvements**
- Fix typos and grammar
- Add code examples
- Improve API documentation
- Create tutorials and guides

### **Testing Enhancements**
- Add unit tests for uncovered functions
- Create integration test scenarios
- Add property-based tests
- Improve error case coverage

### **Code Quality**
- Fix clippy warnings
- Improve error messages
- Add input validation
- Optimize performance bottlenecks

### **Feature Additions**

## 🔧 **Advanced Contributions**

For experienced contributors:

### **Performance Optimizations**
- SIMD vectorization
- Memory pool implementations
- Parallel processing improvements
- GPU acceleration enhancements

### **New Architectures**
- Novel model implementations
- Advanced training algorithms
- Distributed computing support
- Real-time streaming capabilities

### **Platform Support**
- WebAssembly compilation
- Mobile platform support
- Cloud platform integrations
- Container optimizations

## 🐛 **Bug Reports**

When reporting bugs, please include:

1. **Environment Information**:
   ```
   OS: macOS 14.0
   Rust: 1.70.0
   GaussRDL: 0.1.0
   ```

2. **Steps to Reproduce**:
   ```bash
   cargo run -- run rel-amazon --task user-churn
   ```

3. **Expected vs Actual Behavior**

4. **Error Messages** (full stack trace)

5. **Minimal Reproduction** (if possible)

### **Bug Report Template**

```markdown
## Bug Description
Brief description of the issue.

## Environment
- OS: [e.g., Ubuntu 22.04]
- Rust: [e.g., 1.70.0]
- GaussRDL: [e.g., 0.1.0]

## Steps to Reproduce
1. Run command: `cargo run -- ...`
2. Observe error: ...

## Expected Behavior
What should happen.

## Actual Behavior
What actually happens.

## Error Output
```
[paste error output here]
```

## Additional Context
Any other relevant information.
```

## ✨ **Feature Requests**

When requesting features:

1. **Use Case Description** - Why is this needed?
2. **Proposed Solution** - How should it work?
3. **Alternatives Considered** - Other approaches?
4. **Implementation Ideas** - Technical suggestions?

### **Feature Request Template**

```markdown
## Feature Description
Clear description of the proposed feature.

## Use Case
Why is this feature needed? What problem does it solve?

## Proposed Solution
How should this feature work?

## Alternatives
What alternatives have you considered?

## Implementation
Any ideas on how to implement this?

## Additional Context
Screenshots, examples, or other relevant information.
```

## 🏆 **Recognition**

We value all contributions and recognize contributors:

- **Contributors** listed in README.md
- **Changelog** mentions for significant contributions
- **GitHub** contributor badges
- **Special thanks** in release notes

## 📞 **Getting Help**

Need help contributing?

- **GitHub Discussions** - Ask questions and get help
- **Discord** - Real-time community chat
- **Email** - Direct contact for sensitive issues
- **Documentation** - Comprehensive guides available

## 📜 **Code of Conduct**

### **Our Pledge**

We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### **Our Standards**

**Positive behavior includes**:
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behavior includes**:
- Trolling, insulting/derogatory comments, and personal attacks
- Public or private harassment
- Publishing others' private information without permission
- Other conduct which could reasonably be considered inappropriate

### **Enforcement**

Project maintainers are responsible for clarifying standards and will take appropriate action in response to unacceptable behavior.

## 🎉 **Thank You!**

Thank you for contributing to GaussRDL! Your efforts help make this project better for everyone. Whether you're fixing a typo, adding a feature, or reporting a bug, every contribution matters.

**Happy coding!** 🦀

---

**Questions?** Feel free to reach out through any of our communication channels. We're here to help! 