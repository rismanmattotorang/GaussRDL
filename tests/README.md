# GaussRelGT Test Suite

This directory contains comprehensive tests for GaussRelGT functionality, designed to match and exceed the testing standards of RelBench Python implementation.

## Test Structure

### Core Functionality Tests

#### 1. Dataset Tests (`test_datasets.rs`)
Comprehensive testing of dataset functionality:

**Basic Tests:**
- ✅ Dataset creation and initialization
- ✅ Dataset validation and integrity checks
- ✅ Timestamp validation (test > validation)
- ✅ Table structure and schema validation
- ✅ Dataset loading and saving operations

**Advanced Tests:**
- ✅ Cross-dataset comparison
- ✅ Memory usage validation
- ✅ Concurrent access testing
- ✅ Error handling with invalid paths
- ✅ Performance benchmarking

**Integration Tests:**
- ✅ Multiple dataset loading
- ✅ Dataset loading performance
- ✅ Thread safety verification

#### 2. Task Tests (`test_tasks.rs`) 
Comprehensive testing of task functionality:

**Registry Tests:**
- ✅ Task registry initialization
- ✅ Task provider creation
- ✅ Task type validation
- ✅ Metric assignment validation

**Task Loading Tests:**
- ✅ Different task type loading
- ✅ Table generation (train/val/test)
- ✅ Dataset compatibility verification
- ✅ Error handling for invalid tasks

**Evaluation Tests:**
- ✅ Basic prediction evaluation
- ✅ Edge cases (empty, single, large predictions)
- ✅ Performance benchmarking
- ✅ Evaluation consistency
- ✅ Concurrent evaluation

**Integration Tests:**
- ✅ Multiple task loading
- ✅ Dataset-task compatibility
- ✅ Memory usage monitoring
- ✅ Cross-platform compatibility

#### 3. Metrics Tests (`test_metrics.rs`)
Comprehensive testing of evaluation metrics:

**Classification Metrics:**
- ✅ AUROC (perfect, random, good classification)
- ✅ Accuracy (perfect, worst, mixed)
- ✅ Precision and Recall
- ✅ F1 Score and Macro F1
- ✅ Mathematical relationships verification

**Regression Metrics:**
- ✅ RMSE (perfect, test, large errors)
- ✅ MAE with known values
- ✅ R² with linear relationships
- ✅ MAPE calculations
- ✅ Metric bounds validation

**Ranking Metrics:**
- ✅ MAP@k with different k values
- ✅ MRR (first, second relevant item)
- ✅ NDCG@k calculations
- ✅ Hits@k validation
- ✅ Ranking property verification

**Integration Tests:**
- ✅ Metric consistency across runs
- ✅ Performance with large datasets
- ✅ Edge case handling
- ✅ Cross-metric relationship validation

#### 4. Integration Tests (`test_integration.rs`)
End-to-end system testing:

**Workflow Tests:**
- ✅ Complete end-to-end workflow
- ✅ Multiple dataset/task combinations
- ✅ Task registry integration
- ✅ Dataset-task compatibility

**Performance Tests:**
- ✅ Loading performance benchmarks
- ✅ Evaluation performance scaling
- ✅ Memory usage monitoring
- ✅ System stress testing

**Reliability Tests:**
- ✅ Error handling validation
- ✅ Resource cleanup verification
- ✅ Cross-platform compatibility
- ✅ Version compatibility

## Running Tests

### Run All Tests
```bash
cargo test
```

### Run Specific Test Files
```bash
# Dataset tests only
cargo test test_datasets

# Task tests only
cargo test test_tasks

# Metrics tests only
cargo test test_metrics

# Integration tests only
cargo test test_integration
```

### Run with Output
```bash
# Show test output
cargo test -- --nocapture

# Show test output with verbose logging
RUST_LOG=debug cargo test -- --nocapture
```

### Run Specific Test Functions
```bash
# Run specific test
cargo test test_end_to_end_workflow

# Run tests matching pattern
cargo test dataset_creation

# Run with threads
cargo test -- --test-threads=4
```

## Test Categories

### Unit Tests
- Individual component testing
- Function-level validation
- Error condition testing
- Boundary value testing

### Integration Tests
- Component interaction testing
- End-to-end workflow validation
- Cross-module functionality
- System behavior verification

### Performance Tests
- Speed benchmarking
- Memory usage validation
- Scalability testing
- Resource efficiency

### Stress Tests
- Large dataset handling
- Concurrent access
- Resource exhaustion
- Error recovery

## Test Data and Mocking

### Mock Data Strategy
- Synthetic datasets for consistent testing
- Predictable data patterns
- Minimal resource usage
- Fast test execution

### Test Data Characteristics
- **Datasets**: Mock rel-amazon, rel-f1, rel-hm
- **Tasks**: Classification and regression tasks
- **Predictions**: Generated ranges and patterns
- **Metrics**: Known expected values

## Expected Test Results

### Performance Benchmarks
- Task loading: < 10 seconds
- Data loading: < 5 seconds  
- Evaluation (1K predictions): < 100ms
- Evaluation (10K predictions): < 1s
- Memory usage: < 1GB for test datasets

### Quality Metrics
- Test coverage: > 90%
- All tests pass consistently
- No memory leaks detected
- Cross-platform compatibility verified

## Test Environment Setup

### Prerequisites
```bash
# Install Rust and Cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone and build
git clone <repository>
cd gaussrelgt
cargo build
```

### Environment Variables
```bash
# Enable debug logging
export RUST_LOG=debug

# Set test timeout
export RUST_TEST_TIME_UNIT=60s

# Parallel test execution
export RUST_TEST_THREADS=4
```

## Continuous Integration

### Automated Testing
- ✅ All tests run on commit
- ✅ Performance regression detection
- ✅ Memory leak detection
- ✅ Cross-platform validation

### Quality Gates
- ✅ All tests must pass
- ✅ No new warnings introduced
- ✅ Performance within acceptable ranges
- ✅ Memory usage stable

## Test Maintenance

### Adding New Tests
1. **Follow naming convention**: `test_<functionality>`
2. **Include documentation**: Clear test purpose
3. **Add assertions**: Comprehensive validation
4. **Handle errors**: Graceful failure modes
5. **Update README**: Document new test coverage

### Test Best Practices
- **Isolation**: Tests don't depend on each other
- **Determinism**: Consistent results across runs
- **Speed**: Fast execution for development
- **Clarity**: Clear test intent and assertions
- **Coverage**: Test both success and failure paths

## Debugging Test Failures

### Common Issues
- **Mock data mismatch**: Check synthetic data generation
- **Timing issues**: Add appropriate timeouts
- **Resource conflicts**: Ensure proper cleanup
- **Platform differences**: Use platform-agnostic code

### Debugging Commands
```bash
# Run single test with full output
cargo test test_specific_function -- --nocapture --exact

# Run with backtraces
RUST_BACKTRACE=1 cargo test

# Run with debug symbols
cargo test --manifest-path Cargo.toml --target-dir target/debug
```

## Relationship to RelBench Python Tests

This test suite provides equivalent and enhanced coverage compared to RelBench Python tests:

- ✅ **Dataset loading tests** → Enhanced with performance and concurrency
- ✅ **Task evaluation tests** → Extended with edge cases and stress testing  
- ✅ **Metrics calculation tests** → Comprehensive mathematical validation
- ✅ **Integration tests** → End-to-end workflow verification
- ✅ **Performance tests** → Benchmarking and scaling validation
- ✅ **Error handling tests** → Robust failure mode testing

## Contributing to Tests

### Contribution Guidelines
1. **Write comprehensive tests** for new features
2. **Maintain existing test compatibility**
3. **Document test purpose and expectations**
4. **Include performance considerations**
5. **Verify cross-platform compatibility**

### Test Review Checklist
- [ ] Tests cover happy path scenarios
- [ ] Tests cover error conditions
- [ ] Tests include performance validation
- [ ] Tests are deterministic and isolated
- [ ] Tests follow project naming conventions
- [ ] Documentation is clear and complete

## Related Documentation

- [Examples README](../examples/README.md) - Usage examples
- [Main README](../README.md) - Project overview
- [Developer Guide](../DEVELOPERGUIDE.md) - Development setup
- [User Guide](../USERGUIDE.md) - Usage instructions 