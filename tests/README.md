# EnvTorch Test Suite

This directory contains the comprehensive test suite for EnvTorch, covering all aspects of the CodeAct environment implementation.

## Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                 # Pytest configuration and fixtures
├── utils.py                    # Test utilities and helpers
├── test_runner.py              # Test runner script
├── unit/                       # Unit tests for individual components
│   ├── test_types.py          # Type system tests
│   ├── test_executor.py       # Python executor tests
│   ├── test_environment.py    # Environment tests
│   ├── test_transforms.py     # Transform system tests
│   └── test_mcp.py           # MCP integration tests
├── integration/               # Integration tests for full workflows
│   └── test_full_workflows.py # Complete workflow tests
├── test_performance.py        # Performance and benchmark tests
└── test_edge_cases.py        # Edge cases and boundary conditions
```

## Test Categories

### Unit Tests (`tests/unit/`)
Test individual components in isolation:
- **Type System** (`test_types.py`): Action, Observation, State classes
- **Executor** (`test_executor.py`): Python code execution engine
- **Environment** (`test_environment.py`): CodeAct environment core
- **Transforms** (`test_transforms.py`): Reward and observation transforms
- **MCP Integration** (`test_mcp.py`): Tool integration system

### Integration Tests (`tests/integration/`)
Test complete workflows and component interactions:
- Agent problem-solving workflows
- Multi-step data processing
- Tool usage scenarios
- RL training scenarios
- Complex real-world use cases

### Performance Tests (`tests/test_performance.py`)
Benchmark and validate performance characteristics:
- Execution speed benchmarks
- Memory usage validation
- Scalability testing
- Concurrent usage patterns

### Edge Case Tests (`tests/test_edge_cases.py`)
Test robustness against unusual conditions:
- Input validation edge cases
- Error handling scenarios
- State management edge cases
- Transform edge cases
- MCP tool error conditions

## Test Markers

Tests are organized using pytest markers:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.performance` - Performance tests
- `@pytest.mark.edge_case` - Edge case tests
- `@pytest.mark.slow` - Tests that take >5 seconds
- `@pytest.mark.mcp` - Tests involving MCP tools

## Running Tests

### Using the Test Runner

The `test_runner.py` script provides convenient commands:

```bash
# Run all tests
python tests/test_runner.py all

# Run specific categories
python tests/test_runner.py unit
python tests/test_runner.py integration
python tests/test_runner.py performance
python tests/test_runner.py edge-cases

# Run quick tests (exclude slow tests)
python tests/test_runner.py quick

# Run with coverage
python tests/test_runner.py coverage

# Full validation suite
python tests/test_runner.py validate
```

### Using Pytest Directly

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/unit/test_types.py

# Run tests with specific marker
pytest -m "unit and not slow"

# Run with verbose output
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Using Make

```bash
# Run all tests
make test

# Run specific categories
make test-unit
make test-integration
make test-performance
make test-edge

# Full validation
make validate
```

## Test Coverage

The test suite aims for comprehensive coverage:

### Core Components
- ✅ **Type System**: 100% coverage of Action, Observation, State classes
- ✅ **Python Executor**: Complete execution engine testing
- ✅ **Environment**: Full CodeAct environment API coverage
- ✅ **Transforms**: All transform types and compositions
- ✅ **MCP Integration**: Tool system and mock implementations

### Scenarios Covered
- ✅ **Basic Operations**: Code execution, state management, tool usage
- ✅ **Error Handling**: All exception types and recovery scenarios
- ✅ **Performance**: Benchmarks for critical operations
- ✅ **Edge Cases**: Boundary conditions and unusual inputs
- ✅ **Integration**: Full agent workflows and RL training

### Test Statistics
- **Total Tests**: 200+ individual test cases
- **Unit Tests**: 120+ tests covering individual components
- **Integration Tests**: 40+ tests covering full workflows
- **Performance Tests**: 25+ benchmarks and scalability tests
- **Edge Case Tests**: 60+ robustness and boundary tests

## Test Fixtures and Utilities

### Common Fixtures (`conftest.py`)
- `basic_env` - Standard CodeAct environment
- `mcp_env` - Environment with MCP tools
- `sample_actions` - Pre-built test actions
- `sample_transforms` - Transform test objects
- `temp_dir` - Temporary directory for file tests

### Test Utilities (`utils.py`)
- `TestTimer` - Performance timing context manager
- `CodeSequence` - Helper for multi-step code execution
- `PerformanceProfiler` - Performance measurement tools
- `create_math_test_cases()` - Mathematical test case generator
- `run_test_suite()` - Batch test execution helper

## Writing New Tests

### Test Naming Convention
- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`

### Example Test Structure
```python
class TestMyComponent:
    """Test the MyComponent class."""

    def test_basic_functionality(self):
        """Test basic component functionality."""
        # Arrange
        component = MyComponent()

        # Act
        result = component.do_something()

        # Assert
        assert result.success is True

    @pytest.mark.edge_case
    def test_error_conditions(self):
        """Test component error handling."""
        component = MyComponent()

        with pytest.raises(ValueError):
            component.do_invalid_operation()
```

### Best Practices
1. **Isolation**: Each test should be independent
2. **Clarity**: Test names should describe what they verify
3. **Coverage**: Test both success and failure paths
4. **Performance**: Use appropriate markers for slow tests
5. **Documentation**: Include docstrings explaining test purpose

## Continuous Integration

The test suite is designed for CI/CD integration:

```yaml
# Example CI configuration
- name: Run Tests
  run: |
    python tests/test_runner.py validate
```

The validation command runs:
1. Code quality checks (linting)
2. Unit tests
3. Integration tests
4. Edge case tests
5. Performance tests

## Test Data and Fixtures

### Temporary Files
Tests that need files use the `temp_dir` fixture or `temporary_files` context manager:

```python
def test_file_operations(temp_dir):
    file_path = os.path.join(temp_dir, "test.txt")
    # Test file operations...
```

### Mock Objects
The `utils.py` provides `MockTool` for testing tool integration:

```python
def test_tool_usage():
    mock_tool = MockTool(return_value="test_result")
    env = CodeActEnvironment(tools={'test_tool': mock_tool})
    # Test tool usage...
```

## Performance Benchmarks

Performance tests establish baselines:
- Environment reset: < 1ms average
- Simple code execution: < 1ms average
- Variable persistence: No degradation with state growth
- Transform application: < 0.1ms per transform
- Tool invocation: < 10ms average

## Troubleshooting Tests

### Common Issues
1. **Import Errors**: Ensure `src/` is in Python path
2. **Fixture Errors**: Check `conftest.py` for fixture definitions
3. **Marker Warnings**: Register custom markers in `pytest.ini`
4. **Timeout Issues**: Increase timeout for slow tests

### Debug Mode
Run tests with extra debugging:
```bash
pytest tests/ -v -s --tb=long
```

### Coverage Reports
Generate detailed coverage:
```bash
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html  # View in browser
```