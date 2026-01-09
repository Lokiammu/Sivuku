# REPL Environment for OpenEnv

A Python REPL environment for training language models on code execution tasks, based on the [Recursive Language Models (RLM)](https://arxiv.org/abs/2512.24601) paradigm.

## Overview

The RLM paradigm allows language models to:
- Execute Python code in a sandboxed REPL environment
- Make recursive calls to themselves or other LMs via `llm_query()` / `llm_batch()`
- Handle near-infinite context by programmatically decomposing and exploring data
- Terminate with explicit `FINAL(answer)` or `answer = {"content": ..., "ready": True}` signals

## Features

- **Sandboxed Python Execution**: Safe code execution with restricted builtins
- **Context Loading**: Load large contexts that agents can explore programmatically
- **Multiple Finalization Patterns**:
  - RLM-style: `print('FINAL(answer)')` or `print('FINAL_VAR(var_name)')`
  - Prime Intellect style: `answer = {"content": "...", "ready": True}`
- **Iteration Limits**: Configurable maximum steps per episode
- **Reward Signals**: Customizable reward functions for RL training
- **Optional LLM Oracle**: Can enable `llm_query()` and `llm_batch()` for recursive calls

## Quick Start

```python
from repl_env import REPLEnv, REPLAction

# Start from Docker
env = REPLEnv.from_docker_image("repl-env:latest")

# Reset with context
result = env.reset(
    context="This is a large document with lots of text...",
    task_prompt="Find the word count"
)

# Execute code iteratively
result = env.execute("words = context.split()")
result = env.execute("count = len(words)")
result = env.execute("print(f'FINAL({count})')")

# Check result
print(f"Done: {result.done}")
print(f"Final Answer: {result.observation.metadata['final_answer']}")

env.close()
```

## Environment Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `REPL_CONTEXT` | Initial context to load | "" |
| `REPL_TASK_PROMPT` | Task description | "" |
| `REPL_MAX_ITERATIONS` | Max steps per episode | 30 |

## Action Space

```python
class REPLAction:
    code: str           # Python code to execute
    is_final: bool      # Whether this signals the final answer
    final_answer: str   # The final answer (if is_final=True)
```

## Observation Space

```python
class REPLObservation:
    result: CodeBlockResult    # Execution result (stdout, stderr, etc.)
    context_preview: str       # First 500 chars of context
    context_length: int        # Total context length
    available_variables: list  # Variables in namespace
    iteration: int             # Current iteration
    max_iterations: int        # Max iterations
    done: bool                 # Episode complete?
    reward: float              # Step reward
```

## Finalization Patterns

### Pattern 1: FINAL() in stdout
```python
result = env.execute("answer = 42")
result = env.execute("print(f'FINAL({answer})')")
# -> done=True, final_answer="42"
```

### Pattern 2: FINAL_VAR() for variable reference
```python
result = env.execute("my_result = 'The answer is 42'")
result = env.execute("print('FINAL_VAR(my_result)')")
# -> done=True, final_answer="The answer is 42"
```

### Pattern 3: Prime Intellect style answer dict
```python
result = env.execute("answer['content'] = '42'")
result = env.execute("answer['ready'] = True")
# -> done=True, final_answer="42"
```

## Running Locally

### Using UV
```bash
cd envs/repl_env
uv run --project . server
```

### Using Docker
```bash
docker build -t repl-env:latest -f server/Dockerfile .
docker run -p 8000:8000 repl-env:latest
```

### Testing
```bash
pytest tests/
```

## References

- [RLM Paper (arXiv:2512.24601)](https://arxiv.org/abs/2512.24601)
- [MIT RLM Implementation](https://github.com/MIT-OASYS-Lab/rlm)
- [Alex Zhang's RLM Blog](https://alexzhang13.github.io/blog/2025/rlm/)
- [Prime Intellect RLM Blog](https://www.primeintellect.ai/blog/rlm)
