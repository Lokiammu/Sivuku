#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
REPL Environment with LLM Integration.

This example demonstrates how to use the REPL environment with an actual LLM,
implementing the Recursive Language Model (RLM) paradigm:

1. LLM generates Python code to solve a task
2. Code is executed in the sandboxed REPL
3. LLM sees the output and generates more code
4. Process repeats until FINAL() is called

This is similar to the MIT RLM implementation but uses OpenEnv's repl_env.

Requirements:
    pip install transformers torch accelerate

Usage:
    python examples/repl_with_llm.py
"""
from __future__ import annotations

import re

from repl_env.server.repl_environment import REPLEnvironment
from repl_env.models import REPLAction

# System prompt for the RLM
RLM_SYSTEM_PROMPT = """You are an AI assistant with access to a Python REPL.

IMPORTANT RULES:
1. The variable 'context' already contains the data - DO NOT redefine it
2. Write Python code in ```python``` blocks
3. To submit your final answer, use: print(f'FINAL(your_answer_here)')

Example:
```python
# context is already defined, just use it
word_count = len(context.split())
print(f'FINAL({word_count})')
```

Available: context (the data), answer (dict), standard library modules.
"""


def extract_code_blocks(text: str) -> list[str]:
    """Extract Python code blocks from LLM response."""
    pattern = r"```python\s*(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return [m.strip() for m in matches if m.strip()]


def format_observation(obs, iteration: int) -> str:
    """Format observation for the LLM."""
    parts = []

    if obs.result.success:
        if obs.result.stdout:
            parts.append(f"Output: {obs.result.stdout.strip()}")
        parts.append("Code executed successfully.")
    else:
        # Show error clearly
        error = obs.result.stderr or obs.result.exception or "Unknown error"
        # Truncate long errors
        if len(error) > 300:
            error = error[:300] + "..."
        parts.append(f"ERROR: {error}")
        parts.append("Fix the error and try again. Remember: 'context' is already defined.")

    parts.append(f"Variables available: {obs.available_variables}")
    parts.append("Submit answer with: print(f'FINAL(answer)')")

    return "\n".join(parts)


def create_qwen_llm():
    """Create an LLM function using the smallest Qwen instruct model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )

    def llm_fn(messages: list[dict]) -> str:
        """Generate response using Qwen model."""
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    return llm_fn


def run_rlm_loop(
    llm_fn,
    context: str,
    task_prompt: str,
    max_iterations: int = 30,
    verbose: bool = True,
):
    """
    Run the RLM loop with an LLM and the REPL environment.

    Args:
        llm_fn: Function that takes messages and returns LLM response
        context: The context/data to process
        task_prompt: The task to accomplish
        max_iterations: Maximum REPL iterations
        verbose: Print progress

    Returns:
        The final answer string
    """
    env = REPLEnvironment(
        context=context,
        task_prompt=task_prompt,
        max_iterations=max_iterations,
    )

    try:
        obs = env.reset()

        # Build initial messages
        messages = [
            {"role": "system", "content": RLM_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""Task: {task_prompt}

Context ({obs.context_length} chars):
{obs.context_preview}{'...' if obs.context_length > 500 else ''}

Available variables: {obs.available_variables}

Generate Python code to solve this task. End with FINAL(answer) when done.""",
            },
        ]

        for iteration in range(1, max_iterations + 1):
            if verbose:
                print(f"\n--- Iteration {iteration} ---")

            # Get LLM response
            response = llm_fn(messages)

            if verbose:
                print(f"LLM Response:\n{response[:500]}{'...' if len(response) > 500 else ''}")

            # Extract and execute code blocks
            code_blocks = extract_code_blocks(response)

            if not code_blocks:
                # No code, ask LLM to provide code
                messages.append({"role": "assistant", "content": response})
                messages.append(
                    {
                        "role": "user",
                        "content": "Please provide Python code in ```python``` blocks to solve the task.",
                    }
                )
                continue

            # Execute each code block
            for code in code_blocks:
                if verbose:
                    print(f"\nExecuting:\n{code[:200]}{'...' if len(code) > 200 else ''}")

                obs = env.step(REPLAction(code=code))

                if verbose:
                    print(f"Success: {obs.result.success}")
                    if obs.result.stdout:
                        print(f"Output: {obs.result.stdout[:200]}")

                if obs.done:
                    final_answer = obs.metadata.get("final_answer")
                    if verbose:
                        print(f"\n=== Final Answer: {final_answer} ===")
                    return final_answer

            # Format observation for next iteration
            obs_text = format_observation(obs, iteration)
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": obs_text})

        # Max iterations reached
        if verbose:
            print("\nMax iterations reached without final answer")
        return None

    finally:
        env.close()


def main():
    """Run example with Qwen model."""
    print("=" * 60)
    print("REPL Environment with LLM Integration (Qwen)")
    print("=" * 60)

    # Create the LLM function
    llm_fn = create_qwen_llm()

    # Example task
    context = """
    The quick brown fox jumps over the lazy dog.
    This is a sample text for testing the REPL environment.
    It contains multiple sentences that we can analyze.
    The RLM paradigm allows models to process data programmatically.
    """

    task = "Count the total number of words in the context"

    print(f"\nTask: {task}")
    print(f"Context: {context[:100]}...")

    result = run_rlm_loop(
        llm_fn=llm_fn,
        context=context,
        task_prompt=task,
        max_iterations=10,
        verbose=True,
    )

    print(f"\n{'=' * 60}")
    print(f"Final Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
