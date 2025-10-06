#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Simple example demonstrating EnvTorch environment usage.

This shows the minimal steps to get started with code execution environments.
"""

from src import CodeAction, CodeExecutionEnvironment, CodingEnv, Transform


def basic_code_execution_example():
    """Basic example using CodeExecutionEnvironment."""
    print("=== Basic Code Execution Example ===")

    # Create basic code execution environment
    env = CodeExecutionEnvironment()

    print("Note: This example shows the interface but requires Docker to actually run")
    print("Environment created successfully!")

    # Create an action to calculate compound interest
    action = CodeAction(
        code="""
# Calculate compound interest
principal = 1000
rate = 0.05
time = 3

final_amount = principal * (1 + rate) ** time
interest_earned = final_amount - principal

print(f"Principal: ${principal}")
print(f"Rate: {rate*100}%")
print(f"Time: {time} years")
print(f"Final amount: ${final_amount:.2f}")
print(f"Interest earned: ${interest_earned:.2f}")

final_amount
"""
    )

    print(f"Created action with code length: {len(action.code)} characters")
    print()


def coding_environment_example():
    """Example using CodingEnv with safety and quality transforms."""
    print("=== Coding Environment Example ===")

    # Create coding environment with built-in transforms
    env = CodingEnv()

    print("CodingEnv created with safety and quality transforms!")
    print("This environment includes:")
    print("• Code safety checks")
    print("• Code quality analysis")
    print("• Composite transform system")

    # Example of safe code
    safe_action = CodeAction(
        code="""
# Safe mathematical calculation
import math

def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

# Calculate first 10 Fibonacci numbers
fib_sequence = [calculate_fibonacci(i) for i in range(10)]
print(f"First 10 Fibonacci numbers: {fib_sequence}")
fib_sequence
"""
    )

    print(f"Created safe action with code length: {len(safe_action.code)} characters")
    print()


def transform_system_example():
    """Example showing how to create custom transforms."""
    print("=== Transform System Example ===")

    # Example custom transform
    class RewardTransform(Transform):
        """Transform that adds rewards based on code execution results."""

        def __call__(self, observation):
            # This is just an example - actual implementation would need
            # a proper observation object with execution results
            print("Custom transform would analyze execution results here")
            print("and add rewards based on success criteria")
            return observation

    transform = RewardTransform()
    print("Created custom RewardTransform")

    print("Transform system allows:")
    print("• Chaining multiple transforms")
    print("• Adding rewards for RL training")
    print("• Custom observation processing")
    print("• Safety and quality checks")
    print()


if __name__ == "__main__":
    print("EnvTorch Environment Examples")
    print("=" * 40)
    print()

    basic_code_execution_example()
    coding_environment_example()
    transform_system_example()

    print("=" * 40)
    print("Examples complete! 🎉")
    print()
    print("Key takeaways:")
    print("• CodeAction(code='...') for arbitrary Python execution")
    print("• CodeExecutionEnvironment provides base functionality")
    print("• CodingEnv adds safety and quality transforms")
    print("• Transform system enables customization and RL training")
    print("• Docker integration provides sandboxed execution")
    print("=" * 40)
