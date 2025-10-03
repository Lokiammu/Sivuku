#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Simple example demonstrating CodeAct environment usage.

This shows the minimal steps to get started with CodeAct for both
agent execution and RL training scenarios.
"""

from src import *


def basic_agent_example():
    """Basic agent using CodeAct for problem solving."""
    print("=== Basic Agent Example ===")

    # Create environment with standard tools
    env = create_codeact_env()
    obs = env.reset()

    # Agent executes code to solve a problem
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

    obs = env.step(action)
    print("Agent calculation:")
    print(obs.execution_result.stdout)
    print(f"Result: ${obs.execution_result.return_value:.2f}")
    print()


def rl_training_example():
    """RL training example with rewards."""
    print("=== RL Training Example ===")

    # Create environment that rewards solving x^2 + 3x + 2 = 0
    # Correct answers are x = -1 and x = -2
    transform = create_math_env_transform(expected_answer=[-2, -1])

    # Custom transform for quadratic solutions
    class QuadraticTransform(Transform):
        def __call__(self, observation):
            if isinstance(observation, CodeObservation):
                result = observation.execution_result.return_value
                # Accept either root as correct
                if result in [-1, -2] or (
                    isinstance(result, (list, tuple)) and set(result) == {-1, -2}
                ):
                    observation.reward = 1.0
                else:
                    observation.reward = 0.0
            return observation

    env = create_codeact_env()
    env.transform = QuadraticTransform()

    # Train on different solution approaches
    approaches = [
        "# Factoring: (x+1)(x+2) = 0\nx = -1  # First root",
        "# Factoring: (x+1)(x+2) = 0\nx = -2  # Second root",
        "# Quadratic formula\nimport math\na, b, c = 1, 3, 2\ndiscriminant = b**2 - 4*a*c\nx1 = (-b + math.sqrt(discriminant)) / (2*a)\nx2 = (-b - math.sqrt(discriminant)) / (2*a)\n[x1, x2]",
        "# Wrong answer\nx = 0",
    ]

    for i, code in enumerate(approaches, 1):
        obs = env.reset()
        action = CodeAction(code=code)
        obs = env.step(action)

        reward = obs.reward or 0
        print(f"Approach {i}: reward = {reward:.1f}")

    print()


def mcp_agent_example():
    """Agent using MCP tools for enhanced capabilities."""
    print("=== MCP Agent Example ===")

    # Create environment with MCP tools
    env = create_mcp_environment()
    obs = env.reset()

    print(f"Available MCP tools: {', '.join(obs.available_tools)}")

    # Agent uses tools to analyze and process data
    action = CodeAction(
        code="""
# Create and analyze sales data
import json

sales_data = [
    {"month": "Jan", "revenue": 10000},
    {"month": "Feb", "revenue": 12000},
    {"month": "Mar", "revenue": 9000}
]

# Calculate metrics
total_revenue = sum(item["revenue"] for item in sales_data)
avg_revenue = total_revenue / len(sales_data)
best_month = max(sales_data, key=lambda x: x["revenue"])

# Create report
report = f'''
SALES ANALYSIS
==============
Total Revenue: ${total_revenue:,}
Average Monthly Revenue: ${avg_revenue:,.2f}
Best Month: {best_month["month"]} (${best_month["revenue"]:,})

Detailed Data:
{json.dumps(sales_data, indent=2)}
'''

# Save using MCP file tool
file_write("/tmp/sales_analysis.txt", report)
print("Sales analysis complete!")
print("Report saved to /tmp/sales_analysis.txt")

# Verify by reading back
saved_content = file_read("/tmp/sales_analysis.txt")
print("\\nReport contents:")
print(saved_content)
"""
    )

    obs = env.step(action)
    if obs.execution_result.success:
        print("MCP tool usage:")
        print(obs.execution_result.stdout)
    else:
        print(f"Error: {obs.execution_result.exception_message}")

    print()


if __name__ == "__main__":
    print("CodeAct Environment Examples")
    print("=" * 40)
    print()

    basic_agent_example()
    rl_training_example()
    mcp_agent_example()

    print("=" * 40)
    print("Examples complete! 🎉")
    print()
    print("Key takeaways:")
    print("• CodeAction(code='...') for arbitrary Python execution")
    print("• State persists across steps within episodes")
    print("• Tools available as regular Python objects")
    print("• Transforms enable RL training with rewards")
    print("• MCP integration provides external capabilities")
    print("=" * 40)
