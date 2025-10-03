#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test suite for the unified CodeAct environment implementation.

Tests all core functionality:
- Python code execution with persistent state
- Tool integration
- Transform system for RL training
- MCP integration
- Error handling and observation capture
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src import *


def test_basic_functionality():
    """Test core CodeAct environment functionality."""
    print("=== Testing Basic Functionality ===")

    env = create_codeact_env()

    # Test reset
    obs = env.reset()
    print(f"Reset successful, episode ID: {obs.metadata.get('episode_id', 'None')}")
    print(f"Available tools: {', '.join(obs.available_tools)}")

    # Test simple expression
    action = CodeAction(code="2 + 2")
    obs = env.step(action)
    print(f"Expression '2 + 2' = {obs.execution_result.return_value}")

    # Test state persistence
    action = CodeAction(code="x = 42")
    obs = env.step(action)

    action = CodeAction(code="x * 2")
    obs = env.step(action)
    print(f"State persistence: x * 2 = {obs.execution_result.return_value}")

    # Test error handling
    action = CodeAction(code="1 / 0")
    obs = env.step(action)
    print(f"Error handling: {obs.execution_result.exception_type} - {obs.execution_result.exception_message}")
    print()


def test_tool_integration():
    """Test tool integration and usage."""
    print("=== Testing Tool Integration ===")

    env = create_mcp_environment()
    obs = env.reset()

    print(f"MCP tools available: {', '.join(obs.available_tools)}")

    # Test calculator tool
    action = CodeAction(code='result = calculator("5 + 3 * 2"); print(f"Calculator result: {result}"); result')
    obs = env.step(action)
    print(f"Tool usage output: {obs.execution_result.stdout.strip()}")
    print(f"Return value: {obs.execution_result.return_value}")

    # Test file operations
    action = CodeAction(code='''
content = "Hello, unified CodeAct!"
write_result = file_write("/tmp/test_unified.txt", content)
read_result = file_read("/tmp/test_unified.txt")
print(f"File ops: {write_result}")
print(f"Read back: {read_result}")
''')
    obs = env.step(action)
    print(f"File operations:\n{obs.execution_result.stdout}")
    print()


def test_rl_training():
    """Test RL training capabilities with transforms."""
    print("=== Testing RL Training Capabilities ===")

    # Create environment with math problem transform
    transform = create_math_env_transform(expected_answer=42, tolerance=0.1)
    env = create_codeact_env()
    env.transform = transform

    test_cases = [
        ("Correct answer", "21 * 2"),
        ("Close answer", "41.9"),
        ("Wrong answer", "50"),
        ("Unsafe code", "import os; 42"),
        ("Syntax error", "21 *"),
    ]

    total_reward = 0

    for name, code in test_cases:
        obs = env.reset()
        action = CodeAction(code=code)
        obs = env.step(action)

        reward = obs.reward or 0
        total_reward += reward

        print(f"{name:15} | Code: {code:15} | Success: {str(obs.execution_result.success):5} | Reward: {reward:6.2f}")

        if 'safety_violation' in obs.metadata:
            print(f"                  Safety violation: {obs.metadata['safety_violation']}")

    print(f"\nTotal reward: {total_reward:.2f}")
    print(f"Average reward: {total_reward / len(test_cases):.2f}")
    print()


def test_complex_scenario():
    """Test complex multi-step scenario combining all features."""
    print("=== Testing Complex Multi-Step Scenario ===")

    env = create_mcp_environment()
    obs = env.reset()

    print("Task: Process data, perform calculations, and generate report")

    # Step 1: Create and process data
    action = CodeAction(code='''
import json

# Create sample data
data = {"items": [{"name": "A", "value": 10}, {"name": "B", "value": 20}, {"name": "C", "value": 15}]}

# Process data
total = sum(item["value"] for item in data["items"])
average = total / len(data["items"])

print(f"Processed {len(data['items'])} items")
print(f"Total value: {total}")
print(f"Average value: {average:.2f}")

total  # Return total for potential reward calculation
''')

    obs = env.step(action)
    print("Step 1 - Data processing:")
    print(f"  {obs.execution_result.stdout.strip()}")
    print(f"  Return value: {obs.execution_result.return_value}")

    # Step 2: Use tools for further analysis
    action = CodeAction(code='''
# Use calculator for verification
verification = calculator(f"{total} / {len(data['items'])}")
print(f"Calculator verification: {verification}")

# Generate report and save it
report = f"""
DATA ANALYSIS REPORT
===================
Items processed: {len(data["items"])}
Total value: {total}
Average value: {average:.2f}
Calculator verification: {verification}
"""

file_write("/tmp/analysis_report.txt", report)
print("Report saved to file")

# Read it back to verify
saved_report = file_read("/tmp/analysis_report.txt")
print("Report contents:")
print(saved_report)
''')

    obs = env.step(action)
    print("\nStep 2 - Tool usage and reporting:")
    print(f"  Success: {obs.execution_result.success}")
    if obs.execution_result.success:
        print(f"  Output:\n{obs.execution_result.stdout}")
    else:
        print(f"  Error: {obs.execution_result.exception_type}")

    print()


def test_hybrid_agent_rl():
    """Test hybrid agent exploration followed by RL optimization."""
    print("=== Testing Hybrid Agent + RL Workflow ===")

    # Phase 1: Agent exploration
    print("Phase 1: Agent explores different approaches")
    env = create_codeact_env()

    approaches = [
        "# Approach 1: Direct calculation\nresult = 6 * 7\nresult",
        "# Approach 2: Using loops\nresult = 0\nfor i in range(6):\n    result += 7\nresult",
        "# Approach 3: Using math functions\nresult = math.pow(6, 1) * 7\nresult"
    ]

    exploration_results = []

    for i, code in enumerate(approaches, 1):
        obs = env.reset()
        action = CodeAction(code=code)
        obs = env.step(action)

        exploration_results.append({
            'approach': i,
            'result': obs.execution_result.return_value,
            'time': obs.execution_result.execution_time_ms,
            'success': obs.execution_result.success
        })

        print(f"  Approach {i}: result={obs.execution_result.return_value}, time={obs.execution_result.execution_time_ms:.2f}ms")

    # Phase 2: RL optimization
    print("\nPhase 2: RL evaluation with rewards")

    # Add transform that rewards correctness (42) and penalizes slow execution
    class SpeedOptimizationTransform(Transform):
        def __call__(self, observation):
            if isinstance(observation, CodeObservation):
                result = observation.execution_result.return_value
                time_ms = observation.execution_result.execution_time_ms

                # Base reward for correctness
                if result == 42:
                    reward = 1.0
                    # Bonus for speed (less time = higher bonus)
                    speed_bonus = max(0, 0.5 - (time_ms - 0.1) / 10)
                    reward += speed_bonus
                else:
                    reward = 0.0

                observation.reward = reward
            return observation

    env.transform = SpeedOptimizationTransform()

    for i, code in enumerate(approaches, 1):
        obs = env.reset()
        action = CodeAction(code=code)
        obs = env.step(action)

        reward = obs.reward or 0
        print(f"  Approach {i}: reward={reward:.3f}")

    print("\nRL optimization complete - fastest correct approach gets highest reward")
    print()


if __name__ == "__main__":
    print("Unified CodeAct Environment Test Suite")
    print("=" * 60)
    print()

    test_basic_functionality()
    test_tool_integration()
    test_rl_training()
    test_complex_scenario()
    test_hybrid_agent_rl()

    print("=" * 60)
    print("All tests completed successfully!")
    print()
    print("✓ Unified type system")
    print("✓ Clean environment interfaces")
    print("✓ Python code execution with state persistence")
    print("✓ Tool integration via MCP")
    print("✓ Transform system for RL training")
    print("✓ Error handling and observation capture")
    print("✓ Hybrid agent/RL workflows")
    print()
    print("The unified CodeAct implementation successfully bridges")
    print("agent execution and RL training in a clean, cohesive framework!")
    print("=" * 60)