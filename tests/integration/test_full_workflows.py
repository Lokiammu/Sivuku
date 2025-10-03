# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Integration tests for complete CodeAct workflows.

These tests verify that all components work together correctly
in realistic usage scenarios.
"""

import pytest
import tempfile
import os
from pathlib import Path

from src import (
    create_codeact_env,
    create_mcp_environment,
    CodeAction,
    MathProblemTransform,
    CodeSafetyTransform,
    CompositeTransform,
    create_math_env_transform,
)


class TestBasicAgentWorkflows:
    """Test basic agent execution workflows."""

    def test_simple_math_problem_solving(self):
        """Test agent solving a simple math problem."""
        env = create_codeact_env()
        obs = env.reset()

        # Agent solves quadratic equation: x^2 - 5x + 6 = 0
        code = '''
# Solve x^2 - 5x + 6 = 0 using quadratic formula
import math

a, b, c = 1, -5, 6
discriminant = b**2 - 4*a*c
sqrt_discriminant = math.sqrt(discriminant)

x1 = (-b + sqrt_discriminant) / (2*a)
x2 = (-b - sqrt_discriminant) / (2*a)

solutions = [x1, x2]
print(f"Solutions: x1={x1}, x2={x2}")
solutions
'''
        obs = env.step(CodeAction(code=code))

        assert obs.execution_result.success is True
        solutions = obs.execution_result.return_value
        assert abs(solutions[0] - 3) < 1e-10  # x = 3
        assert abs(solutions[1] - 2) < 1e-10  # x = 2

    def test_data_processing_workflow(self):
        """Test a complete data processing workflow."""
        env = create_codeact_env()
        obs = env.reset()

        # Step 1: Create and process data
        obs1 = env.step(CodeAction(code='''
# Create sample sales data
sales_data = [
    {"product": "A", "quantity": 100, "price": 10.0},
    {"product": "B", "quantity": 75, "price": 15.0},
    {"product": "C", "quantity": 50, "price": 20.0},
]

# Calculate total revenue
total_revenue = sum(item["quantity"] * item["price"] for item in sales_data)
print(f"Total revenue: ${total_revenue}")
'''))
        assert obs1.execution_result.success is True
        assert "Total revenue: $2125.0" in obs1.execution_result.stdout

        # Step 2: Find best performing product
        obs2 = env.step(CodeAction(code='''
# Find product with highest revenue
product_revenues = []
for item in sales_data:
    revenue = item["quantity"] * item["price"]
    product_revenues.append({
        "product": item["product"],
        "revenue": revenue
    })

best_product = max(product_revenues, key=lambda x: x["revenue"])
print(f"Best product: {best_product['product']} with ${best_product['revenue']}")
best_product
'''))
        assert obs2.execution_result.success is True
        best = obs2.execution_result.return_value
        assert best["product"] == "B"
        assert best["revenue"] == 1125.0

    def test_function_definition_and_reuse(self):
        """Test defining functions and reusing them across steps."""
        env = create_codeact_env()
        obs = env.reset()

        # Step 1: Define utility functions
        obs1 = env.step(CodeAction(code='''
def fibonacci(n):
    """Calculate fibonacci number iteratively."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def factorial(n):
    """Calculate factorial."""
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

print("Functions defined successfully")
'''))
        assert obs1.execution_result.success is True

        # Step 2: Use functions in calculations
        obs2 = env.step(CodeAction(code='''
# Calculate some values
fib_10 = fibonacci(10)
fact_5 = factorial(5)

results = {
    "fib_10": fib_10,
    "fact_5": fact_5,
    "combined": fib_10 + fact_5
}

print(f"Fibonacci(10) = {fib_10}")
print(f"Factorial(5) = {fact_5}")
results
'''))
        assert obs2.execution_result.success is True
        results = obs2.execution_result.return_value
        assert results["fib_10"] == 55
        assert results["fact_5"] == 120

    def test_error_recovery_workflow(self):
        """Test agent recovering from errors and continuing."""
        env = create_codeact_env()
        obs = env.reset()

        # Step 1: Code with error
        obs1 = env.step(CodeAction(code='''
# This will cause an error
undefined_variable + 42
'''))
        assert obs1.execution_result.success is False
        assert obs1.execution_result.exception_type == "NameError"

        # Step 2: Agent fixes the error and continues
        obs2 = env.step(CodeAction(code='''
# Fix the error by defining the variable
defined_variable = 10
result = defined_variable + 42
print(f"Fixed: {result}")
result
'''))
        assert obs2.execution_result.success is True
        assert obs2.execution_result.return_value == 52


class TestMCPIntegrationWorkflows:
    """Test workflows using MCP tools."""

    def test_file_processing_workflow(self, temp_dir):
        """Test complete file processing workflow."""
        env = create_mcp_environment()
        obs = env.reset()

        # Step 1: Create and write data to file
        test_file = os.path.join(temp_dir, "data.txt")
        obs1 = env.step(CodeAction(code=f'''
# Create test data
data_lines = [
    "Name,Age,Score",
    "Alice,25,95",
    "Bob,30,87",
    "Charlie,28,92"
]

# Write to file
content = "\\n".join(data_lines)
write_result = file_write("{test_file}", content)
print(f"File write: {{write_result}}")
'''))
        assert obs1.execution_result.success is True

        # Step 2: Read and process file
        obs2 = env.step(CodeAction(code=f'''
# Read file content
content = file_read("{test_file}")
lines = content.strip().split("\\n")

# Parse CSV-like data
header = lines[0].split(",")
data = []
for line in lines[1:]:
    values = line.split(",")
    record = dict(zip(header, values))
    # Convert numeric fields
    record["Age"] = int(record["Age"])
    record["Score"] = int(record["Score"])
    data.append(record)

print(f"Parsed {{len(data)}} records")
data
'''))
        assert obs2.execution_result.success is True
        parsed_data = obs2.execution_result.return_value
        assert len(parsed_data) == 3
        assert parsed_data[0]["Name"] == "Alice"

        # Step 3: Analyze data and save results
        results_file = os.path.join(temp_dir, "analysis.txt")
        obs3 = env.step(CodeAction(code=f'''
# Calculate statistics
total_score = sum(record["Score"] for record in data)
avg_score = total_score / len(data)
avg_age = sum(record["Age"] for record in data) / len(data)

# Find best performer
best_performer = max(data, key=lambda x: x["Score"])

# Create analysis report
report = f"""
ANALYSIS REPORT
===============
Total records: {{len(data)}}
Average age: {{avg_age:.1f}}
Average score: {{avg_score:.1f}}
Best performer: {{best_performer["Name"]}} ({{best_performer["Score"]}})
"""

# Save analysis
file_write("{results_file}", report)
print("Analysis completed and saved")
avg_score
'''))
        assert obs3.execution_result.success is True
        assert abs(obs3.execution_result.return_value - 91.33333333333333) < 1e-10

    def test_calculator_chain_workflow(self):
        """Test chaining calculator operations."""
        env = create_mcp_environment()
        obs = env.reset()

        workflow_code = '''
# Multi-step calculation using calculator tool
step1 = calculator("15 * 8")  # 120
step2 = calculator("100 / 4")  # 25
step3 = calculator(f"{step1} + {step2}")  # 145
step4 = calculator(f"{step3} - 20")  # 125

results = {
    "step1": step1,
    "step2": step2,
    "step3": step3,
    "step4": step4
}

print(f"Calculation chain: {step1} + {step2} - 20 = {step4}")
results
'''
        obs = env.step(CodeAction(code=workflow_code))

        assert obs.execution_result.success is True
        results = obs.execution_result.return_value
        assert results["step1"] == 120
        assert results["step2"] == 25
        assert results["step3"] == 145
        assert results["step4"] == 125

    def test_mixed_tools_workflow(self):
        """Test workflow using multiple types of tools."""
        env = create_mcp_environment()
        obs = env.reset()

        workflow_code = '''
# Use multiple tools in a single workflow
import json

# Generate some data using calculator
values = []
for i in range(1, 6):
    # Use calculator for computation
    result = calculator(f"{i} * {i} + 10")
    values.append({"index": i, "value": result})

# Convert to JSON
json_data = json.dumps(values, indent=2)

# Search for something (mock)
search_result = web_search("mathematical sequences")

# Prepare final report
report = {
    "computed_values": values,
    "json_representation": json_data,
    "search_performed": "mathematical sequences" in search_result
}

print(f"Generated {len(values)} computed values")
report
'''
        obs = env.step(CodeAction(code=workflow_code))

        assert obs.execution_result.success is True
        report = obs.execution_result.return_value
        assert len(report["computed_values"]) == 5
        assert report["computed_values"][0]["value"] == 11  # 1*1 + 10
        assert report["search_performed"] is True


class TestRLTrainingWorkflows:
    """Test RL training integration workflows."""

    def test_math_problem_rl_training(self):
        """Test RL training on math problems."""
        # Create environment with math reward
        transform = create_math_env_transform(expected_answer=42, tolerance=0.1)
        env = create_codeact_env()
        env.transform = transform

        training_episodes = [
            "21 * 2",  # Correct
            "40 + 2",  # Correct
            "50 - 8",  # Correct
            "100 / 2", # Incorrect (50, not 42)
            "import os; 42",  # Correct but unsafe
        ]

        results = []
        for code in training_episodes:
            obs = env.reset()
            obs = env.step(CodeAction(code=code))
            results.append({
                "code": code,
                "success": obs.execution_result.success,
                "result": obs.execution_result.return_value,
                "reward": obs.reward
            })

        # Check rewards
        assert results[0]["reward"] > 1.0  # Correct + quality bonus
        assert results[1]["reward"] > 1.0  # Correct + quality bonus
        assert results[2]["reward"] > 1.0  # Correct + quality bonus
        assert results[3]["reward"] < 1.0  # Wrong answer
        assert results[4]["reward"] < 0    # Unsafe code penalty

    def test_composite_transform_workflow(self):
        """Test workflow with multiple transforms."""
        transforms = CompositeTransform([
            MathProblemTransform(expected_answer=100),
            CodeSafetyTransform(penalty=-2.0),
        ])

        env = create_codeact_env()
        env.transform = transforms

        test_cases = [
            {"code": "50 + 50", "expected_reward_sign": "+"},  # Correct, safe
            {"code": "import os\n100", "expected_reward_sign": "-"},  # Correct but unsafe
            {"code": "99", "expected_reward_sign": "0"},  # Wrong answer, safe
        ]

        for case in test_cases:
            obs = env.reset()
            obs = env.step(CodeAction(code=case["code"]))

            if case["expected_reward_sign"] == "+":
                assert obs.reward > 0
            elif case["expected_reward_sign"] == "-":
                assert obs.reward < 0
            else:
                assert obs.reward <= 0

    def test_progressive_learning_simulation(self):
        """Test simulated progressive learning scenario."""
        env = create_codeact_env()
        env.transform = MathProblemTransform(expected_answer=24)  # 4! = 24

        # Simulate agent learning to compute factorial
        learning_attempts = [
            "4 * 3 * 2 * 1",  # Direct calculation
            "factorial(4)",    # Undefined function (should fail)
            '''
def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

factorial(4)
''',  # Correct implementation
        ]

        rewards = []
        for code in learning_attempts:
            obs = env.reset()
            obs = env.step(CodeAction(code=code))
            rewards.append(obs.reward)

        # Should see improvement over time
        assert rewards[0] > 0  # Correct direct calculation
        assert rewards[1] < 0  # Error penalty
        assert rewards[2] > rewards[1]  # Learning improvement


class TestComplexScenarios:
    """Test complex, realistic usage scenarios."""

    @pytest.mark.slow
    def test_data_science_pipeline(self):
        """Test a complete data science-like pipeline."""
        env = create_mcp_environment()
        obs = env.reset()

        # Multi-step data science workflow
        steps = [
            # Data generation
            '''
import json
import random

# Generate sample dataset
random.seed(42)
dataset = []
for i in range(100):
    dataset.append({
        "id": i,
        "feature1": random.uniform(0, 100),
        "feature2": random.uniform(-50, 50),
        "target": random.choice([0, 1])
    })

print(f"Generated dataset with {len(dataset)} samples")
len(dataset)
''',
            # Data analysis
            '''
# Basic statistics
feature1_mean = sum(d["feature1"] for d in dataset) / len(dataset)
feature2_mean = sum(d["feature2"] for d in dataset) / len(dataset)
target_distribution = sum(d["target"] for d in dataset) / len(dataset)

stats = {
    "feature1_mean": feature1_mean,
    "feature2_mean": feature2_mean,
    "positive_rate": target_distribution
}

print(f"Statistics computed: {stats}")
stats
''',
            # Simple model (linear threshold)
            '''
# Simple threshold model
def simple_model(feature1, feature2):
    # Simple decision boundary
    score = feature1 * 0.4 + feature2 * 0.6
    return 1 if score > 20 else 0

# Evaluate model
correct = 0
for sample in dataset:
    prediction = simple_model(sample["feature1"], sample["feature2"])
    if prediction == sample["target"]:
        correct += 1

accuracy = correct / len(dataset)
print(f"Model accuracy: {accuracy:.3f}")
accuracy
''',
        ]

        results = []
        for i, code in enumerate(steps):
            obs = env.step(CodeAction(code=code))
            assert obs.execution_result.success is True
            results.append(obs.execution_result.return_value)
            print(f"Step {i+1} completed successfully")

        assert results[0] == 100  # Dataset size
        assert isinstance(results[1], dict)  # Statistics
        assert 0 <= results[2] <= 1  # Accuracy

    @pytest.mark.integration
    def test_agent_tool_collaboration(self, temp_dir):
        """Test agent using tools collaboratively to solve complex problem."""
        env = create_mcp_environment()
        obs = env.reset()

        # Problem: Process sales data, compute metrics, and generate report
        sales_file = os.path.join(temp_dir, "sales.csv")
        report_file = os.path.join(temp_dir, "report.txt")

        collaboration_code = f'''
# Step 1: Create sales data
sales_data = """Product,Month,Sales,Price
Widget A,Jan,100,10.50
Widget A,Feb,120,10.50
Widget B,Jan,80,15.00
Widget B,Feb,95,15.00
Widget C,Jan,60,20.00
Widget C,Feb,70,20.00"""

file_write("{sales_file}", sales_data)

# Step 2: Read and parse data
content = file_read("{sales_file}")
lines = content.strip().split("\\n")
header = lines[0].split(",")

parsed_data = []
for line in lines[1:]:
    values = line.split(",")
    record = {{
        "product": values[0],
        "month": values[1],
        "sales": int(values[2]),
        "price": float(values[3])
    }}
    parsed_data.append(record)

# Step 3: Compute metrics using calculator
total_revenue = 0
for record in parsed_data:
    revenue = calculator(f"{{record['sales']}} * {{record['price']}}")
    record["revenue"] = revenue
    total_revenue += revenue

# Step 4: Generate insights
product_totals = {{}}
for record in parsed_data:
    product = record["product"]
    if product not in product_totals:
        product_totals[product] = 0
    product_totals[product] += record["revenue"]

best_product = max(product_totals.items(), key=lambda x: x[1])

# Step 5: Create report
report_content = f"""SALES ANALYSIS REPORT
=====================

Total Revenue: ${{total_revenue:.2f}}
Number of Records: {{len(parsed_data)}}

Product Performance:
"""

for product, revenue in sorted(product_totals.items()):
    report_content += f"- {{product}}: ${{revenue:.2f}}\\n"

report_content += f"\\nBest Performing Product: {{best_product[0]}} (${{best_product[1]:.2f}})"

# Save report
file_write("{report_file}", report_content)

# Verify report was saved
saved_report = file_read("{report_file}")
print("Report generated and verified!")

# Return summary
{{
    "total_revenue": total_revenue,
    "best_product": best_product[0],
    "records_processed": len(parsed_data),
    "report_saved": len(saved_report) > 0
}}
'''

        obs = env.step(CodeAction(code=collaboration_code))

        assert obs.execution_result.success is True
        summary = obs.execution_result.return_value
        assert summary["total_revenue"] > 0
        assert summary["best_product"] in ["Widget A", "Widget B", "Widget C"]
        assert summary["records_processed"] == 6
        assert summary["report_saved"] is True