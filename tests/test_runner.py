#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test runner script for EnvTorch test suite.

This script provides convenient ways to run different test categories
and generate reports.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description="Running command"):
    """Run a shell command and return the result."""
    print(f"{description}: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False

    print("✅ Success")
    if result.stdout:
        print(result.stdout)

    return True


def run_unit_tests():
    """Run only unit tests."""
    cmd = ["python", "-m", "pytest", "tests/unit/", "-v", "-m", "unit or not integration and not performance and not slow"]
    return run_command(cmd, "Running unit tests")


def run_integration_tests():
    """Run only integration tests."""
    cmd = ["python", "-m", "pytest", "tests/integration/", "-v", "-m", "integration"]
    return run_command(cmd, "Running integration tests")


def run_performance_tests():
    """Run performance tests."""
    cmd = ["python", "-m", "pytest", "tests/test_performance.py", "-v", "-m", "performance"]
    return run_command(cmd, "Running performance tests")


def run_edge_case_tests():
    """Run edge case tests."""
    cmd = ["python", "-m", "pytest", "tests/test_edge_cases.py", "-v", "-m", "edge_case"]
    return run_command(cmd, "Running edge case tests")


def run_all_tests():
    """Run all tests."""
    cmd = ["python", "-m", "pytest", "tests/", "-v"]
    return run_command(cmd, "Running all tests")


def run_quick_tests():
    """Run quick tests only (exclude slow tests)."""
    cmd = ["python", "-m", "pytest", "tests/", "-v", "-m", "not slow"]
    return run_command(cmd, "Running quick tests")


def run_tests_with_coverage():
    """Run tests with coverage reporting."""
    cmd = ["python", "-m", "pytest", "tests/", "--cov=src", "--cov-report=html", "--cov-report=term-missing"]
    return run_command(cmd, "Running tests with coverage")


def run_linting():
    """Run linting checks."""
    success = True

    # Run flake8
    cmd = ["flake8", "src/", "tests/"]
    if not run_command(cmd, "Running flake8 linting"):
        success = False

    # Run type checking if mypy is available
    try:
        subprocess.run(["mypy", "--version"], capture_output=True, check=True)
        cmd = ["mypy", "src/"]
        if not run_command(cmd, "Running mypy type checking"):
            success = False
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  mypy not available, skipping type checking")

    return success


def run_full_validation():
    """Run complete validation suite."""
    print("🚀 Starting full validation suite...\n")

    success = True

    # 1. Linting
    print("Step 1: Code Quality Checks")
    print("=" * 50)
    if not run_linting():
        success = False
    print()

    # 2. Unit tests
    print("Step 2: Unit Tests")
    print("=" * 50)
    if not run_unit_tests():
        success = False
    print()

    # 3. Integration tests
    print("Step 3: Integration Tests")
    print("=" * 50)
    if not run_integration_tests():
        success = False
    print()

    # 4. Edge cases
    print("Step 4: Edge Case Tests")
    print("=" * 50)
    if not run_edge_case_tests():
        success = False
    print()

    # 5. Performance tests
    print("Step 5: Performance Tests")
    print("=" * 50)
    if not run_performance_tests():
        success = False
    print()

    # Summary
    print("=" * 60)
    if success:
        print("🎉 All validation steps passed!")
        return True
    else:
        print("❌ Some validation steps failed")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="EnvTorch Test Runner")

    subparsers = parser.add_subparsers(dest='command', help='Test commands')

    # Add subcommands
    subparsers.add_parser('unit', help='Run unit tests only')
    subparsers.add_parser('integration', help='Run integration tests only')
    subparsers.add_parser('performance', help='Run performance tests only')
    subparsers.add_parser('edge-cases', help='Run edge case tests only')
    subparsers.add_parser('all', help='Run all tests')
    subparsers.add_parser('quick', help='Run quick tests (exclude slow tests)')
    subparsers.add_parser('coverage', help='Run tests with coverage reporting')
    subparsers.add_parser('lint', help='Run linting checks only')
    subparsers.add_parser('validate', help='Run full validation suite')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Change to project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # Execute command
    command_map = {
        'unit': run_unit_tests,
        'integration': run_integration_tests,
        'performance': run_performance_tests,
        'edge-cases': run_edge_case_tests,
        'all': run_all_tests,
        'quick': run_quick_tests,
        'coverage': run_tests_with_coverage,
        'lint': run_linting,
        'validate': run_full_validation,
    }

    success = command_map[args.command]()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()