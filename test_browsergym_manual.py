#!/usr/bin/env python3
"""
Manual test script for BrowserGym environment.
This tests the environment locally without Docker.

Usage:
    python test_browsergym_manual.py
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from envs.browsergym_env.server.browsergym_environment import BrowserGymEnvironment
from envs.browsergym_env.models import BrowserGymAction


def test_miniwob_local():
    """Test MiniWoB environment locally (requires browsergym installed)."""
    print("\n" + "=" * 60)
    print("TEST 1: MiniWoB Environment (Local)")
    print("=" * 60)

    try:
        # Create environment
        print("\n1. Creating MiniWoB environment (click-test)...")
        env = BrowserGymEnvironment(
            benchmark="miniwob",
            task_name="click-test",
            headless=True,
        )
        print("✅ Environment created successfully")

        # Test reset
        print("\n2. Testing reset()...")
        obs = env.reset()
        print(f"✅ Reset successful")
        print(f"   - Goal: {obs.goal[:100]}...")
        print(f"   - Text length: {len(obs.text)} chars")
        print(f"   - URL: {obs.url}")
        print(f"   - Done: {obs.done}")
        print(f"   - Reward: {obs.reward}")

        # Test step
        print("\n3. Testing step() with a simple action...")
        action = BrowserGymAction(action_str="click('button')")
        obs = env.step(action)
        print(f"✅ Step successful")
        print(f"   - Done: {obs.done}")
        print(f"   - Reward: {obs.reward}")
        print(f"   - Error: {obs.error or 'None'}")

        # Test state
        print("\n4. Testing state property...")
        state = env.state
        print(f"✅ State retrieved")
        print(f"   - Episode ID: {state.episode_id}")
        print(f"   - Steps: {state.step_count}")
        print(f"   - Benchmark: {state.benchmark}")
        print(f"   - Task: {state.task_name}")

        # Cleanup
        env.close()
        print("\n✅ Environment closed successfully")

        print("\n" + "=" * 60)
        print("TEST 1 PASSED: MiniWoB works locally!")
        print("=" * 60)
        return True

    except ImportError as e:
        print(f"\n⚠️  SKIPPED: BrowserGym not installed")
        print(f"   Install with: pip install browsergym browsergym-miniwob")
        print(f"   Error: {e}")
        return None
    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_models():
    """Test that models can be imported and instantiated."""
    print("\n" + "=" * 60)
    print("TEST 2: Models (Import & Instantiation)")
    print("=" * 60)

    try:
        from envs.browsergym_env.models import (
            BrowserGymAction,
            BrowserGymObservation,
            BrowserGymState,
        )

        print("\n1. Testing BrowserGymAction...")
        action = BrowserGymAction(action_str="click('button')")
        assert action.action_str == "click('button')"
        print("✅ BrowserGymAction works")

        print("\n2. Testing BrowserGymObservation...")
        obs = BrowserGymObservation(
            text="test text",
            url="http://example.com",
            goal="click the button",
            done=False,
            reward=0.0,
        )
        assert obs.text == "test text"
        assert obs.url == "http://example.com"
        print("✅ BrowserGymObservation works")

        print("\n3. Testing BrowserGymState...")
        state = BrowserGymState(
            episode_id="test-123",
            step_count=5,
            benchmark="miniwob",
            task_name="click-test",
        )
        assert state.benchmark == "miniwob"
        assert state.step_count == 5
        print("✅ BrowserGymState works")

        print("\n" + "=" * 60)
        print("TEST 2 PASSED: All models work!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_client_import():
    """Test that client can be imported."""
    print("\n" + "=" * 60)
    print("TEST 3: Client Import")
    print("=" * 60)

    try:
        from envs.browsergym_env.client import BrowserGymEnv

        print("\n1. Importing BrowserGymEnv...")
        print("✅ BrowserGymEnv imported successfully")

        print("\n2. Checking class methods...")
        assert hasattr(BrowserGymEnv, "reset")
        assert hasattr(BrowserGymEnv, "step")
        assert hasattr(BrowserGymEnv, "state")
        assert hasattr(BrowserGymEnv, "from_docker_image")
        print("✅ All required methods present")

        print("\n" + "=" * 60)
        print("TEST 3 PASSED: Client works!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all manual tests."""
    print("\n" + "=" * 60)
    print("BROWSERGYM ENVIRONMENT - MANUAL TESTS")
    print("=" * 60)

    results = {}

    # Test 1: Models (always works)
    results["models"] = test_models()

    # Test 2: Client Import (always works)
    results["client"] = test_client_import()

    # Test 3: MiniWoB Local (requires browsergym installed)
    results["miniwob"] = test_miniwob_local()

    # Summary
    print("\n\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, result in results.items():
        if result is True:
            status = "✅ PASSED"
        elif result is False:
            status = "❌ FAILED"
        else:
            status = "⚠️  SKIPPED"
        print(f"{test_name:20s}: {status}")

    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)

    print("\n" + "=" * 60)
    print(f"Total: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60)

    if failed > 0:
        print("\n❌ Some tests failed!")
        sys.exit(1)
    elif passed > 0:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n⚠️  All tests were skipped (install browsergym to run full tests)")
        sys.exit(0)


if __name__ == "__main__":
    main()
