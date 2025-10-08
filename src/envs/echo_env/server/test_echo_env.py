#!/usr/bin/env python3
"""
Test script for the Echo Environment.

This tests the EchoEnvironment directly (no HTTP server needed).
"""

import sys
from pathlib import Path

# Add src to path (go up 4 levels: test_echo_env.py -> server -> echo_env -> envs -> src)
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from envs.echo_env.models import EchoAction, EchoObservation
from envs.echo_env.server import EchoEnvironment


def test_echo_environment():
    """Test the EchoEnvironment directly."""
    print("=" * 60)
    print("Echo Environment Direct Test")
    print("=" * 60)
    print()

    # Create environment
    print("Creating EchoEnvironment...")
    env = EchoEnvironment()
    print("✓ Environment created\n")

    # Test reset
    print("Testing reset()...")
    obs = env.reset()
    print(f"  Type: {type(obs)}")
    print(f"  Message: {obs.echoed_message}")
    print(f"  Length: {obs.message_length}")
    print(f"  Reward: {obs.reward}")
    print(f"  Done: {obs.done}")
    assert isinstance(obs, EchoObservation)
    assert obs.echoed_message == "Echo environment ready!"
    print("✓ Reset works!\n")

    # Test state
    print("Testing state...")
    state = env.state
    print(f"  Episode ID: {state.episode_id}")
    print(f"  Step count: {state.step_count}")
    assert state.step_count == 0
    print("✓ State works!\n")

    # Test step
    print("Testing step() with 'Hello'...")
    action = EchoAction(message="Hello")
    obs = env.step(action)
    print(f"  Echoed: {obs.echoed_message}")
    print(f"  Length: {obs.message_length}")
    print(f"  Reward: {obs.reward}")
    assert obs.echoed_message == "Hello"
    assert obs.message_length == 5
    assert obs.reward == 0.5  # 5 * 0.1
    print("✓ Step works!\n")

    # Test another step
    print("Testing step() with longer message...")
    action = EchoAction(message="Testing the echo environment")
    obs = env.step(action)
    print(f"  Echoed: {obs.echoed_message}")
    print(f"  Length: {obs.message_length}")
    print(f"  Reward: {obs.reward}")
    assert obs.echoed_message == "Testing the echo environment"
    assert obs.message_length == 28
    assert abs(obs.reward - 2.8) < 0.01  # 28 * 0.1 (check with tolerance)
    print("✓ Step works!\n")

    # Check state updated
    print("Checking state after steps...")
    state = env.state
    print(f"  Step count: {state.step_count}")
    assert state.step_count == 2
    print("✓ State updated correctly!\n")

    print("=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Install FastAPI: pip install fastapi uvicorn")
    print("  2. Start the server: cd src && uvicorn envs.echo_env.server.app:app --reload")
    print("  3. Test with HTTP client")

    return True


if __name__ == "__main__":
    try:
        success = test_echo_environment()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
