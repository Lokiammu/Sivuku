# src/envs/dipg_safety_env/server/test_dipg_safety_env.py

import pytest
from unittest.mock import patch, mock_open
from .dipg_environment import DIPGEnvironment
from ..models import DIPGAction

# Mock data to be returned by the mocked 'open' call
MOCK_DATASET_CONTENT = """
{"messages": [{}, {"content": "Context A\\\\n\\\\nQuestion A"}, {}]}
{"messages": [{}, {"content": "Context B\\\\n\\\\nQuestion B"}, {}]}
"""

@pytest.fixture
def env():
    """ Provides a fresh instance of the environment with a mocked dataset for each test. """
    # Mock the file 'open' operation to make the test self-contained
    with patch("src.envs.dipg_safety_env.server.dipg_environment.open", mock_open(read_data=MOCK_DATASET_CONTENT)):
        # The environment will now use our MOCK_DATASET_CONTENT instead of the real file
        yield DIPGEnvironment()

def test_environment_initialization(env):
    """ Test that the environment initializes correctly and loads the mocked dataset. """
    assert env is not None
    assert len(env.dataset) == 2, "Mocked dataset should contain 2 items."

def test_reset_returns_valid_observation(env):
    """ Test that reset() returns a valid observation. """
    obs = env.reset()
    assert obs is not None
    assert isinstance(obs.context, str) and obs.context
    assert isinstance(obs.question, str) and obs.question
    assert env.state.step_count == 0

def test_step_returns_reward_and_done(env):
    """ Test that a step() call returns a numerical reward and a 'done' flag. """
    env.reset()
    result = env.step(DIPGAction(llm_response="Test"))
    assert result is not None
    assert isinstance(result.reward, (int, float))
    assert result.done is True

def test_reset_changes_episode_deterministically(env):
    """ Test that calling reset() provides different challenges deterministically. """
    # Mock random.choice to return the first item, then the second item.
    with patch('random.choice', side_effect=[env.dataset[0], env.dataset[1]]):
        obs1 = env.reset()
        obs2 = env.reset()
    
    # We can now be 100% certain that two different items were chosen.
    assert obs1.question != obs2.question, "Reset should provide a new, different challenge."