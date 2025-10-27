# src/envs/dipg_safety_env/server/test_dipg_safety_env.py

import pytest
from .dipg_environment import DIPGEnvironment
from ..models import DIPGAction

# This tells pytest that all tests in this file can use the 'env' fixture.
@pytest.fixture
def env():
    """ Provides a fresh instance of the environment for each test. """
    return DIPGEnvironment()

def test_environment_initialization(env):
    """ Test that the environment initializes correctly and loads the dataset. """
    assert env is not None
    assert len(env.dataset) > 0, "Dataset should be loaded and not empty."

def test_reset_returns_valid_observation(env):
    """ Test that reset() returns a valid observation with a context and question. """
    obs = env.reset()
    
    assert obs is not None
    assert isinstance(obs.context, str) and obs.context != "", "Context should be a non-empty string."
    assert isinstance(obs.question, str) and obs.question != "", "Question should be a non-empty string."
    assert env.state.step_count == 0, "Step count should be 0 after reset."

def test_step_returns_reward_and_done(env):
    """ Test that a step() call returns a numerical reward and a 'done' flag. """
    env.reset()
    
    # Create a dummy action (a simple string response)
    dummy_action = DIPGAction(llm_response="This is a test response.")
    
    result = env.step(dummy_action)
    
    assert result is not None
    assert isinstance(result.reward, (int, float)), "Reward should be a number."
    assert result.done is True, "The DIPG environment should always be 'done' after one step."

def test_reset_changes_episode(env):
    """ Test that calling reset() multiple times provides different challenges. """
    obs1 = env.reset()
    question1 = obs1.question
    
    # It's statistically very likely the next question will be different.
    obs2 = env.reset()
    question2 = obs2.question
    
    assert question1 != question2, "Reset should provide a new, different challenge."