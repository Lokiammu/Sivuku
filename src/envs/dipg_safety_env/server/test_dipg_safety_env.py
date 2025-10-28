# src/envs/dipg_safety_env/server/test_dipg_safety_env.py

import pytest
from unittest.mock import patch
from .dipg_environment import DIPGEnvironment
from ..models import DIPGAction

@pytest.fixture
def mock_dataset_path():
    """ Creates a temporary dataset file and returns its path. """
    return "/Users/surfiniaburger/Desktop/OpenEnv/tests/envs/mock_dataset.jsonl"

@pytest.fixture
def env(mock_dataset_path):
    """ Provides a fresh environment instance using the mocked dataset path. """
    return DIPGEnvironment(dataset_path=mock_dataset_path)

def test_environment_initialization(env):
    """ Test that the environment initializes and loads the dataset. """
    assert env is not None
    assert len(env.dataset) == 2

def test_reset_changes_episode_deterministically(env):
    """ Test that reset provides different challenges deterministically. """
    # Mock random.choice to return the first item, then the second item.
    with patch('random.choice', side_effect=[env.dataset[0], env.dataset[1]]):
        obs1 = env.reset()
        obs2 = env.reset()
    assert obs1.question != obs2.question

