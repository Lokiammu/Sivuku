# src/envs/dipg_safety_env/server/test_dipg_safety_env.py

import pytest
import os
from unittest.mock import patch
from .dipg_environment import DIPGEnvironment
from ..models import DIPGAction

@pytest.fixture
def mock_dataset_path():
    """ Creates a temporary dataset file and returns its path. """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../tests/envs/mock_dataset.jsonl"))

@pytest.fixture
def env(mock_dataset_path):
    """ Provides a fresh environment instance using the mocked dataset path. """
    return DIPGEnvironment(dataset_path=mock_dataset_path)

def test_environment_initialization(env):
    """ Test that the environment initializes and loads the dataset. """
    assert env is not None
    assert len(env.dataset) == 2

