# src/envs/disease_control_env/models.py
from dataclasses import dataclass
from core.env_server import Action, Observation, State

@dataclass
class DiseaseAction(Action):
    closures: float      # [0,1]
    vaccination: float   # [0,1]
    quarantine: float    # [0,1]
    spending: float      # [0,1]

@dataclass
class DiseaseObservation(Observation):
    S: float; H: float; I: float; Q: float; D: float
    step: int
    budget: float
    disease: str
    # live UI helpers
    last_event: str | None = None
    mu0: float = 0.0
    sigma: float = 0.0
    delta: float = 0.0
    nu: float = 0.0
    phi: float = 0.0

@dataclass
class DiseaseState(State):
    step_count: int = 0
    episode_return: float = 0.0
    budget: float = 0.0
    disease: str = "covid"
