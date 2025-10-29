from core.http_env_client import HTTPEnvClient
from core.client_types import StepResult
from .models import DiseaseAction, DiseaseObservation, DiseaseState

class DiseaseControlEnv(HTTPEnvClient[DiseaseAction, DiseaseObservation]):
    def _step_payload(self, action: DiseaseAction) -> dict:
        return action.__dict__

    def _parse_result(self, payload: dict) -> StepResult[DiseaseObservation]:
        obs = DiseaseObservation(**payload["observation"])
        return StepResult(observation=obs,
                          reward=payload["reward"],
                          done=payload["done"])

    def _parse_state(self, payload: dict) -> DiseaseState:
        return DiseaseState(**payload)
