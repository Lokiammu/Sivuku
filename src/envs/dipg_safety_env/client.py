from core.http_env_client import HTTPEnvClient, StepResult
from .models import DIPGAction, DIPGObservation, DIPGState

class DIPGSafetyEnv(HTTPEnvClient[DIPGAction, DIPGObservation]):
    def _step_payload(self, action: DIPGAction) -> dict:
        return {"llm_response": action.llm_response}

    def _parse_result(self, payload: dict) -> StepResult[DIPGObservation]:
        # --- ADD THESE DEBUG LINES ---
        print("--- DEBUG: Raw payload received by client ---")
        print(payload)
        print("-------------------------------------------")
        # -----------------------------        
        obs = DIPGObservation(**payload["observation"])
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> DIPGState:
        return DIPGState(**payload)