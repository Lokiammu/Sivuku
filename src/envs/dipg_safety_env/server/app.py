from core.env_server import create_fastapi_app
from ..models import DIPGAction, DIPGObservation
from .dipg_environment import DIPGEnvironment

env = DIPGEnvironment()
app = create_fastapi_app(env, DIPGAction, DIPGObservation)