from envs.coding_env.models import CodeAction

from .coding_env_client import CodingEnv

env = CodingEnv.from_docker_image(image="coding_env:latest")
reset_result = env.reset()
print(f"Initial observation: {reset_result.observation}")

step_result = env.step(CodeAction(code="print('hi')"))
print(step_result.observation.stdout.strip(), step_result.reward)
