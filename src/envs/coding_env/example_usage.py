from envs.coding_env.env import CodingEnv
from envs.coding_env.models import CodeAction

env = CodingEnv(base_url="http://localhost:8080")
obs0 = env.reset()
result = env.step(CodeAction(code="print('hi')"))
print(result.observation.stdout.strip(), result.reward)
