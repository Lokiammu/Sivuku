# SUMO-RL Integration Plan for OpenEnv

**Date**: 2025-10-17
**Status**: Design Phase
**Complexity**: High (Docker + SUMO system dependencies)

---

## 🤔 ULTRATHINK ANALYSIS

### What is SUMO-RL?

**SUMO-RL** is a Reinforcement Learning environment for **Traffic Signal Control** using SUMO (Simulation of Urban MObility).

- **Use Case**: Train RL agents to optimize traffic light timing to minimize vehicle delays
- **Main Class**: `SumoEnvironment` from `sumo_rl.environment.env`
- **APIs**: Supports both Gymnasium (single-agent) and PettingZoo (multi-agent)
- **Repository**: https://github.com/LucasAlegre/sumo-rl
- **Version**: 1.4.5

### How SUMO-RL Works

1. **SUMO Simulator**: Microscopic traffic simulation
2. **Network Files**: `.net.xml` (road network) + `.rou.xml` (vehicle routes)
3. **Traffic Signals**: RL agent controls when lights change phases
4. **Observation**: Lane densities, queues, current phase, min_green flag
5. **Action**: Select next green phase (discrete action space)
6. **Reward**: Change in cumulative vehicle delay (default)

### Example Usage

```python
import gymnasium as gym
import sumo_rl

env = gym.make('sumo-rl-v0',
                net_file='nets/single-intersection.net.xml',
                route_file='nets/single-intersection.rou.xml',
                use_gui=False,
                num_seconds=100000)

obs, info = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

---

## 🎯 Integration Strategy

### Follow Atari Pattern

Like Atari, we'll create:
1. **models.py** - Data models
2. **server/sumo_environment.py** - Environment wrapper
3. **server/app.py** - FastAPI server
4. **server/Dockerfile** - Container with SUMO
5. **client.py** - HTTP client

### Key Differences from Atari

| Aspect | Atari | SUMO-RL |
|--------|-------|---------|
| **External Dependency** | ALE (pip installable) | SUMO (system package) |
| **Configuration** | Game name (simple) | Network + route files (complex) |
| **Observation** | Image pixels | Traffic metrics (vectors) |
| **Action** | Joystick actions | Traffic signal phases |
| **Docker Complexity** | Simple | High (need SUMO system install) |
| **File Dependencies** | None (ROMs bundled) | Network/route XML files required |

---

## 📋 Technical Design

### 1. Data Models (`models.py`)

```python
from dataclasses import dataclass
from typing import List, Optional
from core.env_server import Action, Observation, State

@dataclass
class SumoAction(Action):
    """Action for SUMO environment - select next green phase."""
    phase_id: int  # Which green phase to activate next
    ts_id: str = "0"  # Traffic signal ID (for multi-agent support later)

@dataclass
class SumoObservation(Observation):
    """Observation from SUMO environment."""
    observation: List[float]  # Full observation vector
    observation_shape: List[int]  # Shape for reshaping

    # Observation components (for interpretability)
    current_phase: Optional[int] = None
    min_green_passed: Optional[bool] = None
    lane_densities: Optional[List[float]] = None
    lane_queues: Optional[List[float]] = None

    # Metadata
    action_mask: Optional[List[int]] = None  # Legal actions
    sim_time: float = 0.0  # Current simulation time

    done: bool = False
    reward: Optional[float] = None

@dataclass
class SumoState(State):
    """State of SUMO environment."""
    episode_id: str = ""
    step_count: int = 0

    # SUMO configuration
    net_file: str = ""
    route_file: str = ""
    num_seconds: int = 20000
    delta_time: int = 5
    yellow_time: int = 2
    min_green: int = 5
    max_green: int = 50

    # Runtime state
    sim_time: float = 0.0
    total_vehicles: int = 0
    total_waiting_time: float = 0.0
```

### 2. Environment Wrapper (`server/sumo_environment.py`)

```python
import uuid
from typing import Any, Dict, Literal, Optional
from core.env_server import Action, Environment, Observation
from ..models import SumoAction, SumoObservation, SumoState

import os
os.environ.setdefault('SUMO_HOME', '/usr/share/sumo')

from sumo_rl import SumoEnvironment as BaseSumoEnv

class SumoEnvironment(Environment):
    """
    SUMO-RL Environment wrapper for OpenEnv.

    Wraps the SUMO traffic signal control environment for single-agent RL.

    Args:
        net_file: Path to SUMO network file (.net.xml)
        route_file: Path to SUMO route file (.rou.xml)
        num_seconds: Simulation duration in seconds
        delta_time: Seconds between actions
        yellow_time: Yellow phase duration
        min_green: Minimum green time
        max_green: Maximum green time
        reward_fn: Reward function name
    """

    def __init__(
        self,
        net_file: str,
        route_file: str,
        num_seconds: int = 20000,
        delta_time: int = 5,
        yellow_time: int = 2,
        min_green: int = 5,
        max_green: int = 50,
        reward_fn: str = "diff-waiting-time",
    ):
        super().__init__()

        # Store config
        self.net_file = net_file
        self.route_file = route_file
        self.num_seconds = num_seconds
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.reward_fn = reward_fn

        # Create SUMO environment (single-agent mode)
        self.env = BaseSumoEnv(
            net_file=net_file,
            route_file=route_file,
            use_gui=False,  # No GUI in Docker
            single_agent=True,  # Single-agent for OpenEnv
            num_seconds=num_seconds,
            delta_time=delta_time,
            yellow_time=yellow_time,
            min_green=min_green,
            max_green=max_green,
            reward_fn=reward_fn,
            sumo_warnings=False,
        )

        # Initialize state
        self._state = SumoState(
            net_file=net_file,
            route_file=route_file,
            num_seconds=num_seconds,
            delta_time=delta_time,
            yellow_time=yellow_time,
            min_green=min_green,
            max_green=max_green,
        )

        self._last_obs = None
        self._last_info = None

    def reset(self) -> Observation:
        """Reset the environment."""
        # Reset SUMO
        obs, info = self.env.reset()

        # Update state
        self._state.episode_id = str(uuid.uuid4())
        self._state.step_count = 0
        self._state.sim_time = 0.0

        # Store for later
        self._last_obs = obs
        self._last_info = info

        return self._make_observation(obs, 0.0, False, info)

    def step(self, action: Action) -> Observation:
        """Execute action."""
        if not isinstance(action, SumoAction):
            raise ValueError(f"Expected SumoAction, got {type(action)}")

        # Validate action
        if action.phase_id < 0 or action.phase_id >= self.env.action_space.n:
            raise ValueError(
                f"Invalid phase_id: {action.phase_id}. "
                f"Valid range: [0, {self.env.action_space.n - 1}]"
            )

        # Execute in SUMO
        obs, reward, terminated, truncated, info = self.env.step(action.phase_id)
        done = terminated or truncated

        # Update state
        self._state.step_count += 1
        self._state.sim_time = info.get('step', 0.0)
        self._state.total_vehicles = info.get('system_total_running', 0)
        self._state.total_waiting_time = info.get('system_total_waiting_time', 0.0)

        # Store for later
        self._last_obs = obs
        self._last_info = info

        return self._make_observation(obs, reward, done, info)

    @property
    def state(self) -> SumoState:
        """Get current state."""
        return self._state

    def _make_observation(
        self,
        obs: Any,
        reward: float,
        done: bool,
        info: Dict
    ) -> SumoObservation:
        """Create SumoObservation from SUMO env output."""
        # Convert observation to list
        if hasattr(obs, 'tolist'):
            obs_list = obs.tolist()
        else:
            obs_list = list(obs)

        # Get action mask (all actions valid in SUMO-RL)
        action_mask = list(range(self.env.action_space.n))

        # Create observation
        return SumoObservation(
            observation=obs_list,
            observation_shape=[len(obs_list)],
            action_mask=action_mask,
            sim_time=info.get('step', 0.0),
            done=done,
            reward=reward,
            metadata={
                "num_green_phases": self.env.action_space.n,
                "system_info": {
                    k: v for k, v in info.items() if k.startswith('system_')
                },
            },
        )
```

### 3. FastAPI Server (`server/app.py`)

```python
import os
from core.env_server import create_fastapi_app
from ..models import SumoAction, SumoObservation
from .sumo_environment import SumoEnvironment

# Get configuration from environment
net_file = os.getenv("SUMO_NET_FILE", "/app/nets/single-intersection.net.xml")
route_file = os.getenv("SUMO_ROUTE_FILE", "/app/nets/single-intersection.rou.xml")
num_seconds = int(os.getenv("SUMO_NUM_SECONDS", "20000"))
delta_time = int(os.getenv("SUMO_DELTA_TIME", "5"))
yellow_time = int(os.getenv("SUMO_YELLOW_TIME", "2"))
min_green = int(os.getenv("SUMO_MIN_GREEN", "5"))
max_green = int(os.getenv("SUMO_MAX_GREEN", "50"))
reward_fn = os.getenv("SUMO_REWARD_FN", "diff-waiting-time")

# Create environment
env = SumoEnvironment(
    net_file=net_file,
    route_file=route_file,
    num_seconds=num_seconds,
    delta_time=delta_time,
    yellow_time=yellow_time,
    min_green=min_green,
    max_green=max_green,
    reward_fn=reward_fn,
)

# Create FastAPI app
app = create_fastapi_app(env, SumoAction, SumoObservation)
```

### 4. Dockerfile (`server/Dockerfile`)

```dockerfile
# Configurable base image
ARG BASE_IMAGE=envtorch-base:latest
FROM ${BASE_IMAGE}

# Install SUMO
# SUMO is a microscopic traffic simulation package
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:sumo/stable \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        sumo \
        sumo-tools \
    && rm -rf /var/lib/apt/lists/*

# Set SUMO_HOME
ENV SUMO_HOME=/usr/share/sumo

# Install SUMO-RL and dependencies
RUN pip install --no-cache-dir \
    gymnasium>=0.28 \
    pettingzoo>=1.24.3 \
    numpy>=1.24.0 \
    pandas>=2.0.0 \
    sumolib>=1.14.0 \
    traci>=1.14.0 \
    sumo-rl>=1.4.5

# Copy OpenEnv core
COPY src/core/ /app/src/core/

# Copy SUMO-RL environment code
COPY src/envs/sumo_rl_env/ /app/src/envs/sumo_rl_env/

# Copy example networks
# We'll bundle a simple single-intersection example
COPY sumo-rl/sumo_rl/nets/single-intersection/ /app/nets/

# Environment variables (can be overridden at runtime)
ENV SUMO_NET_FILE=/app/nets/single-intersection.net.xml
ENV SUMO_ROUTE_FILE=/app/nets/single-intersection.rou.xml
ENV SUMO_NUM_SECONDS=20000
ENV SUMO_DELTA_TIME=5
ENV SUMO_YELLOW_TIME=2
ENV SUMO_MIN_GREEN=5
ENV SUMO_MAX_GREEN=50
ENV SUMO_REWARD_FN=diff-waiting-time

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server
CMD ["uvicorn", "envs.sumo_rl_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 5. HTTP Client (`client.py`)

```python
from typing import Any, Dict
from core.http_env_client import HTTPEnvClient
from core.types import StepResult
from .models import SumoAction, SumoObservation, SumoState

class SumoRLEnv(HTTPEnvClient[SumoAction, SumoObservation]):
    """
    HTTP client for SUMO-RL environment.

    Example:
        >>> env = SumoRLEnv.from_docker_image("sumo-rl-env:latest")
        >>> result = env.reset()
        >>> result = env.step(SumoAction(phase_id=1))
        >>> print(f"Reward: {result.reward}, Done: {result.done}")
        >>> env.close()
    """

    def _step_payload(self, action: SumoAction) -> Dict[str, Any]:
        """Convert action to JSON payload."""
        return {
            "phase_id": action.phase_id,
            "ts_id": action.ts_id,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[SumoObservation]:
        """Parse step result from JSON."""
        obs_data = payload.get("observation", {})

        observation = SumoObservation(
            observation=obs_data.get("observation", []),
            observation_shape=obs_data.get("observation_shape", []),
            current_phase=obs_data.get("current_phase"),
            min_green_passed=obs_data.get("min_green_passed"),
            lane_densities=obs_data.get("lane_densities"),
            lane_queues=obs_data.get("lane_queues"),
            action_mask=obs_data.get("action_mask", []),
            sim_time=obs_data.get("sim_time", 0.0),
            done=obs_data.get("done", False),
            reward=obs_data.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> SumoState:
        """Parse state from JSON."""
        return SumoState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            net_file=payload.get("net_file", ""),
            route_file=payload.get("route_file", ""),
            num_seconds=payload.get("num_seconds", 20000),
            delta_time=payload.get("delta_time", 5),
            yellow_time=payload.get("yellow_time", 2),
            min_green=payload.get("min_green", 5),
            max_green=payload.get("max_green", 50),
            sim_time=payload.get("sim_time", 0.0),
            total_vehicles=payload.get("total_vehicles", 0),
            total_waiting_time=payload.get("total_waiting_time", 0.0),
        )
```

---

## ⚠️ Critical Challenges

### 1. SUMO System Dependency

**Challenge**: SUMO must be installed at system level (apt-get), not just pip.

**Solution**:
```dockerfile
RUN add-apt-repository ppa:sumo/stable && \
    apt-get update && \
    apt-get install -y sumo sumo-tools
```

### 2. Network Files Required

**Challenge**: SUMO needs `.net.xml` and `.rou.xml` files to run.

**Solutions**:
- **Bundle examples**: Copy simple networks from sumo-rl repo
- **Volume mount**: Let users mount their own networks
- **Default config**: Use single-intersection as default

### 3. No GUI Support

**Challenge**: Docker can't run SUMO GUI.

**Solution**: Always use `use_gui=False` in Docker environment.

### 4. Long Simulation Times

**Challenge**: Traffic simulations can take minutes to complete.

**Solution**:
- Set reasonable defaults (20000 seconds simulation time)
- Allow configuration via environment variables
- Document expected runtimes

### 5. Multi-Agent Complexity

**Challenge**: SUMO-RL supports multi-agent (multiple traffic lights).

**Solution**: Start with single-agent only for OpenEnv integration. Multi-agent can be added later.

---

## 📊 Configuration Matrix

| Variable | Default | Description |
|----------|---------|-------------|
| `SUMO_NET_FILE` | `/app/nets/single-intersection.net.xml` | Network topology file |
| `SUMO_ROUTE_FILE` | `/app/nets/single-intersection.rou.xml` | Vehicle routes file |
| `SUMO_NUM_SECONDS` | `20000` | Simulation duration |
| `SUMO_DELTA_TIME` | `5` | Seconds between actions |
| `SUMO_YELLOW_TIME` | `2` | Yellow phase duration |
| `SUMO_MIN_GREEN` | `5` | Minimum green time |
| `SUMO_MAX_GREEN` | `50` | Maximum green time |
| `SUMO_REWARD_FN` | `diff-waiting-time` | Reward function |

### Available Reward Functions

From SUMO-RL source:
- `diff-waiting-time` (default) - Change in cumulative waiting time
- `average-speed` - Average speed of vehicles
- `queue` - Total queue length
- `pressure` - Pressure (difference between incoming/outgoing vehicles)

---

## 🧪 Testing Strategy

### 1. Pre-Flight Checks
- Verify network files exist
- Check SUMO installation
- Validate Dockerfile syntax
- Test imports

### 2. Docker Build Test
```bash
docker build -f src/envs/sumo_rl_env/server/Dockerfile -t sumo-rl-env:latest .
```

### 3. Runtime Tests
```bash
docker run -p 8000:8000 sumo-rl-env:latest

curl http://localhost:8000/health
curl -X POST http://localhost:8000/reset
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"phase_id": 1, "ts_id": "0"}}'
```

### 4. Python Client Test
```python
from envs.sumo_rl_env import SumoRLEnv, SumoAction

env = SumoRLEnv.from_docker_image("sumo-rl-env:latest")
result = env.reset()
result = env.step(SumoAction(phase_id=1))
print(f"Reward: {result.reward}, Done: {result.done}")
env.close()
```

---

## 📦 What to Bundle

### Minimal Network Example

Bundle the single-intersection example from sumo-rl:
```
sumo-rl/sumo_rl/nets/single-intersection/
├── single-intersection.net.xml  # Network topology
├── single-intersection.rou.xml  # Vehicle routes
```

This provides a working example out-of-the-box.

### Additional Networks (Optional)

Could bundle RESCO benchmarks for research:
- `grid4x4` - 4×4 grid of intersections
- `arterial4x4` - Arterial road network
- `cologne1` - Real-world Cologne network

But start with single-intersection for simplicity.

---

## 🎯 Implementation Plan

### Phase 1: Core Implementation (4-6 hours)
1. Create `models.py` ✓ (designed)
2. Create `server/sumo_environment.py` ✓ (designed)
3. Create `server/app.py` ✓ (designed)
4. Create `server/Dockerfile` ✓ (designed)
5. Create `client.py` ✓ (designed)

### Phase 2: Testing (2-3 hours)
1. Build Docker image
2. Test basic functionality
3. Test different configurations
4. Verify reward functions work

### Phase 3: Documentation (1-2 hours)
1. Write README.md
2. Create examples
3. Document network file format
4. Add to GitHub Actions

### Phase 4: Integration (1 hour)
1. Add to `.github/workflows/docker-build.yml`
2. Update main README
3. Add to environments list

**Total Estimate**: 8-12 hours

---

## 🚀 Next Steps

1. **Create file structure** in `/Users/sanyambhutani/GH/OpenEnv/src/envs/sumo_rl_env/`
2. **Copy network files** from `/Users/sanyambhutani/OpenEnv/sumo-rl/sumo_rl/nets/`
3. **Implement all files** following the designs above
4. **Build and test Docker image**
5. **Create documentation**
6. **Add to GitHub Actions**

---

## 💡 Key Insights

### Why SUMO-RL is Harder Than Atari

1. **System Dependencies**: Atari (ale-py) is pip-installable, SUMO requires apt-get
2. **Configuration Complexity**: Atari just needs game name, SUMO needs network files
3. **Runtime**: Atari is fast, SUMO simulations can take minutes
4. **File Dependencies**: Atari bundles ROMs, SUMO needs user-provided networks

### Why It's Still Doable

1. **Single-Agent Mode**: Simplifies to standard Gymnasium API
2. **Bundle Example**: Include simple network to start immediately
3. **Environment Variables**: Easy runtime configuration
4. **Pattern Reuse**: Follow exact Atari pattern for consistency

---

## 📚 References

- [SUMO-RL GitHub](https://github.com/LucasAlegre/sumo-rl)
- [SUMO Documentation](https://sumo.dlr.de/docs/)
- [SUMO-RL Docs](https://lucasalegre.github.io/sumo-rl/)
- [RESCO Benchmarks Paper](https://people.engr.tamu.edu/guni/Papers/NeurIPS-signals.pdf)

---

**Status**: Design complete, ready for implementation
**Complexity**: High (system dependencies + network files)
**Time Estimate**: 8-12 hours
**Confidence**: 85% (Dockerfile complexity is main risk)
