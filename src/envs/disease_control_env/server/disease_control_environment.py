# src/envs/disease_control_env/server/disease_control_environment.py
import numpy as np, uuid, random
from core.env_server import Environment
from ..models import DiseaseAction, DiseaseObservation, DiseaseState
from ..profiles import DISEASE_PROFILES, MUTATION_VARIANTS
import numpy as np, uuid, random
from enum import IntEnum
from typing import Any, Dict, List
class P(IntEnum):
    S = 0  # susceptible
    H = 1  # protected (vaccinated/recovered)
    I = 2  # infected
    Q = 3  # quarantined
    D = 4  # deceased

# --- Typing helpers to satisfy Pylance and ensure numeric operations ---
from typing import TypedDict, cast

class _DiseaseProfile(TypedDict):
    mu0: float
    sigma: float
    delta: float
    nu: float
    phi: float

class _VariantParams(TypedDict):
    name: str
    mu0: float
    sigma: float
    delta: float
    nu: float
    phi: float

class DiseaseControlEnvironment(Environment):
    """
    Adds:
      - history buffers for live charts (/timeseries)
      - resource refills (scheduled + noisy)
      - mutation events (single or multi-wave)
    """
    def __init__(
        self,
        T: int = 90,
        dt: float = 0.01,
        disease: str = "covid",
        init_budget: float = 1000.0,
        refill_every_steps: int = 14,
        refill_amount: float = 200.0,
        refill_jitter: float = 0.25,
        mutation_at_steps: tuple[int, ...] = (30,),
        mutation_prob: float = 0.85,
        grid_enabled: bool = True,      # 👈 enable the person grid
        grid_size: int = 32,            # 👈 32x32
        neighborhood: str = "moore",  
    ):
        super().__init__()
        self.T, self.dt, self.substeps = T, dt, int(1/dt)
        self.init_budget = init_budget
        self._rng = np.random.default_rng()

        # event config
        self.refill_every_steps = refill_every_steps
        self.refill_amount = refill_amount
        self.refill_jitter = refill_jitter
        self.mutation_at_steps = set(mutation_at_steps)
        self.mutation_prob = mutation_prob
        self.set_disease_profile(disease)
        # core params (set by preset)
        self.set_disease_profile(disease)
        self.grid_enabled = grid_enabled
        self.N = grid_size
        self.neighborhood = neighborhood
        # histories for live chart
        self._clear_history()

        self.reset()
    
    # -------------------- presets & mutation --------------------
    def set_disease_profile(self, name: str):
        cfg = cast(_DiseaseProfile, DISEASE_PROFILES[name])
        # Ensure attributes are numeric for downstream arithmetic
        self.mu0 = float(cfg["mu0"])      # type: ignore[assignment]
        self.sigma = float(cfg["sigma"])  # type: ignore[assignment]
        self.delta = float(cfg["delta"])  # type: ignore[assignment]
        self.nu = float(cfg["nu"])        # type: ignore[assignment]
        self.phi = float(cfg["phi"])      # type: ignore[assignment]
        self.profile_name = name

    def _apply_mutation_variant(self):
        variant = cast(_VariantParams, random.choice(MUTATION_VARIANTS))
        # Coerce to float to avoid str|float unions from loose dict typing
        self.mu0   *= float(variant["mu0"])   # type: ignore[operator]
        self.sigma *= float(variant["sigma"]) # type: ignore[operator]
        self.delta *= float(variant["delta"]) # type: ignore[operator]
        self.nu    *= float(variant["nu"])    # type: ignore[operator]
        self.phi   *= float(variant["phi"])   # type: ignore[operator]
        return f"mutation:{variant['name']}"

    # -------------------- env lifecycle --------------------
    def _clear_history(self):
        self.hist_step = []
        self.hist_S = []; self.hist_H = []; self.hist_I = []; self.hist_Q = []; self.hist_D = []
        self.hist_budget = []
        self.events = []              # list[str]

    def reset(self) -> DiseaseObservation:
        if self.grid_enabled:
            self.grid = self._init_grid()
            self._sync_macro_from_grid()
        else:
            I0 = self._rng.integers(20, 100) / 100000.0
            self.S, self.H, self.I, self.Q, self.D = 1 - I0, 0.0, I0, 0.0, 0.0

        I0 = self._rng.integers(20, 100) / 100000.0
        self.S, self.H, self.I, self.Q, self.D = 1 - I0, 0.0, I0, 0.0, 0.0
        self.budget = self.init_budget
        self._state = DiseaseState(
            episode_id=str(uuid.uuid4()),
            step_count=0, episode_return=0.0, budget=self.budget, disease=self.profile_name
        )
        self._clear_history()
        self._push_hist(last_event="reset")
        return self._obs(last_event="reset")

    @property
    def state(self) -> DiseaseState:
        return self._state

    # -------------------- helpers --------------------
    def _noise(self) -> float:
        return 1e-4 * np.sqrt(self.dt) * self._rng.normal()

    def _push_hist(self, last_event: str | None):
        s = self._state.step_count
        self.hist_step.append(s)
        self.hist_S.append(float(self.S)); self.hist_H.append(float(self.H))
        self.hist_I.append(float(self.I)); self.hist_Q.append(float(self.Q)); self.hist_D.append(float(self.D))
        self.hist_budget.append(float(self.budget))
        if last_event:
            self.events.append(f"t={s}:{last_event}")

    def _maybe_refill(self) -> str | None:
        if self.refill_every_steps <= 0: return None
        if self._state.step_count > 0 and self._state.step_count % self.refill_every_steps == 0:
            jitter = (1.0 + self._rng.uniform(-self.refill_jitter, self.refill_jitter))
            add = max(0.0, self.refill_amount * jitter)
            self.budget += add
            return f"refill:+{add:.1f}"
        return None

    def _maybe_mutate(self) -> str | None:
        if self._state.step_count in self.mutation_at_steps:
            if random.random() <= self.mutation_prob:
                tag = self._apply_mutation_variant()
                return tag
        return None

    # -------------------- step --------------------
    def step(self, action: DiseaseAction):
        self._state.step_count += 1

        # Continuous controls clamped
        c = float(np.clip(action.closures,    0.0, 1.0))
        v = float(np.clip(action.vaccination, 0.0, 1.0))
        q = float(np.clip(action.quarantine,  0.0, 1.0))
        s = float(np.clip(action.spending,    0.0, 1.0))

        # Spending → intensities (budget constrained)
        closures_cost   = 200.0 * c * s
        vaccine_cost    = 400.0 * v * s
        quarantine_cost = 600.0 * q * s
        total_cost = closures_cost + vaccine_cost + quarantine_cost

        if total_cost > self.budget:
            scale = self.budget / (total_cost + 1e-8)
            c *= scale; v *= scale; q *= scale
            total_cost = self.budget

        self.budget -= total_cost

        # Effective parameters under controls
        mu = self.mu0 / (1.0 + 3.0 * c)
        beta = 0.002 * v
        rho  = 0.020 * q
        if self.grid_enabled:
            # Update micro grid first (one sweep per day)
            self._step_grid(mu=mu, beta=beta, rho=rho)
            # sync macro fractions from grid; compute reward on the delta of macro aggregates
            S_prev, H_prev, I_prev, Q_prev, D_prev = self.S, self.H, self.I, self.Q, self.D
            self._sync_macro_from_grid()
            new_inf = max(self.I - I_prev, 0.0)
            new_deaths = max(self.D - D_prev, 0.0)
        else:
            # existing macro SDE integration (your current loop)
            new_inf = 0.0
            new_deaths = 0.0
            for _ in range(self.substeps):
                ...
                new_inf += max(self.I - I_prev, 0.0)
                new_deaths += max(self.D - D_prev, 0.0)

        econ_term = -(total_cost * 0.01)
        reward = -(10*new_deaths + new_inf) + econ_term
        new_inf = 0.0; new_deaths = 0.0
        for _ in range(self.substeps):
            dS = (self._noise() - self.sigma*self.S*self.I*mu - beta*self.S)
            dH = (beta*self.S + self.phi*(self.I+self.Q) - self.delta*self.H*self.I*mu)
            dI = (self.sigma*self.S*self.I*mu + self.delta*self.H*self.I*mu
                  - (self.nu+self.phi+rho)*self.I)
            dQ = (rho*self.I - (self.nu+self.phi)*self.Q)
            dD = (self.nu*(self.I+self.Q))

            I_prev, D_prev = self.I, self.D

            self.S = max(self.S + dS*self.dt, 0.0)
            self.H = max(self.H + dH*self.dt, 0.0)
            self.I = max(self.I + dI*self.dt, 0.0)
            self.Q = max(self.Q + dQ*self.dt, 0.0)
            self.D = max(self.D + dD*self.dt, 0.0)

            new_inf    += max(self.I - I_prev, 0.0)
            new_deaths += max(self.D - D_prev, 0.0)

        econ_term = -(total_cost * 0.01)       # cost hurts reward
        reward = -(10.0*new_deaths + new_inf) + econ_term

        # Events (refill, mutation) AFTER dynamics this step
        event = None
        ev1 = self._maybe_refill()
        if ev1: event = ev1
        ev2 = self._maybe_mutate()
        if ev2: event = ev2 if event is None else f"{event}|{ev2}"

        self._state.episode_return += reward
        self._state.budget = self.budget

        done = (
            self._state.step_count >= self.T
            or self.I < 1e-6
            or self.D >= 0.1
            or self.budget <= 0.0
        )

        self._push_hist(last_event=event)
        return self._obs(last_event=event), reward, done

    # -------------------- observation --------------------
    def _obs(self, last_event: str | None = None) -> DiseaseObservation:
        return DiseaseObservation(
            S=float(self.S), H=float(self.H), I=float(self.I), Q=float(self.Q), D=float(self.D),
            step=self._state.step_count, budget=float(self.budget), disease=self.profile_name,
            last_event=last_event,
            mu0=float(self.mu0), sigma=float(self.sigma), delta=float(self.delta),
            nu=float(self.nu), phi=float(self.phi),
        )

    # -------------------- endpoints used by web UI --------------------
    # These helpers will be called by FastAPI routes defined in app.py
    def get_timeseries(self, tail: int | None = None) -> dict:
        if tail:
            sl = slice(-tail, None)
        else:
            sl = slice(None)
        return {
            "step":   self.hist_step[sl],
            "S":      self.hist_S[sl],
            "H":      self.hist_H[sl],
            "I":      self.hist_I[sl],
            "Q":      self.hist_Q[sl],
            "D":      self.hist_D[sl],
            "budget": self.hist_budget[sl],
            "events": self.events[sl] if hasattr(self.events, "__getitem__") else self.events,
        }
    
    def _init_grid(self):
        # Start mostly S with a few I sprinkled in
        g = np.full((self.N, self.N), P.S, dtype=np.uint8)
        seeds = self._rng.integers(3, 10)
        xs = self._rng.integers(0, self.N, size=seeds)
        ys = self._rng.integers(0, self.N, size=seeds)
        g[xs, ys] = P.I
        return g
    
    def _neighbors(self, x, y):
        if self.neighborhood == "von_neumann":
         coords = [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]
        else:  # moore
            coords = [(x+i, y+j) for i in (-1,0,1) for j in (-1,0,1) if not (i==0 and j==0)]
        for (i,j) in coords:
            if 0 <= i < self.N and 0 <= j < self.N:
                yield (i,j)
                
                
    def _step_grid(self, mu, beta, rho):
        
        """
        One macro 'day' worth of micro transitions (single sweep).
        Probabilities derived from macro params & actions.
        """
        g = self.grid
        new_g = g.copy()

        # map macro parameters to per-contact probabilities
        # closures (mu↓) reduces effective contacts; beta is S->H vaccination rate (global)
        # base infection chance per infectious neighbor:
        base_inf = min(0.4, float(self.sigma * mu * 0.08))  # bounded for stability
        rec_I = min(0.35, float(self.phi * 0.5))            # I -> H
        die_I = min(0.20, float(self.nu * 8))                # I -> D
        die_Q = min(0.20, float(self.nu * 8))                # Q -> D
        rec_Q = min(0.50, float(self.phi * 1.2))             # Q -> H
        go_Q  = min(0.60, float(rho * 1.0))                  # I -> Q

        # vaccinate a random subset of S globally based on beta
        if beta > 0:
            vac_mask = (g == P.S) & (self._rng.random((self.N, self.N)) < min(0.25, beta * 20.0))
            new_g[vac_mask] = P.H

        # loop cells (vectorization is possible; keep simple & clear)
        for x in range(self.N):
            for y in range(self.N):
                s = g[x, y]
                if s == P.S:
                    # infection from any infected neighbor
                    inf_p = 1.0
                    infected_neighbors = 0
                    for (i, j) in self._neighbors(x, y):
                        if g[i, j] == P.I:
                            infected_neighbors += 1
                            inf_p *= (1.0 - base_inf)
                    # probability at least one successful infection
                    p_any = 1.0 - inf_p
                    if self._rng.random() < p_any:
                        new_g[x, y] = P.I

                elif s == P.I:
                    # die?
                    r = self._rng.random()
                    if r < die_I:
                        new_g[x, y] = P.D
                    else:
                        # quarantine?
                        if self._rng.random() < go_Q:
                            new_g[x, y] = P.Q
                        else:
                            # recover?
                            if self._rng.random() < rec_I:
                                new_g[x, y] = P.H

                elif s == P.Q:
                    # die or recover
                    if self._rng.random() < die_Q:
                        new_g[x, y] = P.D
                    elif self._rng.random() < rec_Q:
                        new_g[x, y] = P.H
                # H, D remain as-is

        self.grid = new_g
        
        
    def _sync_macro_from_grid(self):
        # Convert grid counts -> macro fractions (S,H,I,Q,D)
        tot = float(self.N * self.N)
        counts = np.bincount(self.grid.ravel(), minlength=5)
        self.S = counts[P.S] / tot
        self.H = counts[P.H] / tot
        self.I = counts[P.I] / tot
        self.Q = counts[P.Q] / tot
        self.D = counts[P.D] / tot
        
       
    def get_grid(self) -> Dict[str, Any]:
        if not getattr(self, "grid_enabled", False):
            return {
                "enabled": False,
                "size": 0,
                "grid": [],
                "palette": {},
            }

        return {
            "enabled": True,
            "size": self.N,
            "grid": self.grid.tolist(),
            "palette": { "S":0, "H":1, "I":2, "Q":3, "D":4 },
        }