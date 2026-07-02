"""Small C51 agent with a clean interface for StructRL.

Key fixes vs prototype to match V1 (structrl_minimal) behaviour:
  1. Atom range v_min=-80 / v_max=10  (matches V1; goal=10, max path ~90 steps → min return ≈ -80)
  2. _init_small_sigma(): concentrate initial distribution on centre atom so
     sigma starts near 0 — critical for t* tracking to work correctly.
  3. lr=5e-4  (same as V1)
  4. eps_decay=0.995 in analysis phase / 0.98 in training (same schedule as V1)
  5. gradient clipping (norm 5.0) matching V1
  6. train_step() single-transition method retained for compatibility;
     train_batch() wraps it efficiently for the new modular interface.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Iterable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s_next: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 10_000):
        self.data: deque[Transition] = deque(maxlen=capacity)

    def add(self, s, a, r, s_next, done) -> None:
        self.data.append(Transition(
            np.asarray(s, dtype=np.float32), int(a), float(r),
            np.asarray(s_next, dtype=np.float32), bool(done)
        ))

    def __len__(self) -> int:
        return len(self.data)

    def sample_uniform(self, batch_size: int) -> list[Transition]:
        return random.sample(list(self.data), min(batch_size, len(self.data)))

    def sample_with_probs(self, probs: np.ndarray, batch_size: int) -> list[Transition]:
        probs = np.asarray(probs, dtype=np.float64)
        probs = probs / (probs.sum() + 1e-12)
        idx = np.random.choice(len(self.data), size=batch_size, p=probs, replace=True)
        items = list(self.data)
        return [items[i] for i in idx]

    def all(self) -> list[Transition]:
        return list(self.data)


class C51Net(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, n_atoms: int = 51, hidden: int = 64):
        super().__init__()
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        # V1 uses hidden=64 (two layers of 64)
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim * n_atoms),
        )
        self._init_small_sigma()

    def _init_small_sigma(self) -> None:
        """Initialise output layer to concentrate distribution on centre atom.

        bias=5 gives sigma_0 ≈ 6 (not truly zero, but stable for learning).
        bias=20 gives sigma_0 ≈ 0.004 but collapses learning performance.

        Note: true sigma_0=0 requires separate initialization strategy.
        The sigmoid trajectory theory assumes sigma_0≈eps; this is an
        approximation in the current implementation.
        """
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        center = self.n_atoms // 2
        for a in range(self.action_dim):
            last.bias.data[a * self.n_atoms + center] = 5.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x).view(-1, self.action_dim, self.n_atoms)
        return F.softmax(logits, dim=-1)


class C51Agent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_atoms: int = 51,
        v_min: float = -80.0,   # V1-compatible (goal=10, long paths → min ≈ -80)
        v_max: float = 10.0,    # V1-compatible
        gamma: float = 0.99,
        lr: float = 5e-4,       # V1-compatible
        eps_start: float = 1.0,
        eps_min: float = 0.1,   # V1 uses 0.1
        eps_decay: float = 0.995,
        device: str | None = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.gamma = gamma
        self.eps = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.atoms = torch.linspace(v_min, v_max, n_atoms, device=self.device)
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        self.net = C51Net(state_dim, action_dim, n_atoms).to(self.device)
        self.target = C51Net(state_dim, action_dim, n_atoms).to(self.device)
        self.target.load_state_dict(self.net.state_dict())
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _t(self, arr) -> torch.Tensor:
        return torch.as_tensor(np.asarray(arr), dtype=torch.float32, device=self.device)

    def q_values(self, state_batch: torch.Tensor) -> torch.Tensor:
        pmf = self.net(state_batch)
        return torch.sum(pmf * self.atoms.view(1, 1, -1), dim=-1)

    def fit_q_values(
        self,
        state_batch: torch.Tensor,
        q_targets:   torch.Tensor,
        lr_override: float = 0.005,
    ) -> None:
        """Supervised MSE update: push Q(s,a) toward q_targets (N, A)."""
        self.net.train()
        old_lr = [pg["lr"] for pg in self.optim.param_groups]
        for pg in self.optim.param_groups:
            pg["lr"] = lr_override

        pmf    = self.net(state_batch)
        q_pred = torch.sum(pmf * self.atoms.view(1, 1, -1), dim=-1)
        loss   = torch.nn.functional.mse_loss(q_pred, q_targets)
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        self.optim.step()

        for pg, lr in zip(self.optim.param_groups, old_lr):
            pg["lr"] = lr

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(self, state: np.ndarray, explore: bool = True) -> int:
        if explore and np.random.rand() < self.eps:
            return int(np.random.randint(self.action_dim))
        with torch.no_grad():
            q = self.q_values(self._t(state).unsqueeze(0))
        return int(torch.argmax(q, dim=1).item())

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self, s, a, r, s_next, done, weight: float = 1.0) -> float:
        """Single-transition update — matches V1 interface exactly."""
        s_t = self._t(s).unsqueeze(0)
        sn_t = self._t(s_next).unsqueeze(0)

        with torch.no_grad():
            next_dist = self.target(sn_t)
            atoms = self.atoms.view(1, 1, -1)
            q_next = torch.sum(next_dist * atoms, dim=2)
            a_next = q_next.argmax(dim=1).item()
            target_pmf = next_dist[0, a_next]

            Tz = r + (1 - int(done)) * self.gamma * self.atoms
            Tz = Tz.clamp(self.v_min, self.v_max)
            b = (Tz - self.v_min) / self.delta_z
            lower = b.floor().long().clamp(0, self.n_atoms - 1)
            upper = b.ceil().long().clamp(0, self.n_atoms - 1)

            proj = torch.zeros(self.n_atoms, device=self.device)
            eq = lower == upper
            if eq.any():
                proj.index_add_(0, lower[eq], target_pmf[eq])
            neq = ~eq
            if neq.any():
                proj.index_add_(0, lower[neq], target_pmf[neq] * (upper[neq].float() - b[neq]))
                proj.index_add_(0, upper[neq], target_pmf[neq] * (b[neq] - lower[neq].float()))

        pmf = self.net(s_t)[0, a]
        loss = -(proj * torch.log(pmf.clamp(min=1e-8))).sum()
        loss = loss * (1.0 + 0.5 * (weight - 1.0))

        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
        self.optim.step()
        return float(loss.detach().cpu())

    def train_batch(self, batch: list[Transition]) -> float:
        """Vectorised batch update — single forward+backward pass."""
        if not batch:
            return 0.0

        B = len(batch)
        s_arr  = np.stack([t.s      for t in batch])   # (B, state_dim)
        sn_arr = np.stack([t.s_next for t in batch])
        a_arr  = np.array([t.a      for t in batch], dtype=np.int64)
        r_arr  = np.array([t.r      for t in batch], dtype=np.float32)
        done_arr = np.array([float(t.done) for t in batch], dtype=np.float32)

        s_t  = torch.tensor(s_arr,  dtype=torch.float32, device=self.device)
        sn_t = torch.tensor(sn_arr, dtype=torch.float32, device=self.device)
        r_t  = torch.tensor(r_arr,  dtype=torch.float32, device=self.device)
        done_t = torch.tensor(done_arr, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_dist = self.target(sn_t)               # (B, A, N)
            atoms_v   = self.atoms.view(1, 1, -1)
            q_next    = (next_dist * atoms_v).sum(dim=2) # (B, A)
            a_next    = q_next.argmax(dim=1)             # (B,)
            target_pmf = next_dist[torch.arange(B), a_next]  # (B, N)

            Tz = (r_t.unsqueeze(1)
                  + (1 - done_t).unsqueeze(1) * self.gamma * self.atoms)
            Tz = Tz.clamp(self.v_min, self.v_max)
            b  = (Tz - self.v_min) / self.delta_z
            lower = b.floor().long().clamp(0, self.n_atoms - 1)
            upper = b.ceil().long().clamp(0, self.n_atoms - 1)

            proj = torch.zeros(B, self.n_atoms, device=self.device)
            eq  = lower == upper
            neq = ~eq

            # equal
            proj.scatter_add_(1, lower * eq, target_pmf * eq)

            # lower contribution
            proj.scatter_add_(
                1, lower * neq,
                target_pmf * neq * (upper.float() - b) * neq
            )
            # upper contribution
            proj.scatter_add_(
                1, upper * neq,
                target_pmf * neq * (b - lower.float()) * neq
            )

        pmf_all = self.net(s_t)                          # (B, A, N)
        pmf_a   = pmf_all[torch.arange(B),
                          torch.tensor(a_arr, device=self.device)]  # (B, N)
        loss = -(proj * torch.log(pmf_a.clamp(min=1e-8))).sum(dim=1).mean()

        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
        self.optim.step()
        return float(loss.detach().cpu())

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def update_target(self) -> None:
        self.target.load_state_dict(self.net.state_dict())

    def decay_epsilon(self, rate: float | None = None) -> None:
        rate = rate if rate is not None else self.eps_decay
        self.eps = max(self.eps_min, self.eps * rate)

    # ------------------------------------------------------------------
    # Value / uncertainty queries
    # ------------------------------------------------------------------

    def get_value(self, state: np.ndarray) -> float:
        with torch.no_grad():
            q = self.q_values(self._t(state).unsqueeze(0))
        return float(q.max(dim=1).values.item())

    def get_sigma(self, state: np.ndarray) -> float:
        """Return std-dev of the return distribution for the greedy action."""
        with torch.no_grad():
            x = self._t(state).unsqueeze(0)
            q = self.q_values(x)
            a = int(q.argmax(dim=1).item())
            pmf = self.net(x)[0, a]
            mean = torch.sum(pmf * self.atoms)
            var = torch.sum(pmf * (self.atoms - mean) ** 2)
        return float(torch.sqrt(var + 1e-8).cpu())

    def get_td_error(self, t: Transition) -> float:
        with torch.no_grad():
            v = self.get_value(t.s)
            v_next = 0.0 if t.done else self.get_value(t.s_next)
        return float(t.r + self.gamma * v_next - v)


# ===========================================================================
# MultiResourceC51Agent — separate C51-Köpfe pro Ressource
# ===========================================================================

class MultiResourceC51Agent:
    """Manages n separate C51 agents, one per resource.

    Each agent is trained only on the reward of its own resource.
    New resources can be added dynamically at runtime when the
    agent discovers an unexpected reward.

    Resource 0 is always the global task (final reward).
    Resources 1, 2, ... are intermediate goals (keys, checkpoints, ...).

    Note: this class is planned future work — currently StructRL uses
    a single global C51 network. Sub-networks per resource would allow
    clean phase-separated t*(s) signals.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        global_v_min: float = -200.0,
        global_v_max: float = 15.0,
        resource_v_min: float = -50.0,
        resource_v_max: float = 10.0,
        lr: float = 5e-4,
        eps_start: float = 1.0,
        eps_min: float = 0.1,
        eps_decay: float = 0.995,
    ):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.eps        = eps_start
        self.eps_min    = eps_min
        self.eps_decay  = eps_decay

        # Ressource 0: globales Ziel
        self.agents: list[C51Agent] = [
            C51Agent(
                state_dim, action_dim,
                v_min=global_v_min, v_max=global_v_max,
                lr=lr, eps_start=eps_start,
                eps_min=eps_min, eps_decay=eps_decay,
            )
        ]
        # Atom-Range für neue Ressourcen
        self._res_v_min = resource_v_min
        self._res_v_max = resource_v_max
        self._lr = lr

        # Reward-Quellen die bereits eine Ressource haben
        self._known_rewards: set[float] = set()

    # ------------------------------------------------------------------
    # Ressourcen-Verwaltung
    # ------------------------------------------------------------------

    @property
    def n_resources(self) -> int:
        return len(self.agents)

    def add_resource(self, reward_magnitude: float) -> int:
        """Fügt eine neue Ressource hinzu. Gibt den Index zurück."""
        if reward_magnitude in self._known_rewards:
            return -1   # bereits bekannt
        self._known_rewards.add(reward_magnitude)
        idx = len(self.agents)
        self.agents.append(
            C51Agent(
                self.state_dim, self.action_dim,
                v_min=self._res_v_min,
                v_max=self._res_v_max,
                lr=self._lr,
                eps_start=self.eps,   # starte mit aktueller eps
                eps_min=self.eps_min,
                eps_decay=self.eps_decay,
            )
        )
        return idx

    def maybe_add_resource(
        self, reward: float, threshold: float = 0.5
    ) -> int:
        """Fügt Ressource hinzu wenn reward > threshold und neu.
        Gibt Index der neuen Ressource zurück, oder -1 wenn nicht neu.
        """
        if reward > threshold:
            r_rounded = round(reward, 1)   # Quantisierung
            return self.add_resource(r_rounded)
        return -1

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_global(self, batch: list) -> float:
        """Trainiert Ressource 0 (globales Ziel) mit dem vollen Reward."""
        return self.agents[0].train_batch(batch)

    def train_resource(
        self, resource_idx: int, batch: list,
        reward_filter: float | None = None
    ) -> float:
        """Trainiert Ressource i nur mit Reward von Ressource i.

        reward_filter: wenn gesetzt, wird Reward auf 0 gesetzt
        außer wenn abs(r - reward_filter) < 0.5.
        Das isoliert das Signal der jeweiligen Ressource.
        """
        if resource_idx >= len(self.agents):
            return 0.0
        if reward_filter is None:
            return self.agents[resource_idx].train_batch(batch)

        # Reward maskieren: nur Reward der Zielressource behalten
        from agent import Transition
        filtered = []
        for t in batch:
            r = t.r if abs(t.r - reward_filter) < 0.5 else 0.0
            filtered.append(Transition(t.s, t.a, r, t.s_next, t.done))
        return self.agents[resource_idx].train_batch(filtered)

    def train_all(
        self, batch: list,
        resource_rewards: list[float] | None = None
    ) -> list[float]:
        """Trainiert alle Ressourcen. Gibt Liste der Losses zurück."""
        losses = []
        losses.append(self.agents[0].train_batch(batch))
        for i in range(1, len(self.agents)):
            rr = resource_rewards[i] if resource_rewards else None
            losses.append(self.train_resource(i, batch, rr))
        return losses

    # ------------------------------------------------------------------
    # Sigma und t* pro Ressource
    # ------------------------------------------------------------------

    def get_sigma(self, state, resource_idx: int = 0) -> float:
        return self.agents[resource_idx].get_sigma(state)

    def get_all_sigmas(self, state) -> list[float]:
        """Gibt sigma(s) für alle Ressourcen zurück."""
        return [ag.get_sigma(state) for ag in self.agents]

    def get_sigma_batch(
        self, states: list, resource_idx: int = 0
    ):
        """Batch-Sigma für eine Ressource."""
        from structrl import _batch_sigmas
        return _batch_sigmas(self.agents[resource_idx], states)

    # ------------------------------------------------------------------
    # Aktionsauswahl (globale Ressource entscheidet)
    # ------------------------------------------------------------------

    def act(self, state, explore: bool = False) -> int:
        return self.agents[0].act(state, explore)

    def decay_epsilon(self) -> None:
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
        for ag in self.agents:
            ag.eps = self.eps

    def update_target(self) -> None:
        for ag in self.agents:
            ag.update_target()
