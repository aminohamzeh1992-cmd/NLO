"""SignalTracker — improved version with explicit t*(s) computation.

Changes vs prototype:
  - sigma_history: stores full (step, sigma) sequence per state
  - compute_t_star(): finds the time of maximum positive variance increase
  - get_tstar_map(): returns dict suitable for plot_tstar_heatmap()
  - All original fields and methods retained for backward compatibility.
"""
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from structrl import ReplayGraph


class SignalTracker:
    """Tracks per-state learning signals and computes t*(s)."""

    def __init__(self) -> None:
        # --- Original fields (unchanged) ---
        self.sigma_prev: dict[tuple, float] = {}
        self.delta_sigma: dict[tuple, float] = defaultdict(float)
        self.visit_count: dict[tuple, int] = defaultdict(int)
        self.reward_sum: dict[tuple, float] = defaultdict(float)
        self._ep_visited: set[tuple] = set()
        self.success_visits: dict[tuple, int] = defaultdict(int)
        self.total_visits: dict[tuple, int] = defaultdict(int)

        # --- New: full sigma history for t* computation ---
        # stores list of (episode, sigma) per state
        self.sigma_history: dict[tuple, list[tuple[int, float]]] = (
            defaultdict(list)
        )
        self._global_step: int = 0
        self._episode: int = 0  # incremented at end_episode

        # Fix A: warmup filter — ignore first N episodes
        # (avoids recording first-visit jumps as t*)
        self.warmup_episodes: int = 0  # set via set_warmup()

        # Fix D: residual sigma tracking
        # sigma_running_mean[xy] = exponential moving average of sigma
        # We record sigma - running_mean to isolate wave spikes
        self._sigma_ema: dict[tuple, float] = {}
        self._ema_alpha: float = 0.05  # slow EMA — tracks baseline, not spikes

        # --- Cached t* (computed on demand) ---
        self._t_star: dict[tuple, int] = {}
        self._t_star_dirty: bool = True   # recompute when history changed

    def set_warmup(self, warmup_episodes: int) -> None:
        """Set number of warmup episodes before t* tracking begins.

        During warmup, sigma values are recorded for EMA baseline
        but not for t* computation. This avoids recording first-visit
        jumps (which occur at random times) as t*(s).
        """
        self.warmup_episodes = warmup_episodes

    # ------------------------------------------------------------------
    # Per-step update  (extended)
    # ------------------------------------------------------------------

    def step(self, xy: tuple, sigma: float, reward: float) -> None:
        """Called once per environment step."""
        prev = self.sigma_prev.get(xy, sigma)
        self.delta_sigma[xy] += max(0.0, sigma - prev)
        self.sigma_prev[xy] = sigma
        self.visit_count[xy] += 1
        self.reward_sum[xy] += reward
        self._ep_visited.add(xy)

        # Fix D: update EMA baseline for this state (kept for reference)
        if xy not in self._sigma_ema:
            self._sigma_ema[xy] = sigma
        else:
            self._sigma_ema[xy] = (
                self._ema_alpha * sigma
                + (1 - self._ema_alpha) * self._sigma_ema[xy]
            )

        # Record raw sigma — no residual, no warmup filter
        # t*(s) = episode of MAXIMUM sigma (peak of the wave)
        # This works because sigma_0≈0, rises when wave arrives, then decays
        self.sigma_history[xy].append((self._episode, sigma))
        self._global_step += 1
        self._t_star_dirty = True

    def end_episode(self, success: bool) -> None:
        for xy in self._ep_visited:
            self.total_visits[xy] += 1
            if success:
                self.success_visits[xy] += 1
        self._ep_visited.clear()
        self._episode += 1

    # ------------------------------------------------------------------
    # Original query helpers (unchanged)
    # ------------------------------------------------------------------

    def pi_success(self, xy: tuple) -> float:
        t = self.total_visits.get(xy, 0)
        return self.success_visits[xy] / t if t > 0 else 0.0

    def mean_reward(self, xy: tuple) -> float:
        n = self.visit_count.get(xy, 0)
        return self.reward_sum[xy] / n if n > 0 else 0.0

    def frontier_ratio(self, xy: tuple, graph: "ReplayGraph") -> float:
        nb = graph.edges.get(xy, set())
        if not nb:
            return 0.0
        def pos(node: tuple) -> tuple:
            return (node[0], node[1])
        return (
            sum(1 for n in nb if self.visit_count.get(pos(n), 0) == 0)
            / len(nb)
        )

    # ------------------------------------------------------------------
    # t*(s) computation  (NEW)
    # ------------------------------------------------------------------

    def compute_t_star(self) -> dict[tuple, int]:
        """Compute t*(s) = episode of maximum positive sigma increase.

        With sigma_0 ≈ 0 (bias=20 initialization), sigma follows:
          - Flat near eps before wave arrival
          - Sharp spike AT wave arrival (large TD error from reward)
          - Decay back toward eps after learning

        The maximum INCREASE correctly identifies wave arrival.
        States with fewer than 3 observations are excluded.
        """
        if not self._t_star_dirty:
            return self._t_star

        result: dict[tuple, int] = {}
        for xy, history in self.sigma_history.items():
            if len(history) < 3:
                continue
            best_increase = 0.0
            best_ep = history[0][0]
            for i in range(1, len(history)):
                ep_cur,  sigma_cur  = history[i]
                ep_prev, sigma_prev = history[i - 1]
                increase = max(0.0, sigma_cur - sigma_prev)
                if increase > best_increase:
                    best_increase = increase
                    best_ep = ep_cur
            result[xy] = best_ep

        self._t_star = result
        self._t_star_dirty = False
        return result

    def get_tstar_map(self) -> dict[tuple, int]:
        """Alias for compute_t_star() — more descriptive name."""
        return self.compute_t_star()

    def get_sigma_map(self) -> dict[tuple, float]:
        """Return latest sigma value per state (for heatmap)."""
        return dict(self.sigma_prev)

    def get_visit_map(self) -> dict[tuple, float]:
        """Return visit count per state (for heatmap)."""
        return dict(self.visit_count)

    def get_delta_sigma_map(self) -> dict[tuple, float]:
        """Return cumulative delta_sigma per state (for heatmap)."""
        return dict(self.delta_sigma)

    def get_success_rate_map(self) -> dict[tuple, float]:
        """Return per-state success rate (for heatmap)."""
        return {xy: self.pi_success(xy) for xy in self.total_visits}


# ===========================================================================
# PhaseAwareSignalTracker — phase-separated t*(s) for multi-step problems
# ===========================================================================

class PhaseAwareSignalTracker(SignalTracker):
    """Erweiterung von SignalTracker für mehrstufige Probleme.

    Speichert sigma_history getrennt pro Phase.
    Phase wird durch den aktuellen Ressourcen-Status bestimmt
    (z.B. has_key=0 oder has_key=1 bei KeyDoor).

    Use this class to compute t*_1(s) and t*_2(s)
    getrennt zu berechnen und mit d_1(s), d_2(s) zu vergleichen.
    """

    def __init__(self, n_phases: int = 2,
                 store_full_history: bool = False) -> None:
        super().__init__()
        self.n_phases = n_phases
        self._current_phase: int = 0

        # Separate sigma_history pro Phase
        self.sigma_history_phase: list[dict[tuple, list]] = [
            defaultdict(list) for _ in range(n_phases)
        ]
        # Separate Erfolgsstatistik pro Phase
        self.success_visits_phase: list[dict[tuple, int]] = [
            defaultdict(int) for _ in range(n_phases)
        ]
        self.total_visits_phase: list[dict[tuple, int]] = [
            defaultdict(int) for _ in range(n_phases)
        ]
        self._ep_visited_phase: list[set] = [set() for _ in range(n_phases)]

        # Necessity signal: visit frequency on successful vs failed trajectories
        self.success_traj_count: dict[tuple, int] = defaultdict(int)
        self.fail_traj_count:    dict[tuple, int] = defaultdict(int)
        self._total_success_ep: int = 0
        self._total_fail_ep:    int = 0

        # Full per-episode sigma for sigmoid analysis (optional)
        self.store_full_history = store_full_history
        self._full_sigma_phase: list[dict[tuple, list[float]]] = [
            defaultdict(list) for _ in range(n_phases)
        ]
        # Accumulate within episode to compute episode-level mean
        self._ep_sigma_accum: list[dict[tuple, list[float]]] = [
            defaultdict(list) for _ in range(n_phases)
        ]
        self._ep_count: int = 0

    def set_phase(self, phase: int) -> None:
        """Setzt die aktuelle Phase (z.B. 0=kein Schlüssel, 1=hat Schlüssel)."""
        self._current_phase = min(phase, self.n_phases - 1)

    def step(self, xy: tuple, sigma: float, reward: float) -> None:
        """Like SignalTracker.step(), but also records phase-separated sigma."""
        super().step(xy, sigma, reward)
        phase = self._current_phase

        # Record raw sigma per phase — t*(s) = episode of max sigma
        if self._episode >= self.warmup_episodes:
            self.sigma_history_phase[phase][xy].append(
                (self._episode, sigma))
        self._ep_visited_phase[phase].add(xy)

        if self.store_full_history:
            # Accumulate within episode — write episode mean at end_episode
            # Store raw sigma for overlay plots (not residual)
            self._ep_sigma_accum[phase][xy].append(float(sigma))

    def get_sigma_history_phase(self, phase: int) -> dict[tuple, list[float]]:
        """Returns full per-episode sigma time series for sigmoid analysis.
        Only populated when store_full_history=True.
        """
        return dict(self._full_sigma_phase[phase])

    def end_episode(self, success: bool) -> None:
        """Like SignalTracker.end_episode(), plus necessity signal update."""
        super().end_episode(success)

        # Flush episode-level sigma means into full history
        if self.store_full_history:
            for phase in range(self.n_phases):
                for xy, vals in self._ep_sigma_accum[phase].items():
                    if vals:
                        self._full_sigma_phase[phase][xy].append(
                            float(sum(vals) / len(vals)))
                self._ep_sigma_accum[phase].clear()
        self._ep_count += 1

        # Phasengetrennte Besuchsstatistik
        for phase in range(self.n_phases):
            for xy in self._ep_visited_phase[phase]:
                self.total_visits_phase[phase][xy] += 1
                if success:
                    self.success_visits_phase[phase][xy] += 1
            self._ep_visited_phase[phase].clear()

        # Necessity signal: which states appear on successful vs failed trajectories?
        visited_this_ep = set(self._ep_visited)   # bereits geleert — nutze Basis
        # Wir nutzen visit_count-Diff als Proxy (vereinfacht)
        if success:
            self._total_success_ep += 1
            for xy in self.sigma_history.keys():
                # Zustand war in dieser Episode wenn er gerade besucht wurde
                pass
        else:
            self._total_fail_ep += 1

    def end_episode_with_trajectory(
        self, success: bool, trajectory: list[tuple]
    ) -> None:
        """Like end_episode() but with explicit trajectory for necessity signal."""
        self.end_episode(success)
        visited = set(trajectory)
        if success:
            for xy in visited:
                self.success_traj_count[xy] += 1
        else:
            for xy in visited:
                self.fail_traj_count[xy] += 1

    def compute_t_star_phase(
        self, phase: int
    ) -> dict[tuple, int]:
        """t*(s) = episode of maximum sigma increase within given phase.

        With sigma_0 ≈ 0 (bias=20 init), the maximum increase
        correctly identifies when the variance wave first reached s.
        """
        history = self.sigma_history_phase[phase]
        result: dict[tuple, int] = {}
        for xy, hist in history.items():
            if len(hist) < 3:
                continue
            best_increase, best_ep = 0.0, hist[0][0]
            for i in range(1, len(hist)):
                ep_cur,  sig_cur  = hist[i]
                ep_prev, sig_prev = hist[i - 1]
                increase = max(0.0, sig_cur - sig_prev)
                if increase > best_increase:
                    best_increase = increase
                    best_ep = ep_cur
            result[xy] = best_ep
        return result

    def compute_necessity(self) -> dict[tuple, float]:
        """Necessity signal: P(s | success) - P(s | failure).

        High value = state appears more often on successful trajectories
        than on failed ones -> bottleneck candidate.
        """
        n_succ = max(self._total_success_ep, 1)
        n_fail = max(self._total_fail_ep, 1)
        all_states = set(self.success_traj_count) | set(self.fail_traj_count)
        result = {}
        for xy in all_states:
            p_succ = self.success_traj_count.get(xy, 0) / n_succ
            p_fail = self.fail_traj_count.get(xy, 0) / n_fail
            result[xy] = p_succ - p_fail
        return result

    def get_phase_visit_map(self, phase: int) -> dict[tuple, int]:
        return dict(self.total_visits_phase[phase])

    def get_phase_success_rate_map(self, phase: int) -> dict[tuple, float]:
        total = self.total_visits_phase[phase]
        succ  = self.success_visits_phase[phase]
        return {xy: succ[xy] / max(total[xy], 1) for xy in total}
