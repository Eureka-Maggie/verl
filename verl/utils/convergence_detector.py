"""Convergence detector for GRPO/DAPO training.

Detects rubric-switch point using two metrics:
  1. reward/mean plateau  (20-step MA, 30-step window, delta < plateau_thresh)
  2. group_max_minus_second_avg below a threshold

Usage in training loop:
    detector = ConvergenceDetector()
    ...
    conv_metrics = detector.step(
        reward_mean=metrics["reward/mean"],
        max_minus_second=metrics.get("reward/group_max_minus_second_avg"),
        std_avg=metrics.get("reward/group_std_avg"),
    )
    metrics.update(conv_metrics)
"""

from collections import deque


class ConvergenceDetector:
    """Detects when model has stopped learning from current rubric.

    Args:
        ma_window:       Steps for reward/mean moving average (smoothing).
        plateau_window:  Steps to check for plateau after smoothing.
        plateau_thresh:  Max allowed change in smoothed reward over plateau_window.
        max2nd_thresh:   group_max_minus_second_avg threshold for convergence.
    """

    def __init__(
        self,
        ma_window: int = 20,
        plateau_window: int = 30,
        plateau_thresh: float = 0.02,
        max2nd_thresh: float = 0.05,
        min_reward_ma: float = 0.0,
    ):
        self.ma_window = ma_window
        self.plateau_window = plateau_window
        self.plateau_thresh = plateau_thresh
        self.max2nd_thresh = max2nd_thresh
        self.min_reward_ma = min_reward_ma

        self._raw = deque(maxlen=ma_window + plateau_window)
        self._ma = deque(maxlen=plateau_window)
        self._switch_logged = False
        # True only on the step where should_switch first becomes True
        self._prev_should_switch = False
        self._num_switches = 0

    def reset(self) -> None:
        """Reset accumulated windows after a rubric switch.

        Clears _raw and _ma so the next rubric's convergence is measured
        from a clean slate, avoiding contamination from the previous rubric's
        reward distribution.
        """
        self._raw.clear()
        self._ma.clear()
        self._prev_should_switch = False
        self._switch_logged = False
        self._num_switches += 1
        print(f"[ConvergenceDetector] Reset after rubric switch #{self._num_switches}. "
              f"Window accumulation restarts.")

    def step(
        self,
        reward_mean: float,
        max_minus_second: float | None = None,
        std_avg: float | None = None,
    ) -> dict:
        """Call once per training step. Returns dict of convergence metrics."""
        self._raw.append(reward_mean)

        # Compute moving average once we have enough raw values.
        ma_val = None
        if len(self._raw) >= self.ma_window:
            raw_list = list(self._raw)
            ma_val = sum(raw_list[-self.ma_window:]) / self.ma_window
            self._ma.append(ma_val)

        # Check plateau in smoothed reward.
        is_plateau = False
        if len(self._ma) >= self.plateau_window:
            ma_list = list(self._ma)
            window = ma_list[-self.plateau_window:]
            is_plateau = (max(window) - min(window)) < self.plateau_thresh

        # Check max-2nd threshold.
        max2nd_low = (max_minus_second is not None) and (max_minus_second < self.max2nd_thresh)

        # Final switch signal: both conditions must be met.
        should_switch = is_plateau and max2nd_low

        # first_trigger: True only on the step that transitions False → True
        first_trigger = should_switch and not self._prev_should_switch
        self._prev_should_switch = should_switch

        # Diagnostic: interpret std_avg level.
        std_level = "unknown"
        if std_avg is not None:
            if std_avg > 0.6:
                std_level = "high"    # signal strong, rubric saturated
            elif std_avg > 0.3:
                std_level = "medium"  # signal fading
            else:
                std_level = "low"     # signal exhausted

        out = {
            "convergence/reward_ma":       ma_val if ma_val is not None else float("nan"),
            "convergence/is_plateau":      float(is_plateau),
            "convergence/max2nd_low":      float(max2nd_low),
            "convergence/should_switch":   float(should_switch),
            "convergence/first_trigger":   float(first_trigger),
        }
        if std_avg is not None:
            # 0=low, 1=medium, 2=high — numeric so wandb can plot it.
            out["convergence/std_level"] = {"low": 0.0, "medium": 1.0, "high": 2.0}.get(std_level, -1.0)

        if first_trigger:
            print(
                f"\n{'='*60}\n"
                f"[ConvergenceDetector] SWITCH SIGNAL (first trigger)\n"
                f"  reward_ma={ma_val:.4f}  max-2nd={max_minus_second:.4f}  "
                f"std_avg={std_avg if std_avg is not None else 'N/A'} ({std_level})\n"
                f"{'='*60}"
            )

        return out
