"""Routing policies. Each has select(x) -> arm index and update(x, arm, reward)."""
from __future__ import annotations

import numpy as np


class RidgeArm:
    """Ridge regression for one arm: A = reg*I + sum x x^T, b = sum r x."""

    def __init__(self, d: int, reg: float = 1.0):
        self.A = reg * np.eye(d)
        self.b = np.zeros(d)

    def update(self, x: np.ndarray, r: float) -> None:
        self.A += np.outer(x, x)
        self.b += r * x

    @property
    def theta(self) -> np.ndarray:
        return np.linalg.solve(self.A, self.b)


class Policy:
    name = "policy"

    def select(self, x: np.ndarray) -> int:
        raise NotImplementedError

    def update(self, x: np.ndarray, arm: int, reward: float) -> None:
        pass


class AlwaysSmall(Policy):
    name = "always-small"

    def select(self, x):
        return 0


class AlwaysLarge(Policy):
    name = "always-large"

    def select(self, x):
        return 1


class RandomPolicy(Policy):
    name = "random"

    def __init__(self, n_arms: int, seed: int = 0):
        self.n, self.rng = n_arms, np.random.default_rng(seed)

    def select(self, x):
        return int(self.rng.integers(self.n))


class EpsilonGreedy(Policy):
    """Linear reward model per arm, greedy on it, random arm with probability eps."""
    name = "epsilon-greedy"

    def __init__(self, n_arms: int, d: int, eps: float = 0.1, seed: int = 0):
        self.arms = [RidgeArm(d) for _ in range(n_arms)]
        self.eps, self.rng = eps, np.random.default_rng(seed)

    def select(self, x):
        if self.rng.random() < self.eps:
            return int(self.rng.integers(len(self.arms)))
        return int(np.argmax([a.theta @ x for a in self.arms]))

    def update(self, x, arm, reward):
        self.arms[arm].update(x, reward)


class LinUCB(Policy):
    """Disjoint LinUCB (Li et al. 2010): score = theta.x + alpha * sqrt(x' A^-1 x)."""
    name = "linucb"

    def __init__(self, n_arms: int, d: int, alpha: float = 0.5, reg: float = 1.0):
        self.arms = [RidgeArm(d, reg) for _ in range(n_arms)]
        self.alpha = alpha

    def scores(self, x):
        out = []
        for a in self.arms:
            bonus = np.sqrt(x @ np.linalg.solve(a.A, x))
            out.append(a.theta @ x + self.alpha * bonus)
        return np.array(out)

    def select(self, x):
        return int(np.argmax(self.scores(x)))

    def update(self, x, arm, reward):
        self.arms[arm].update(x, reward)


class LinThompson(Policy):
    """Linear Thompson sampling: draw theta ~ N(theta_hat, v^2 A^-1) per arm, pick the best draw."""
    name = "thompson"

    def __init__(self, n_arms: int, d: int, v: float = 0.3, reg: float = 1.0, seed: int = 0):
        self.arms = [RidgeArm(d, reg) for _ in range(n_arms)]
        self.v, self.rng = v, np.random.default_rng(seed)

    def select(self, x):
        draws = []
        for a in self.arms:
            cov = self.v ** 2 * np.linalg.inv(a.A)
            cov = (cov + cov.T) / 2
            draws.append(self.rng.multivariate_normal(a.theta, cov) @ x)
        return int(np.argmax(draws))

    def update(self, x, arm, reward):
        self.arms[arm].update(x, reward)


def make_policies(n_arms: int, d: int, seed: int) -> list[Policy]:
    return [AlwaysSmall(), AlwaysLarge(), RandomPolicy(n_arms, seed),
            EpsilonGreedy(n_arms, d, 0.1, seed), LinUCB(n_arms, d), LinThompson(n_arms, d, seed=seed)]
