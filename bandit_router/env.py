"""Synthetic routing environment. No real LLM is called anywhere in this repo.

Each request has features:  [bias, prompt_length, difficulty, is_code, is_math]
(prompt_length and difficulty are in [0, 1]; task type is chat / code / math).
Each arm (model) has a hidden success probability that depends on the features
and a fixed cost. Reward = quality - lam * cost, where quality is 1 if the
model "answers well" and 0 otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ARMS = ["small", "large"]
FEATURE_NAMES = ["bias", "prompt_length", "difficulty", "is_code", "is_math"]
D = len(FEATURE_NAMES)

# Hidden weights of the success probability: p = sigmoid(W[arm] @ x)
W = np.array([
    [4.2, -0.8, -7.0, -0.5, -0.8],   # small: good on easy chat, weak on hard, code and math
    [3.6, -0.3, -2.0, -0.1, -0.2],   # large: strong everywhere, mild drop on hard
])
COST = np.array([0.1, 1.0])          # relative price per request


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def success_prob(x: np.ndarray) -> np.ndarray:
    """x: (..., D). Returns (..., n_arms) success probabilities."""
    return sigmoid(x @ W.T)


@dataclass
class Stream:
    """A fixed stream of requests and pre-drawn randomness, so every policy
    sees exactly the same requests and the same luck (common random numbers)."""
    X: np.ndarray        # (T, D)
    P: np.ndarray        # (T, n_arms) true success probabilities
    U: np.ndarray        # (T, n_arms) uniforms: arm succeeds if U < P
    lam: float

    def __len__(self):
        return len(self.X)

    def expected_reward(self) -> np.ndarray:
        """(T, n_arms) expected reward of each arm: P - lam * cost."""
        return self.P - self.lam * COST

    def outcome(self, t: int, arm: int) -> tuple[float, float]:
        """Returns (quality in {0,1}, reward)."""
        q = float(self.U[t, arm] < self.P[t, arm])
        return q, q - self.lam * COST[arm]


def make_stream(T: int, seed: int, lam: float = 0.3) -> Stream:
    rng = np.random.default_rng(seed)
    length = rng.beta(2, 3, size=T)
    difficulty = rng.beta(2, 2, size=T)
    task = rng.choice(3, size=T, p=[0.5, 0.25, 0.25])      # chat, code, math
    # code and math prompts are a little harder on average
    difficulty = np.clip(difficulty + 0.1 * (task > 0), 0, 1)
    X = np.stack([np.ones(T), length, difficulty, task == 1, task == 2], axis=1).astype(float)
    return Stream(X=X, P=success_prob(X), U=rng.random((T, len(ARMS))), lam=lam)
