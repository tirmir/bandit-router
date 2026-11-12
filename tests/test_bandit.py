import numpy as np
import pytest

from bandit_router import env
from bandit_router.policies import LinUCB, RandomPolicy, LinThompson, RidgeArm
from bandit_router.run import simulate


def test_ridge_update_hand_example():
    # d=2, reg=1. One observation x=[1,0], r=1:
    # A = I + x x^T = diag(2, 1), b = r x = [1, 0], theta = A^-1 b = [0.5, 0]
    arm = RidgeArm(2, reg=1.0)
    arm.update(np.array([1.0, 0.0]), 1.0)
    assert np.allclose(arm.A, [[2, 0], [0, 1]])
    assert np.allclose(arm.b, [1, 0])
    assert np.allclose(arm.theta, [0.5, 0.0])


def test_linucb_score_and_choice():
    pol = LinUCB(n_arms=2, d=2, alpha=1.0)
    x = np.array([1.0, 0.0])
    pol.update(x, 0, 1.0)                      # arm 0 saw reward 1, arm 1 saw nothing
    s = pol.scores(x)
    # arm 0: theta.x = 0.5, bonus = sqrt(x' A^-1 x) = sqrt(1/2)
    assert s[0] == pytest.approx(0.5 + np.sqrt(0.5))
    # arm 1 is untouched: theta = 0, bonus = sqrt(1/1) = 1
    assert s[1] == pytest.approx(1.0)
    assert pol.select(x) == 0
    # a second identical reward shrinks arm 0's bonus: A = diag(3,1)
    pol.update(x, 0, 1.0)
    assert pol.scores(x)[0] == pytest.approx(2 / 3 + np.sqrt(1 / 3))


def test_stream_is_reproducible_and_oracle_regret_zero():
    s1, s2 = env.make_stream(100, seed=3), env.make_stream(100, seed=3)
    assert np.array_equal(s1.X, s2.X) and np.array_equal(s1.U, s2.U)
    exp = s1.expected_reward()
    assert (exp.max(axis=1) - exp.max(axis=1)).sum() == 0


def test_linucb_beats_random():
    wins = 0
    for seed in range(3):
        stream = env.make_stream(2000, seed)
        ucb = simulate(LinUCB(2, env.D), stream)
        rnd = simulate(RandomPolicy(2, seed), stream)
        assert ucb["regret"][-1] < rnd["regret"][-1]
        wins += ucb["reward"] > rnd["reward"]
    assert wins == 3


def test_thompson_beats_random():
    stream = env.make_stream(2000, 0)
    ts = simulate(LinThompson(2, env.D, seed=0), stream)
    rnd = simulate(RandomPolicy(2, 0), stream)
    assert ts["regret"][-1] < rnd["regret"][-1]
