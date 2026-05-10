from __future__ import annotations

import numpy as np


def split_indices(n: int, seed: int, fracs=(0.70, 0.15, 0.15)):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(round(fracs[0] * n))
    n_val = int(round(fracs[1] * n))
    train_i = idx[:n_train]
    val_i = idx[n_train : n_train + n_val]
    test_i = idx[n_train + n_val :]
    return train_i, val_i, test_i
