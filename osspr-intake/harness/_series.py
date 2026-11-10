"""Fixed piecewise-constant series for the intake benchmarks.

NumPy only, so every script builds the same data before torch is involved.
"""

import numpy as np

# One to three noise standard deviations, four regimes. Same idea as
# benchmarks/workloads.py, kept local so the harness does not import the
# library's benchmark package.
SEGMENT_MEANS = (0.0, 2.0, -1.0, 1.5)


def piecewise_series(n_obs, dims=1, seed=0):
    """float64 array, shape ``[n_obs]`` or ``[n_obs, dims]``, plus true starts.

    Four equal segments. In more than one dimension every segment mean is the
    scalar level times one fixed unit vector, so the shift has the same size
    in every dimension count.
    """
    if n_obs < 4:
        raise ValueError(f"n_obs must be at least 4, got {n_obs}")
    rng = np.random.default_rng(seed)
    count = len(SEGMENT_MEANS)
    sizes = [n_obs // count] * (count - 1)
    sizes.append(n_obs - sum(sizes))
    if dims == 1:
        parts = [np.full(size, mean) for size, mean in zip(sizes, SEGMENT_MEANS)]
        data = np.concatenate(parts) + rng.standard_normal(n_obs)
    else:
        direction = rng.standard_normal(dims)
        direction /= np.linalg.norm(direction)
        parts = [
            np.tile(mean * direction, (size, 1))
            for size, mean in zip(sizes, SEGMENT_MEANS)
        ]
        data = np.concatenate(parts) + rng.standard_normal((n_obs, dims))
    truth = np.cumsum(sizes)[:-1]
    return np.ascontiguousarray(data, dtype=np.float64), [int(t) for t in truth]
