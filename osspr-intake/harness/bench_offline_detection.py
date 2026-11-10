# ruff: noqa: E402
"""Wall time of one offline_changepoint_detection call.

Realistic univariate series (four mean regimes, length 1200). Prints the
milliseconds of one steady-state call on stdout. The changepoint table
``Pcp`` is the bulk of that time: each of roughly two hundred rows is one
``logsumexp`` over a shrinking block of segment scores.

Run from the repository root::

    python3 osspr-intake/harness/bench_offline_detection.py
"""

import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from functools import partial

import torch
from _series import piecewise_series
from _timing import pin_torch_threads, steady_call_ms

from bayesian_changepoint_detection.bayesian_models import offline_changepoint_detection
from bayesian_changepoint_detection.offline_likelihoods import StudentT
from bayesian_changepoint_detection.priors import const_prior

N_OBS = 1200
SEED = 0


def main():
    pin_torch_threads()
    data_np, _truth = piecewise_series(N_OBS, dims=1, seed=SEED)
    data = torch.tensor(data_np, dtype=torch.float64)
    prior = partial(const_prior, p=1.0 / (N_OBS + 1))

    def _run():
        _Q, _P, pcp = offline_changepoint_detection(
            data, prior, StudentT(device="cpu"), device="cpu"
        )
        return pcp

    elapsed_ms = steady_call_ms(_run, warmup=1)
    print(f"{elapsed_ms:.4f}")


if __name__ == "__main__":
    main()
