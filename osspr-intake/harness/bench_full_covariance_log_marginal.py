# ruff: noqa: E402
"""Wall time of offline detection under FullCovarianceLikelihood.

The timed call is one offline detection on a length-480, 12-dimensional
series. ``FullCovarianceLikelihood._log_marginal`` (via ``pdf_rows``, once
per segment start) is the majority of that time: a chain of small float64
kernels for every start index.

Run from the repository root::

    python3 osspr-intake/harness/bench_full_covariance_log_marginal.py
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
from bayesian_changepoint_detection.offline_likelihoods import FullCovarianceLikelihood
from bayesian_changepoint_detection.priors import const_prior

N_OBS = 480
DIMS = 12
SEED = 0


def main():
    pin_torch_threads()
    data_np, _truth = piecewise_series(N_OBS, dims=DIMS, seed=SEED)
    data = torch.tensor(data_np, dtype=torch.float64)
    prior = partial(const_prior, p=1.0 / (N_OBS + 1))

    def _run():
        _Q, _P, pcp = offline_changepoint_detection(
            data,
            prior,
            FullCovarianceLikelihood(device="cpu"),
            device="cpu",
        )
        return pcp

    elapsed_ms = steady_call_ms(_run, warmup=1)
    print(f"{elapsed_ms:.4f}")


if __name__ == "__main__":
    main()
