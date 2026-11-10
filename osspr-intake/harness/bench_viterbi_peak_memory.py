# ruff: noqa: E402
"""Peak resident-set growth of one viterbi_changepoints call.

``viterbi_changepoints`` keeps two dense ``(T + 1) x (T + 1)`` tables: the
float32 scores ``V`` and the int64 backpointers. This process builds a
length-3500 series, records ``VmRSS``, runs the detector once, and prints
``VmHWM - VmRSS_before`` in bytes. There is no full-size warmup: the high-water
mark is monotonic, so a discarded run would hide the allocation.

Run from the repository root::

    python3 osspr-intake/harness/bench_viterbi_peak_memory.py
"""

import gc
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

from bayesian_changepoint_detection.bayesian_models import viterbi_changepoints
from bayesian_changepoint_detection.hazard_functions import constant_hazard
from bayesian_changepoint_detection.online_likelihoods import StudentT

N_OBS = 3500
SEED = 0


def _proc_bytes(field):
    with open("/proc/self/status", encoding="ascii") as status:
        for line in status:
            if line.startswith(field + ":"):
                return int(line.split()[1]) * 1024
    raise RuntimeError(f"{field} missing from /proc/self/status")


def main():
    torch.set_num_threads(1)
    data_np, _truth = piecewise_series(N_OBS, dims=1, seed=SEED)
    data = torch.tensor(data_np, dtype=torch.float32)
    hazard = partial(constant_hazard, 200)
    # Touch the likelihood class before the window so its import cost is not
    # part of the delta. The model itself is built inside the window: it is
    # part of the call, and it is O(T), far under the two tables.
    gc.collect()
    before = _proc_bytes("VmRSS")
    path, changepoints = viterbi_changepoints(
        data,
        hazard,
        StudentT(alpha=0.1, beta=0.01, kappa=1.0, mu=0.0, device="cpu"),
        device="cpu",
    )
    # Keep both results live so their storage counts toward the high-water mark.
    _keep = (path, changepoints, int(path[-1]), int(changepoints.shape[0]))
    peak = _proc_bytes("VmHWM") - before
    print(int(peak))
    return _keep


if __name__ == "__main__":
    main()
