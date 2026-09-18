"""Process-parallel BAF evaluation; workers never mutate notebook result stores."""
from dataclasses import dataclass, field
from copy import copy

import numpy as np
from joblib import Parallel, delayed, parallel_config
import torch

from generator import run_baf_with_intervention


@dataclass
class SimulationJob:
    model: object
    bank: object
    agent: object
    seed: int
    steps: int = 5
    opt_out: bool = True
    settings: dict = field(default_factory=dict)
    context: dict = field(default_factory=dict)
    include_graph: bool = False

    def simulation_settings(self):
        return {**self.settings, 'seed': self.seed, 'graph_seed': self.seed}


def _run_job(job):
    """Seed each run explicitly, including n_jobs=1, without leaking RNG state."""
    state = np.random.get_state()
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        np.random.seed(job.seed)
        with torch.no_grad():
            result = run_baf_with_intervention(
                job.model, copy(job.agent), job.steps, repayment_model=job.bank,
                enable_opt_out=job.opt_out, **job.simulation_settings())
        if not job.include_graph:
            # Metrics/capture need histories, not the dense adjacency matrix.
            result = (result[0], None, None, *result[3:])
        return result
    finally:
        np.random.set_state(state)
        if previous_threads != 1:
            torch.set_num_threads(previous_threads)


def iter_simulations(jobs, n_jobs=5):
    """Yield (original parent job, result) in input order with bounded dispatch.

    Keep original object identities for RunResults.capture in the parent. A
    streaming result iterator avoids retaining all dense histories at once.
    n_jobs=1 provides the same seeded worker path for debugging and comparison.
    """
    jobs = list(jobs)
    if n_jobs < 1:
        raise ValueError('n_jobs must be a positive integer.')
    if not jobs:
        return
    workers = min(n_jobs, len(jobs))
    if workers == 1:
        for job in jobs:
            yield job, _run_job(job)
        return
    with parallel_config(backend='loky', inner_max_num_threads=1):
        results = Parallel(n_jobs=workers, return_as='generator', batch_size=1,
                           pre_dispatch=workers)(delayed(_run_job)(job) for job in jobs)
        for job, result in zip(jobs, results):
            yield job, result
