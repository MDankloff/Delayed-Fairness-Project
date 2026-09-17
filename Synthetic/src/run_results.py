"""Export a BAF notebook run without changing its simulation or estimators."""
from datetime import datetime, timezone
from pathlib import Path
import json
import re
from uuid import uuid4

import numpy as np
import pandas as pd


def _json(value):
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Cannot serialize {type(value).__name__}')


def _slug(value):
    return re.sub(r'[^a-z0-9]+', '_', value.lower()).strip('_')


class RunResults:
    """One unique directory per setup-cell execution; repeated evaluations replace by key."""

    def __init__(self, root):
        self.run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_' + uuid4().hex[:6]
        self.folder = Path(root) / self.run_id
        self.folder.mkdir(parents=True, exist_ok=False)
        self.trajectories = {}
        self.rows = {}
        self.figures = {}
        self.last_context = None
        self.write_metadata({'status': 'running'})

    def write_metadata(self, metadata):
        (self.folder / 'run_metadata.json').write_text(
            json.dumps({'run_id': self.run_id, **metadata}, default=_json, indent=2) + '\n'
        )

    def capture(self, experiments, model, agent, result, opt_out, settings):
        # Training populations are deliberately excluded from evaluation results.
        self.last_context = None
        for dataset, experiment in experiments.items():
            if agent is not experiment['test_agent']:
                continue
            for name, fitted_model in experiment.get('models', {}).items():
                if fitted_model is model:
                    seed = settings.get('seed', settings.get('graph_seed'))
                    context = dict(dataset=dataset, model=name, opt_out=bool(opt_out), seed=seed,
                                   graph_seed=settings.get('graph_seed'), steps=len(result[3]))
                    context.update({k: v for k, v in settings.items()
                                    if isinstance(v, (bool, int, float, str)) and k != 'seed'})
                    context.update(eps=agent.eps, base_s0=agent.base[0], base_s1=agent.base[1],
                                   test_n=agent.n_samples, test_initialization_seed=agent.seed)
                    key = (dataset, name, bool(opt_out), seed)
                    self.trajectories[key] = (context, (result[0], result[3], result[4], result[5], result[9]))
                    self.last_context = context
                    return

    def add(self, context, source, metric, value, *, step=None, statistic='value',
            unit='fraction', group='all', **extra):
        row = dict(run_id=self.run_id, **context, source=source, metric=metric,
                   step=step, statistic=statistic, unit=unit, group=group, value=float(value), **extra)
        key = tuple((k, str(v)) for k, v in row.items() if k != 'value')
        self.rows[key] = row

    def series(self, context, source, metrics):
        percentages = {'S=0', 'S=1', 'retention_s0', 'retention_s1'}
        ratios = {'rep_ratio', 'h_s0', 'h_s1', 'h_ratio'}
        for metric, values in metrics.items():
            unit = ('percent' if metric in percentages else 'percentage_points' if metric == 'disparity'
                    else 'ratio' if metric in ratios else 'fraction')
            for step, value in enumerate(values, 1):
                self.add(context, source, metric, value, step=step, unit=unit)

    def summary(self, source, values, seeds, metric, unit):
        if self.last_context is None:
            return
        context = {k: v for k, v in self.last_context.items() if k not in ('seed', 'graph_seed')}
        for seed, value in zip(seeds, values):
            self.add(dict(context, seed=seed, graph_seed=seed), source, metric, value, unit=unit)
        for statistic, value in [('mean', np.mean(values)), ('std', np.std(values))]:
            self.add(context, source, metric, value, statistic=statistic, unit=unit,
                     n_seeds=len(seeds), seeds=json.dumps(list(seeds)), std_ddof=0)

    def save_figure(self, fig, dataset, name):
        stem = f'{_slug(dataset)}__{name}__seed_2026'
        files = []
        for extension in ('png', 'pdf'):
            filename = f'{stem}.{extension}'
            fig.savefig(self.folder / filename, dpi=300, bbox_inches='tight')
            files.append(filename)
        self.figures[stem] = dict(dataset=dataset, plot=name, seed=2026, files=files)

    def export(self, experiments, extract_metrics, detailed_metrics, metadata):
        import contextlib
        import io
        from evaluation import compute_accuracy, compute_short_cond_fairness
        # Reuse captured trajectories: no training or simulation is run here.
        for context, trajectory in self.trajectories.values():
            s, Xs, Ys, Ds, As = trajectory
            model = experiments[context['dataset']]['models'][context['model']]
            self.series(context, 'evaluation_probability_policy_repayment_labels', extract_metrics(trajectory, model))
            with contextlib.redirect_stdout(io.StringIO()):
                details = detailed_metrics(s, As, Ds, Ys)
            self.series(context, 'detailed_evaluation', details)
            for t, (X, Y, D, A) in enumerate(zip(Xs, Ys, Ds, As), 1):
                A, D, Y = np.asarray(A), np.asarray(D), np.asarray(Y)
                self.add(context, 'evaluation', 'accuracy_full_population_model_prediction',
                         compute_accuracy(s, X, Y, model), step=t)
                self.add(context, 'evaluation', 'short_fairness_probability_full_population',
                         abs(compute_short_cond_fairness(s, X, model)), step=t)
                for group in ('all', 'S=0', 'S=1'):
                    mask = np.ones(len(s), dtype=bool) if group == 'all' else s == int(group[-1])
                    active = mask & (A == 1)
                    self.add(context, 'evaluation', 'active_count', active.sum(), step=t, group=group, unit='count')
                    self.add(context, 'evaluation', 'retention', A[mask].mean() * 100, step=t, group=group, unit='percent')
                    self.add(context, 'evaluation', 'approval_rate_active', D[active].mean() if active.any() else np.nan,
                             step=t, group=group)
                    self.add(context, 'evaluation', 'accuracy_active_sampled_decisions',
                             (D[active] == Y[active]).mean() if active.any() else np.nan, step=t, group=group)
        # Save exactly the arrays used by the plots, separately from repeated-seed evaluations.
        for dataset, experiment in experiments.items():
            for key, opt_out in [('metrics', True), ('metrics_no_opt_out', False), ('detailed_metrics', True)]:
                for model, metrics in experiment[key].items():
                    self.series(dict(dataset=dataset, model=model, opt_out=opt_out, seed=2026,
                                     graph_seed=2026), 'figure_' + key, metrics)
        table = pd.DataFrame(self.rows.values())
        for setting in ('fairness_denominator_clip', 'decision_policy', 'fairness_policy'):
            table[setting] = metadata.get(setting)
        first = ['run_id', 'dataset', 'model', 'opt_out', 'seed', 'graph_seed', 'source',
                 'step', 'group', 'metric', 'statistic', 'value', 'unit']
        table = table[first + [c for c in table if c not in first]]
        path = self.folder / 'all_results.csv'
        table.to_csv(path, index=False, float_format='%.17g')
        self.write_metadata(dict(metadata, status='complete', result_rows=len(table), figures=list(self.figures.values())))
        (self.folder / 'README.txt').write_text(
            'all_results.csv: one numeric value per row. Filter by source before comparing.\n'
            'figure_* rows are the exact plotted arrays (seed 2026).\n'
            'evaluation_probability_policy_repayment_labels uses shared clipping, policy probabilities and Y labels.\n'
            'diagnostic_* rows use the same probability policy and repayment labels as the evaluation.\n'
            'Summary rows include individual seeds, mean and population std (ddof=0).\n'
            'Step numbers start at 1. Average retention disparity excludes step 1.\n'
            'Percent values range from 0 to 100; disparity is in percentage points.\n'
            'Other fairness gaps and signed_delta are fractions, not percentages.\n'
            'NaN/blank means undefined; inf is retained. Metrics are exported without changing definitions.\n'
            'run_metadata.json records settings, fitted parameters, and the figure list.\n'
        )
        return table
