"""Seeded sanity check of 95% IQM interval coverage on standard normal scores."""
import json

import numpy as np

from run import backend, iqm

trials, reps = 100, 2_000
rng = np.random.default_rng(1107)
covered = {'rliable': 0, 'ferrograd': 0}
widths = {'rliable': [], 'ferrograd': []}
for trial in range(trials):
    data = {'algorithm': rng.normal(size=(5, 26))}
    for name in covered:
        kwargs = dict(reps=reps, seed=trial)
        if name == 'ferrograd':
            kwargs['threads'] = 1
        point, interval = backend(name)(data, **kwargs)
        lower, upper = interval['algorithm'][:, 0]
        covered[name] += int(lower <= 0 <= upper)
        widths[name].append(float(upper - lower))
        assert abs(point['algorithm'][0] - iqm(data['algorithm'])) < 1e-12
print(json.dumps(dict(distribution='standard normal', shape=[5, 26],
                      population_iqm=0, seed=1107, trials=trials, reps=reps,
                      confidence=.95, coverage=covered,
                      mean_width={name: float(np.mean(values)) for name, values in widths.items()}),
                 indent=2))
