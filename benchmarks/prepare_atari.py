"""Create the notebook's human-normalized SPR final-score matrix."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
constants = json.loads((HERE / 'data/atari_100k_baselines.json').read_text())
assert set(constants) == {'RANDOM_SCORES', 'HUMAN_SCORES'}
raw = json.loads((HERE / 'data/SPR.json').read_text())
assert set(raw) == set(constants['RANDOM_SCORES']) == set(constants['HUMAN_SCORES'])
assert {len(v) for v in raw.values()} == {100}
games = sorted(raw)
scores = np.stack([(np.asarray(raw[g], dtype=np.float64) - constants['RANDOM_SCORES'][g]) /
                   (constants['HUMAN_SCORES'][g] - constants['RANDOM_SCORES'][g])
                   for g in games], axis=1)
assert scores.shape == (100, 26) and np.isfinite(scores).all()
np.save(HERE / 'data/atari_spr.npy', scores)
print('games:', games)
print('matrix:', scores.shape)
