# Atari 100k data provenance

All three result files are byte-for-byte downloads from the pinned IRIS
revision `24326aaaa283c527f42b89b44cfdecf2665a7a16`, under
`https://raw.githubusercontent.com/eloialonso/iris/24326aaaa283c527f42b89b44cfdecf2665a7a16/results/data/`.

| File | Runs per task | SHA-256 |
|---|---:|---|
| `SPR.json` | 100 | `055133e22ddb1e7154bbb48b6d229e3b064ea374911158f5d5c1a92cd59e2ad3` |
| `DrQ(eps).json` | 100 | `a86f9f4243b7eb3f994b6db4c9c87fb698742bfa08e7b06e414be41abf1f0574` |
| `IRIS.json` | 5 | `40f45fc2b02e4830e90476ab41ada4251682d9ccc8edaa7b8e25f0ec51bafb5f` |

Each maps the same 26 Atari task names to final scores. The analysis keeps
every run and aligns columns by sorted task name.

`atari_100k_baselines.json` contains random and human reference scores
extracted from Google Research's rliable notebook at revision
`3ccd9f4dea577a04d3d2b557f259aac08badbd81`. Its SHA-256 is
`8f8f34a7fb3ccd28d28cf8e7d35b24521b82ef606105ff31107d4ec685991523`.
See [benchmark documentation](../README.md#published-data-provenance) for the
notebook hash, original SPR mirror, and license links.
