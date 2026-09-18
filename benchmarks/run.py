"""Reproducible phase-1 IQM speed gate. Run with the pinned reference Python."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time

# Set before NumPy or scientific-library imports, including in child processes.
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[variable] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ferrograd-matplotlib')
import numpy as np

HERE = Path(__file__).resolve().parent


def iqm(scores):
    flat = np.array(scores, dtype=np.float64, copy=True).reshape(-1)
    trim = flat.size // 4
    flat.partition((trim, flat.size - trim - 1))
    return flat[trim:flat.size-trim].mean()


def numpy_interval(score_dict, *, reps, seed=0, confidence=.95):
    points, intervals = {}, {}
    for name, value in score_dict.items():
        scores = np.ascontiguousarray(value, dtype=np.float64)
        if scores.ndim != 2 or not all(scores.shape) or not np.isfinite(scores).all():
            raise ValueError('invalid scores')
        runs, tasks = scores.shape
        count = scores.size
        trim = count // 4
        # Simultaneously-live int64 indices + float64 scores <=64 MiB.
        batch = max(1, (64 * 1024**2) // (16 * count))
        rng = np.random.default_rng(seed)
        values = np.empty(reps)
        columns = np.arange(tasks)
        for start in range(0, reps, batch):
            size = min(batch, reps-start)
            indices = rng.integers(runs, size=(size, runs, tasks))
            sampled = scores[indices, columns].reshape(size, count)
            del indices
            sampled.partition((trim, count-trim-1), axis=1)
            values[start:start+size] = sampled[:, trim:count-trim].mean(axis=1)
            del sampled
        points[name] = np.array([iqm(scores)])
        intervals[name] = np.percentile(values, [50*(1-confidence), 50*(1+confidence)])[:, None]
    return points, intervals


def load(case):
    if case == 'atari_spr':
        return np.load(HERE/'data'/'atari_spr.npy')
    distribution, runs, tasks = case.split('_')
    shape = (int(runs), int(tasks))
    rng = np.random.default_rng(20260916)
    if distribution == 'normal':
        return rng.normal(size=shape)
    if distribution == 'tied':
        return rng.integers(-2, 3, size=shape).astype(float)
    return rng.standard_t(2, size=shape)


def backend(name):
    if name == 'ferrograd':
        from ferrograd import get_interval_estimates
        return get_interval_estimates
    if name == 'numpy':
        return numpy_interval
    from rliable import library, metrics
    def reference(data, *, reps, seed=0):
        np.random.seed(seed)  # rliable 1.2.0 update_indices uses NumPy global RNG.
        return library.get_interval_estimates(
            data, lambda x: np.array([metrics.aggregate_iqm(x)]),
            reps=reps, method='percentile', confidence_interval_size=.95)
    return reference


def cpu_policy():
    cores = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        base = Path(f'/sys/devices/system/cpu/cpu{cpu}')
        key = ((base/'topology/physical_package_id').read_text(),
               (base/'topology/core_id').read_text())
        cores.setdefault(key, (int((base/'cpufreq/cpuinfo_max_freq').read_text()), cpu))
    return [cpu for frequency, cpu in sorted(cores.values(), key=lambda x: (-x[0], x[1]))]


def child(args):
    os.sched_setaffinity(0, cpu_policy()[:args.threads])
    scores = load(args.case)
    call = backend(args.backend)
    kwargs = dict(reps=args.reps, seed=42)
    if args.backend == 'ferrograd':
        kwargs['threads'] = args.threads
    baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if args.memory:
        result = call({'algorithm': scores}, **kwargs)
        print(json.dumps(dict(peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                              baseline_rss_kib=baseline)))
        return
    call({'algorithm': scores}, **kwargs)  # exactly one unmeasured warm-up
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        result = call({'algorithm': scores}, **kwargs)
        samples.append(time.perf_counter()-start)
    print(json.dumps(dict(seconds=samples, median=statistics.median(samples),
                          minimum=min(samples), maximum=max(samples),
                          point=result[0]['algorithm'].tolist(),
                          interval=result[1]['algorithm'].tolist(),
                          affinity=sorted(os.sched_getaffinity(0)))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'results'/'measurements.jsonl')
    parser.add_argument('--case')
    parser.add_argument('--backend', choices=['rliable', 'numpy', 'ferrograd'])
    parser.add_argument('--reps', type=int)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--memory', action='store_true')
    parser.add_argument('--scaling', action='store_true', help='Ferrograd 2/4-thread matrix')
    args = parser.parse_args()
    if args.child:
        child(args)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(python=sys.version, platform=platform.platform(),
                    cpu=Path('/proc/cpuinfo').read_text().split('model name')[1].split('\n')[0],
                    affinity=sorted(os.sched_getaffinity(0)),
                    versions={p: importlib.metadata.version(p) for p in
                              ['numpy', 'rliable', 'arch', 'scipy', 'pandas', 'ferrograd']},
                    source_sha256={str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in [HERE/'run.py', HERE.parent/'src'/'lib.rs',
                                             HERE.parent/'Cargo.lock',
                                             HERE.parent/'python/ferrograd/__init__.py',
                                             *sorted((HERE/'data').glob('*.npy'))]})
    metadata_path = args.output.with_suffix('.metadata.json')
    metadata['cpu_policy'] = cpu_policy()
    metadata['cpu_policy_description'] = 'one logical CPU per physical core, descending maximum frequency then ascending CPU ID'
    metadata['power'] = {name: Path('/sys/devices/system/cpu/cpu0/cpufreq', name).read_text().strip()
                         for name in ['scaling_governor', 'energy_performance_preference']}
    if metadata_path.exists() and args.output.exists():
        if json.loads(metadata_path.read_text()) != metadata:
            raise SystemExit('Metadata changed: use a new --output to avoid mixing builds/environments')
    else:
        metadata_path.write_text(json.dumps(metadata, indent=2)+'\n')
    existing = [json.loads(line) for line in args.output.read_text().splitlines()] if args.output.exists() else []
    done = {(x['case'], x['backend'], x['reps'], x['threads']) for x in existing}
    cases = [f'{d}_{r}_{t}' for r,t in [(5,26), (20,26), (100,100)]
             for d in ['normal', 'tied', 'heavy']]
    if (HERE/'data'/'atari_spr.npy').exists():
        cases.insert(0, 'atari_spr')
    if args.case:
        cases = [args.case]
    for case in cases:
        for reps in ([args.reps] if args.reps else [2000, 50000]):
            for name in ([args.backend] if args.backend else (['ferrograd'] if args.scaling else ['rliable', 'numpy', 'ferrograd'])):
                for threads in ([2,4] if args.scaling else [args.threads]):
                    key = (case, name, reps, threads)
                    if key in done:
                        continue
                    command = [sys.executable, str(Path(__file__).resolve()), '--child', '--case', case,
                               '--backend', name, '--reps', str(reps), '--threads', str(threads)]
                    print(f'Running {key}', flush=True)
                    timing = json.loads(subprocess.check_output(command, text=True))
                    memory = json.loads(subprocess.check_output(command+['--memory'], text=True))
                    row = dict(case=case, shape=list(load(case).shape), backend=name,
                               reps=reps, threads=threads, **timing, **memory)
                    with args.output.open('a') as stream:
                        stream.write(json.dumps(row)+'\n')
                        stream.flush()
                        os.fsync(stream.fileno())
                    print(f"  median {row['median']:.6f}s, peak {row['peak_rss_kib']/1024:.1f} MiB", flush=True)


if __name__ == '__main__':
    main()
