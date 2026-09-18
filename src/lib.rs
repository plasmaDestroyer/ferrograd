#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iqm_trims_floor_quarters() {
        assert_eq!(iqm(&mut [1.0, 2.0, 3.0]), 2.0);
        assert_eq!(iqm(&mut [-100.0, 2.0, 4.0, 100.0]), 3.0);
        assert_eq!(iqm(&mut [-100.0, 2.0, 4.0, 6.0, 100.0]), 4.0);
        for len in 1..=32 {
            let values: Vec<f64> = (0..len).map(|i| ((i * 7) % 11) as f64 - 5.0).collect();
            let mut sorted = values.clone();
            sorted.sort_unstable_by(f64::total_cmp);
            let trim = len / 4;
            let middle = &sorted[trim..len - trim];
            let expected = middle.iter().sum::<f64>() / middle.len() as f64;
            assert_eq!(iqm(&mut values.clone()), expected, "len={len}");
            let mut reversed = values;
            reversed.reverse();
            assert_eq!(iqm(&mut reversed), expected, "reversed len={len}");
        }
    }

    #[test]
    fn percentile_uses_linear_interpolation() {
        let bounds = percentile_bounds(&mut [10.0, 0.0, 30.0, 20.0], 0.5).unwrap();
        assert_eq!(bounds, [7.5, 22.5]);
    }

    #[test]
    fn rliable_statistics_and_ties() {
        let scores = [0.0, 2.0, 4.0, 6.0];
        let metrics = [Metric::Mean, Metric::Median, Metric::Iqm, Metric::Gap];
        assert_eq!(
            metric_values(&scores, 2, 2, &metrics, 2.0, &mut Vec::new()).unwrap(),
            vec![3.0, 3.0, 3.0, 0.5]
        );
        assert_eq!(
            profile_values(&scores, 2, 2, &[1.0, 3.0, 5.0], true).unwrap(),
            vec![0.75, 0.5, 0.25]
        );
        assert_eq!(
            profile_values(&scores, 2, 2, &[1.0, 3.0, 5.0], false).unwrap(),
            vec![1.0, 0.5, 0.0]
        );
        assert_eq!(improvement(&scores, &scores, 2, 2, 2), 0.5);
        assert_eq!(improvement(&[3.0, 4.0], &[2.0], 2, 1, 1), 1.0);
    }

    #[test]
    fn overflowing_task_mean_is_rejected() {
        let scores = [f64::MAX, f64::MAX, -f64::MAX, -f64::MAX];
        assert!(profile_values(&scores, 4, 1, &[0.0], false).is_err());
        assert!(metric_values(&scores, 4, 1, &[Metric::Median], 1.0, &mut Vec::new()).is_err());
    }
}

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyReadonlyArray3, ndarray::Array2};
use pyo3::{exceptions::PyValueError, prelude::*};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

fn iqm(values: &mut [f64]) -> f64 {
    let len = values.len();
    let trim = len / 4;
    if trim > 0 {
        values.select_nth_unstable_by(trim, f64::total_cmp);
        values[trim..].select_nth_unstable_by(len - 2 * trim, f64::total_cmp);
    }
    let middle = &values[trim..len - trim];
    middle.iter().sum::<f64>() / middle.len() as f64
}

fn owned_scores(scores: PyReadonlyArray2<'_, f64>) -> PyResult<(Vec<f64>, usize, usize)> {
    let array = scores.as_array();
    let (runs, tasks) = array.dim();
    if runs == 0 || tasks == 0 || array.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("scores must be nonempty and finite"));
    }
    Ok((array.iter().copied().collect(), runs, tasks))
}

#[derive(Clone, Copy)]
enum Metric {
    Mean,
    Median,
    Iqm,
    Gap,
}

fn parse_metrics(metrics: Vec<String>, gamma: f64) -> PyResult<Vec<Metric>> {
    if metrics.is_empty() || !gamma.is_finite() {
        return Err(PyValueError::new_err(
            "metrics must be nonempty and gamma finite",
        ));
    }
    metrics
        .into_iter()
        .map(|name| match name.as_str() {
            "mean" => Ok(Metric::Mean),
            "median" => Ok(Metric::Median),
            "iqm" => Ok(Metric::Iqm),
            "optimality_gap" => Ok(Metric::Gap),
            _ => Err(PyValueError::new_err(format!("unknown metric: {name}"))),
        })
        .collect()
}

fn check_options(reps: usize, confidence: f64, threads: usize) -> PyResult<()> {
    if reps < 2 || threads == 0 || !(0.0 < confidence && confidence < 1.0) {
        Err(PyValueError::new_err(
            "invalid reps, threads, or confidence",
        ))
    } else {
        Ok(())
    }
}

fn task_means(scores: &[f64], runs: usize, tasks: usize) -> PyResult<Vec<f64>> {
    let mut means = vec![0.0; tasks];
    for run in 0..runs {
        for task in 0..tasks {
            means[task] += scores[run * tasks + task];
            if !means[task].is_finite() {
                return Err(PyValueError::new_err("nonfinite computed task mean"));
            }
        }
    }
    for mean in &mut means {
        *mean /= runs as f64;
    }
    Ok(means)
}

fn metric_values(
    scores: &[f64],
    runs: usize,
    tasks: usize,
    metrics: &[Metric],
    gamma: f64,
    scratch: &mut Vec<f64>,
) -> PyResult<Vec<f64>> {
    let needs_tasks = metrics
        .iter()
        .any(|m| matches!(m, Metric::Mean | Metric::Median));
    let means = if needs_tasks {
        task_means(scores, runs, tasks)?
    } else {
        Vec::new()
    };
    Ok(metrics
        .iter()
        .map(|metric| match metric {
            Metric::Mean => means.iter().sum::<f64>() / tasks as f64,
            Metric::Median => {
                scratch.clear();
                scratch.extend_from_slice(&means);
                scratch.sort_unstable_by(f64::total_cmp);
                (scratch[(tasks - 1) / 2] + scratch[tasks / 2]) / 2.0
            }
            Metric::Iqm => {
                scratch.clear();
                scratch.extend_from_slice(scores);
                iqm(scratch)
            }
            Metric::Gap => {
                gamma - scores.iter().map(|&v| v.min(gamma)).sum::<f64>() / scores.len() as f64
            }
        })
        .collect())
}

fn profile_values(
    scores: &[f64],
    runs: usize,
    tasks: usize,
    thresholds: &[f64],
    use_score_distribution: bool,
) -> PyResult<Vec<f64>> {
    if use_score_distribution {
        Ok(thresholds
            .iter()
            .map(|&threshold| {
                scores.iter().filter(|&&v| v > threshold).count() as f64 / scores.len() as f64
            })
            .collect())
    } else {
        let means = task_means(scores, runs, tasks)?;
        Ok(thresholds
            .iter()
            .map(|&threshold| {
                means.iter().filter(|&&v| v > threshold).count() as f64 / tasks as f64
            })
            .collect())
    }
}

fn improvement(x: &[f64], y: &[f64], runs_x: usize, runs_y: usize, tasks: usize) -> f64 {
    let mut wins = 0.0;
    for task in 0..tasks {
        for run_x in 0..runs_x {
            for run_y in 0..runs_y {
                let a = x[run_x * tasks + task];
                let b = y[run_y * tasks + task];
                wins += f64::from(a > b) + 0.5 * f64::from(a == b);
            }
        }
    }
    wins / (runs_x as f64 * runs_y as f64 * tasks as f64)
}

fn sample(
    scores: &[f64],
    runs: usize,
    tasks: usize,
    task_bootstrap: bool,
    seed: u64,
    rep: usize,
    out: &mut [f64],
) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(rep as u64);
    for task in 0..tasks {
        let source_task = if task_bootstrap {
            rng.gen_range(0..tasks)
        } else {
            task
        };
        for run in 0..runs {
            out[run * tasks + task] = scores[rng.gen_range(0..runs) * tasks + source_task];
        }
    }
}

fn intervals(
    point: Vec<f64>,
    values: Vec<Vec<f64>>,
    confidence: f64,
) -> PyResult<(Vec<f64>, Vec<Vec<f64>>)> {
    if point.iter().any(|v| !v.is_finite()) || values.iter().flatten().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("nonfinite computed statistic"));
    }
    let mut lower = Vec::with_capacity(point.len());
    let mut upper = Vec::with_capacity(point.len());
    for col in 0..point.len() {
        let mut column: Vec<f64> = values.iter().map(|row| row[col]).collect();
        let [lo, hi] = percentile_bounds(&mut column, confidence)?;
        lower.push(lo);
        upper.push(hi);
    }
    Ok((point, vec![lower, upper]))
}

fn pool(threads: usize) -> PyResult<rayon::ThreadPool> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

fn fixed_indices(
    run_indices: PyReadonlyArray3<'_, u64>,
    task_indices: PyReadonlyArray2<'_, u64>,
    runs: usize,
    tasks: usize,
) -> PyResult<(Vec<u64>, Vec<u64>, usize)> {
    let ri = run_indices.as_array();
    let ti = task_indices.as_array();
    let reps = ri.dim().0;
    if reps == 0
        || ri.dim() != (reps, runs, tasks)
        || ti.dim() != (reps, tasks)
        || ri.iter().any(|&r| r >= runs as u64)
        || ti.iter().any(|&t| t >= tasks as u64)
    {
        return Err(PyValueError::new_err("invalid fixed resampling indices"));
    }
    Ok((
        ri.iter().copied().collect(),
        ti.iter().copied().collect(),
        reps,
    ))
}

fn fixed_sample(
    scores: &[f64],
    runs: usize,
    tasks: usize,
    ri: &[u64],
    ti: &[u64],
    rep: usize,
    out: &mut [f64],
) {
    for run in 0..runs {
        for task in 0..tasks {
            out[run * tasks + task] = scores[ri[(rep * runs + run) * tasks + task] as usize
                * tasks
                + ti[rep * tasks + task] as usize];
        }
    }
}

fn as_array2<'py>(
    py: Python<'py>,
    values: Vec<Vec<f64>>,
    columns: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let rows = values.len();
    let flat: Vec<f64> = values.into_iter().flatten().collect();
    if flat.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("nonfinite computed statistic"));
    }
    Ok(Array2::from_shape_vec((rows, columns), flat)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .into_pyarray(py))
}

#[allow(clippy::too_many_arguments)]
fn replicate(
    scores: &[f64],
    runs: usize,
    tasks: usize,
    task_bootstrap: bool,
    seed: u64,
    rep: usize,
    scratch: &mut [f64],
) -> f64 {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(rep as u64);
    for task in 0..tasks {
        let source_task = if task_bootstrap {
            rng.gen_range(0..tasks)
        } else {
            task
        };
        for run in 0..runs {
            scratch[run * tasks + task] = scores[rng.gen_range(0..runs) * tasks + source_task];
        }
    }
    iqm(scratch)
}

fn percentile_bounds(results: &mut [f64], confidence: f64) -> PyResult<[f64; 2]> {
    if results.len() < 2
        || !(0.0 < confidence && confidence < 1.0)
        || results.iter().any(|v| !v.is_finite())
    {
        return Err(PyValueError::new_err("invalid percentile input"));
    }
    results.sort_unstable_by(f64::total_cmp);
    let quantile = |q: f64| {
        let position = (results.len() - 1) as f64 * q;
        let lower = position.floor() as usize;
        let upper = position.ceil() as usize;
        let weight = position - lower as f64;
        results[lower] * (1.0 - weight) + results[upper] * weight
    };
    let alpha = (1.0 - confidence) / 2.0;
    let bounds = [quantile(alpha), quantile(1.0 - alpha)];
    if bounds.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("nonfinite computed interval"));
    }
    Ok(bounds)
}

#[pyfunction]
fn fixed_percentile(mut values: Vec<f64>, confidence: f64) -> PyResult<[f64; 2]> {
    percentile_bounds(&mut values, confidence)
}

#[pyfunction]
fn interval(
    py: Python<'_>,
    scores: PyReadonlyArray2<'_, f64>,
    reps: usize,
    confidence: f64,
    task_bootstrap: bool,
    seed: u64,
    threads: usize,
) -> PyResult<(f64, [f64; 2])> {
    if reps < 2 || threads == 0 || !(0.0 < confidence && confidence < 1.0) {
        return Err(PyValueError::new_err(
            "invalid reps, threads, or confidence",
        ));
    }
    let (scores, runs, tasks) = owned_scores(scores)?;
    // Owned scores cross the detach boundary: https://pyo3.rs/v0.29.2/parallelism.html
    py.detach(move || {
        let point = iqm(&mut scores.clone());
        let mut results: Vec<f64> = if threads == 1 {
            let mut scratch = vec![0.0; scores.len()];
            (0..reps)
                .map(|rep| {
                    replicate(
                        &scores,
                        runs,
                        tasks,
                        task_bootstrap,
                        seed,
                        rep,
                        &mut scratch,
                    )
                })
                .collect()
        } else {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            pool.install(|| {
                (0..reps)
                    .into_par_iter()
                    .map_init(
                        || vec![0.0; scores.len()],
                        |scratch, rep| {
                            replicate(&scores, runs, tasks, task_bootstrap, seed, rep, scratch)
                        },
                    )
                    .collect()
            })
        };
        if !point.is_finite() || results.iter().any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err("nonfinite computed IQM"));
        }
        let bounds = percentile_bounds(&mut results, confidence)?;
        Ok((point, bounds))
    })
}

#[pyfunction]
fn fixed_iqm<'py>(
    py: Python<'py>,
    scores: PyReadonlyArray2<'py, f64>,
    run_indices: PyReadonlyArray3<'py, u64>,
    task_indices: PyReadonlyArray2<'py, u64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (scores, runs, tasks) = owned_scores(scores)?;
    let ri = run_indices.as_array();
    let ti = task_indices.as_array();
    let reps = ri.dim().0;
    if reps == 0
        || ri.dim() != (reps, runs, tasks)
        || ti.dim() != (reps, tasks)
        || ri.iter().any(|&r| r >= runs as u64)
        || ti.iter().any(|&t| t >= tasks as u64)
    {
        return Err(PyValueError::new_err("invalid fixed resampling indices"));
    }
    let ri: Vec<u64> = ri.iter().copied().collect();
    let ti: Vec<u64> = ti.iter().copied().collect();
    let results: Vec<f64> = py.detach(move || {
        let mut scratch = vec![0.0; scores.len()];
        (0..reps)
            .map(|rep| {
                for run in 0..runs {
                    for task in 0..tasks {
                        scratch[run * tasks + task] =
                            scores[ri[(rep * runs + run) * tasks + task] as usize * tasks
                                + ti[rep * tasks + task] as usize];
                    }
                }
                iqm(&mut scratch)
            })
            .collect()
    });
    if results.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("nonfinite computed IQM"));
    }
    Ok(results.into_pyarray(py))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn aggregate(
    py: Python<'_>,
    scores: PyReadonlyArray2<'_, f64>,
    metrics: Vec<String>,
    reps: usize,
    confidence: f64,
    task_bootstrap: bool,
    seed: u64,
    threads: usize,
    gamma: f64,
) -> PyResult<(Vec<f64>, Vec<Vec<f64>>)> {
    check_options(reps, confidence, threads)?;
    let metrics = parse_metrics(metrics, gamma)?;
    let (scores, runs, tasks) = owned_scores(scores)?;
    py.detach(move || {
        let point = metric_values(&scores, runs, tasks, &metrics, gamma, &mut Vec::new())?;
        let compute = |rep: usize, state: &mut (Vec<f64>, Vec<f64>)| {
            sample(
                &scores,
                runs,
                tasks,
                task_bootstrap,
                seed,
                rep,
                &mut state.0,
            );
            metric_values(&state.0, runs, tasks, &metrics, gamma, &mut state.1)
        };
        let values = if threads == 1 {
            let mut state = (vec![0.0; scores.len()], Vec::new());
            (0..reps)
                .map(|rep| compute(rep, &mut state))
                .collect::<PyResult<_>>()?
        } else {
            pool(threads)?.install(|| {
                (0..reps)
                    .into_par_iter()
                    .map_init(
                        || (vec![0.0; scores.len()], Vec::new()),
                        |state, rep| compute(rep, state),
                    )
                    .collect::<PyResult<_>>()
            })?
        };
        intervals(point, values, confidence)
    })
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn profile(
    py: Python<'_>,
    scores: PyReadonlyArray2<'_, f64>,
    thresholds: Vec<f64>,
    reps: usize,
    confidence: f64,
    task_bootstrap: bool,
    seed: u64,
    threads: usize,
    use_score_distribution: bool,
) -> PyResult<(Vec<f64>, Vec<Vec<f64>>)> {
    check_options(reps, confidence, threads)?;
    if thresholds.is_empty() || thresholds.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err(
            "thresholds must be nonempty and finite",
        ));
    }
    let (scores, runs, tasks) = owned_scores(scores)?;
    py.detach(move || {
        let point = profile_values(&scores, runs, tasks, &thresholds, use_score_distribution)?;
        let compute = |rep: usize, scratch: &mut Vec<f64>| {
            sample(&scores, runs, tasks, task_bootstrap, seed, rep, scratch);
            profile_values(scratch, runs, tasks, &thresholds, use_score_distribution)
        };
        let values = if threads == 1 {
            let mut scratch = vec![0.0; scores.len()];
            (0..reps)
                .map(|rep| compute(rep, &mut scratch))
                .collect::<PyResult<_>>()?
        } else {
            pool(threads)?.install(|| {
                (0..reps)
                    .into_par_iter()
                    .map_init(
                        || vec![0.0; scores.len()],
                        |scratch, rep| compute(rep, scratch),
                    )
                    .collect::<PyResult<_>>()
            })?
        };
        intervals(point, values, confidence)
    })
}

#[pyfunction]
fn compare(
    py: Python<'_>,
    x: PyReadonlyArray2<'_, f64>,
    y: PyReadonlyArray2<'_, f64>,
    reps: usize,
    confidence: f64,
    seed: u64,
    threads: usize,
) -> PyResult<(Vec<f64>, Vec<Vec<f64>>)> {
    check_options(reps, confidence, threads)?;
    let (x, runs_x, tasks) = owned_scores(x)?;
    let (y, runs_y, tasks_y) = owned_scores(y)?;
    if tasks != tasks_y {
        return Err(PyValueError::new_err(
            "score arrays must have the same number of tasks",
        ));
    }
    py.detach(move || {
        let point = vec![improvement(&x, &y, runs_x, runs_y, tasks)];
        let compute = |rep: usize, state: &mut (Vec<f64>, Vec<f64>)| {
            sample(&x, runs_x, tasks, false, seed, rep, &mut state.0);
            sample(
                &y,
                runs_y,
                tasks,
                false,
                seed ^ 0x9e3779b97f4a7c15,
                rep,
                &mut state.1,
            );
            vec![improvement(&state.0, &state.1, runs_x, runs_y, tasks)]
        };
        let values = if threads == 1 {
            let mut state = (vec![0.0; x.len()], vec![0.0; y.len()]);
            (0..reps).map(|rep| compute(rep, &mut state)).collect()
        } else {
            pool(threads)?.install(|| {
                (0..reps)
                    .into_par_iter()
                    .map_init(
                        || (vec![0.0; x.len()], vec![0.0; y.len()]),
                        |state, rep| compute(rep, state),
                    )
                    .collect()
            })
        };
        intervals(point, values, confidence)
    })
}

#[pyfunction]
fn fixed_aggregates<'py>(
    py: Python<'py>,
    scores: PyReadonlyArray2<'py, f64>,
    run_indices: PyReadonlyArray3<'py, u64>,
    task_indices: PyReadonlyArray2<'py, u64>,
    metrics: Vec<String>,
    gamma: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let metrics = parse_metrics(metrics, gamma)?;
    let columns = metrics.len();
    let (scores, runs, tasks) = owned_scores(scores)?;
    let (ri, ti, reps) = fixed_indices(run_indices, task_indices, runs, tasks)?;
    let values = py.detach(move || {
        let mut sampled = vec![0.0; scores.len()];
        let mut scratch = Vec::new();
        (0..reps)
            .map(|rep| {
                fixed_sample(&scores, runs, tasks, &ri, &ti, rep, &mut sampled);
                metric_values(&sampled, runs, tasks, &metrics, gamma, &mut scratch)
            })
            .collect::<PyResult<_>>()
    })?;
    as_array2(py, values, columns)
}

#[pyfunction]
fn fixed_profile<'py>(
    py: Python<'py>,
    scores: PyReadonlyArray2<'py, f64>,
    run_indices: PyReadonlyArray3<'py, u64>,
    task_indices: PyReadonlyArray2<'py, u64>,
    thresholds: Vec<f64>,
    use_score_distribution: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if thresholds.is_empty() || thresholds.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err(
            "thresholds must be nonempty and finite",
        ));
    }
    let columns = thresholds.len();
    let (scores, runs, tasks) = owned_scores(scores)?;
    let (ri, ti, reps) = fixed_indices(run_indices, task_indices, runs, tasks)?;
    let values = py.detach(move || {
        let mut sampled = vec![0.0; scores.len()];
        (0..reps)
            .map(|rep| {
                fixed_sample(&scores, runs, tasks, &ri, &ti, rep, &mut sampled);
                profile_values(&sampled, runs, tasks, &thresholds, use_score_distribution)
            })
            .collect::<PyResult<_>>()
    })?;
    as_array2(py, values, columns)
}

#[pyfunction]
fn fixed_compare<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
    ri_x: PyReadonlyArray3<'py, u64>,
    ri_y: PyReadonlyArray3<'py, u64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (x, runs_x, tasks) = owned_scores(x)?;
    let (y, runs_y, tasks_y) = owned_scores(y)?;
    if tasks != tasks_y {
        return Err(PyValueError::new_err(
            "score arrays must have the same number of tasks",
        ));
    }
    let ix = ri_x.as_array();
    let iy = ri_y.as_array();
    let reps = ix.dim().0;
    if reps == 0
        || ix.dim() != (reps, runs_x, tasks)
        || iy.dim() != (reps, runs_y, tasks)
        || ix.iter().any(|&r| r >= runs_x as u64)
        || iy.iter().any(|&r| r >= runs_y as u64)
    {
        return Err(PyValueError::new_err("invalid fixed resampling indices"));
    }
    let ix: Vec<u64> = ix.iter().copied().collect();
    let iy: Vec<u64> = iy.iter().copied().collect();
    let values = py.detach(move || {
        let mut sx = vec![0.0; x.len()];
        let mut sy = vec![0.0; y.len()];
        (0..reps)
            .map(|rep| {
                for run in 0..runs_x {
                    for task in 0..tasks {
                        sx[run * tasks + task] =
                            x[ix[(rep * runs_x + run) * tasks + task] as usize * tasks + task];
                    }
                }
                for run in 0..runs_y {
                    for task in 0..tasks {
                        sy[run * tasks + task] =
                            y[iy[(rep * runs_y + run) * tasks + task] as usize * tasks + task];
                    }
                }
                vec![improvement(&sx, &sy, runs_x, runs_y, tasks)]
            })
            .collect()
    });
    as_array2(py, values, 1)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(interval, m)?)?;
    m.add_function(wrap_pyfunction!(fixed_iqm, m)?)?;
    m.add_function(wrap_pyfunction!(fixed_percentile, m)?)?;
    m.add_function(wrap_pyfunction!(aggregate, m)?)?;
    m.add_function(wrap_pyfunction!(profile, m)?)?;
    m.add_function(wrap_pyfunction!(compare, m)?)?;
    m.add_function(wrap_pyfunction!(fixed_aggregates, m)?)?;
    m.add_function(wrap_pyfunction!(fixed_profile, m)?)?;
    m.add_function(wrap_pyfunction!(fixed_compare, m)?)?;
    Ok(())
}
