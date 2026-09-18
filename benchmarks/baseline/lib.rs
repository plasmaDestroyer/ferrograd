pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iqm_trims_floor_quarters() {
        assert_eq!(iqm(&mut [1.0, 2.0, 3.0]), 2.0);
        assert_eq!(iqm(&mut [-100.0, 2.0, 4.0, 100.0]), 3.0);
        assert_eq!(iqm(&mut [-100.0, 2.0, 4.0, 6.0, 100.0]), 4.0);
    }

    #[test]
    fn percentile_uses_linear_interpolation() {
        let bounds = percentile_bounds(&mut [10.0, 0.0, 30.0, 20.0], 0.5).unwrap();
        assert_eq!(bounds, [7.5, 22.5]);
    }

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::{exceptions::PyValueError, prelude::*};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

fn iqm(values: &mut [f64]) -> f64 {
    values.sort_unstable_by(f64::total_cmp);
    let trim = values.len() / 4;
    let middle = &values[trim..values.len() - trim];
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

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(interval, m)?)?;
    m.add_function(wrap_pyfunction!(fixed_iqm, m)?)?;
    m.add_function(wrap_pyfunction!(fixed_percentile, m)?)?;
    Ok(())
}
