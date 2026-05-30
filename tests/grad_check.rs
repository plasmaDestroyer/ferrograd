//! Finite-difference gradient-checking harness (Phase A0).
//!
//! `central_difference` is engine-agnostic: it numerically estimates the gradient of any
//! scalar function `f: R^n -> R` via the central difference
//!     (f(x + eps*e_i) - f(x - eps*e_i)) / (2*eps).
//!
//! Once the Tensor/Tape API exists, each op test will:
//!   1. build input tensors from `x`,
//!   2. run the op forward to a SCALAR loss,
//!   3. call `.backward()` and read the analytic grad of the inputs,
//!   4. compare against `central_difference(|x| forward_to_scalar(x), x, eps)`.
//!
//! Note on f32: finite differences are noisy. Use eps ~ 1e-2..1e-3 and a tolerance around
//! 1e-3. If an op's grad check is borderline, accumulate reductions (sum/mean) in f64.

/// Numerically estimate the gradient of `f` at `x` using central differences.
pub fn central_difference<F: Fn(&[f32]) -> f32>(f: F, x: &[f32], eps: f32) -> Vec<f32> {
    let mut grad = vec![0.0f32; x.len()];
    let mut probe = x.to_vec();
    for i in 0..x.len() {
        let orig = probe[i];
        probe[i] = orig + eps;
        let f_plus = f(&probe);
        probe[i] = orig - eps;
        let f_minus = f(&probe);
        probe[i] = orig; // restore
        grad[i] = (f_plus - f_minus) / (2.0 * eps);
    }
    grad
}

/// Largest absolute difference between two equal-length slices (the grad-check metric).
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch in max_abs_diff");
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max)
}

// --- Self-test: the harness's own numeric core, verified BEFORE any engine exists. ---
#[test]
fn central_difference_matches_known_gradient() {
    // f(x) = sum(x_i^2)  =>  grad_i = 2*x_i.  (x^2 has zero truncation error for central
    // differences since its third derivative is 0, so this isolates rounding.)
    let f = |x: &[f32]| x.iter().map(|v| v * v).sum::<f32>();
    let x = [1.0_f32, -2.0, 3.0, 0.5];
    let numeric = central_difference(f, &x, 1e-2);
    let analytic: Vec<f32> = x.iter().map(|v| 2.0 * v).collect();
    let err = max_abs_diff(&numeric, &analytic);
    assert!(err < 1e-3, "central_difference off by {err}: {numeric:?} vs {analytic:?}");
}

// TODO(A1): once Tensor/Tape exist, add per-op grad checks here, e.g.
//   #[test] fn add_grad() { ... build a,b ; c = a + b ; loss = c.sum() ; loss.backward();
//   compare a.grad()/b.grad() to central_difference of the same forward. }
