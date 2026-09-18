//! Historical probe of the original full-sort IQM loops in benchmarks/baseline/lib.rs.
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::time::{Duration, Instant};

fn main() {
    let bytes = include_bytes!("../benchmarks/data/atari_spr.npy");
    assert_eq!(&bytes[..8], b"\x93NUMPY\x01\0");
    let offset = 10 + u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let scores: Vec<f64> = bytes[offset..]
        .as_chunks::<8>()
        .0
        .iter()
        .map(|chunk| f64::from_le_bytes(*chunk))
        .collect();
    assert_eq!(scores.len(), 100 * 26);
    let (runs, tasks, reps) = (100, 26, 50_000);
    let mut scratch = vec![0.0; scores.len()];
    let mut sample_time = Duration::ZERO;
    let mut sort_time = Duration::ZERO;
    let mut checksum = 0.0;
    for rep in 0..reps {
        let start = Instant::now();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        rng.set_stream(rep as u64);
        for task in 0..tasks {
            for run in 0..runs {
                scratch[run * tasks + task] = scores[rng.gen_range(0..runs) * tasks + task];
            }
        }
        sample_time += start.elapsed();
        let start = Instant::now();
        scratch.sort_unstable_by(f64::total_cmp);
        let trim = scratch.len() / 4;
        checksum += scratch[trim..scratch.len() - trim].iter().sum::<f64>()
            / (scratch.len() - 2 * trim) as f64;
        sort_time += start.elapsed();
    }
    println!(
        "rng_sampling_seconds={} sorting_iqm_seconds={} checksum={checksum}",
        sample_time.as_secs_f64(),
        sort_time.as_secs_f64()
    );
}
