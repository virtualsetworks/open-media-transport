//! The threads that code a frame's slices in parallel.
//!
//! Without the `rayon` feature every frame starts its workers afresh
//! (`std::thread::scope`), as libvmx's own fallback does. At 60 frames a
//! second and four workers that is 240 thread starts a second per stream,
//! which costs CPU and keeps the processor from idling between frames. With
//! the feature, each encoder or decoder keeps a rayon pool of its own size
//! for its whole life, and a frame only hands its jobs to it. Output is the
//! same either way.

/// Runs a frame's jobs, on this thread for one worker or on a pool.
pub(crate) struct Workers {
    n: usize,
    #[cfg(feature = "rayon")]
    pool: Option<rayon::ThreadPool>,
}

impl Workers {
    /// `n` workers; 1 or less codes on the calling thread.
    pub(crate) fn new(n: usize) -> Self {
        let n = n.max(1);
        Self {
            n,
            // A pool that cannot start falls back to scoped threads.
            #[cfg(feature = "rayon")]
            pool: if n > 1 {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .thread_name(|i| format!("vmx-{i}"))
                    .build()
                    .ok()
            } else {
                None
            },
        }
    }

    /// The number of workers.
    pub(crate) fn n(&self) -> usize {
        self.n
    }

    /// Runs every job, in parallel when there is more than one worker, and
    /// returns their results in order. A panic in a job is passed on.
    pub(crate) fn run<F, R>(&self, jobs: Vec<F>) -> Vec<R>
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        if self.n <= 1 || jobs.len() < 2 {
            return jobs.into_iter().map(|f| f()).collect();
        }
        #[cfg(feature = "rayon")]
        if let Some(pool) = &self.pool {
            use rayon::prelude::*;
            return pool.install(|| jobs.into_par_iter().map(|f| f()).collect());
        }
        std::thread::scope(|scope| {
            let handles: Vec<_> = jobs.into_iter().map(|f| scope.spawn(f)).collect();
            handles.into_iter().map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e))).collect()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn results_come_back_in_order_for_any_number_of_workers() {
        for n in [0, 1, 2, 4, 9] {
            let w = Workers::new(n);
            let jobs: Vec<_> = (0..7).map(|i| move || i * 10).collect();
            assert_eq!(w.run(jobs), vec![0, 10, 20, 30, 40, 50, 60], "{n} workers");
        }
    }

    #[test]
    fn jobs_may_borrow_from_the_caller() {
        let mut rows = vec![vec![0u8; 4]; 3];
        let w = Workers::new(3);
        let jobs: Vec<_> = rows.iter_mut().enumerate().map(|(i, r)| move || r.fill(i as u8)).collect();
        w.run(jobs);
        assert_eq!(rows, vec![vec![0; 4], vec![1; 4], vec![2; 4]]);
    }

    /// Runs without libvmx: frame after frame, through workers kept between
    /// frames and through a change of their number, the bytes stay the same.
    #[test]
    fn workers_kept_between_frames_code_the_same_bytes_as_one_thread() {
        use crate::{Decoder, Encoder, EncoderConfig, Frame, PixelFormat};
        let (w, h) = (1920, 1080);
        let mut one = Encoder::new(EncoderConfig::new(w, h)).unwrap();
        let mut cfg = EncoderConfig::new(w, h);
        cfg.threads = 4;
        let mut many = Encoder::new(cfg).unwrap();
        let mut d1 = Decoder::new(w, h).unwrap();
        let mut d4 = Decoder::new(w, h).unwrap();
        d4.set_threads(4);
        for t in 0..4u32 {
            let mut frame = Frame::new(w, h, PixelFormat::Uyvy);
            for (i, b) in frame.planes[0].data.iter_mut().enumerate() {
                *b = ((i as u32).wrapping_mul(2654435761).rotate_left(t) >> 24) as u8 | (i % 7) as u8;
            }
            if t == 2 {
                many.set_threads(3);
                d4.set_threads(2);
            }
            let a = one.encode(&frame).unwrap();
            assert_eq!(a, many.encode(&frame).unwrap(), "frame {t} encoded");
            assert_eq!(d1.decode(&a, PixelFormat::Uyvy).unwrap(), d4.decode(&a, PixelFormat::Uyvy).unwrap(), "frame {t} decoded");
        }
    }
}
