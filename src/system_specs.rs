//! Runtime system inspection — used to auto-tune concurrency knobs that
//! depend on local hardware (today: `RENDER_CONCURRENCY`).
//!
//! Deliberately dependency-free: `std::thread::available_parallelism()` for
//! CPU count and `/proc/meminfo` parsing for memory. Adding `sysinfo` would
//! double the build's transitive crate count for one number.

use std::thread::available_parallelism;

/// Snapshot of the local machine relevant to clipper throughput.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct SystemSpecs {
    /// Logical cores reported by the OS (i.e. hyperthreads, not physical cores).
    pub logical_cores: usize,
    /// Total system memory in MiB. `None` on platforms where we couldn't
    /// determine it (non-Linux, or `/proc/meminfo` unreadable). Render
    /// auto-tuning treats `None` as "assume enough" — RAM is rarely the
    /// bottleneck for x264 encodes anyway.
    pub total_mem_mib: Option<u64>,
}

impl SystemSpecs {
    /// Inspect the running system. Cheap (~no syscalls beyond two reads),
    /// safe to call once at startup or on every API request.
    pub fn detect() -> Self {
        let logical_cores = available_parallelism().map(|n| n.get()).unwrap_or(1);
        let total_mem_mib = read_total_mem_mib();
        Self {
            logical_cores,
            total_mem_mib,
        }
    }
}

/// Auto-pick render concurrency from the detected specs.
///
/// Heuristic: an `ffmpeg -preset medium` 1080p encode pegs ~3-4 effective
/// cores, so divide logical cores by 4 and floor at 1. Cap at 8 because past
/// that point disk-IO + L3 contention dominates and you stop seeing wall-clock
/// wins. On a 32-core box this yields 8; on a 4-core laptop, 1.
///
/// Returned value is used directly as the `buffer_unordered` width over the
/// per-variant render stream — see [`crate::clipper`] render sites.
pub fn auto_render_concurrency(specs: &SystemSpecs) -> usize {
    (specs.logical_cores / 4).clamp(1, 8)
}

/// Read `MemTotal:` from `/proc/meminfo` and return its value in MiB.
/// Returns `None` on any parse failure or non-Linux platforms.
fn read_total_mem_mib() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let txt = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in txt.lines() {
            if let Some(rest) = line.strip_prefix("MemTotal:") {
                // Format: "MemTotal:       131088552 kB"
                let kb: u64 = rest
                    .trim()
                    .split_whitespace()
                    .next()?
                    .parse()
                    .ok()?;
                return Some(kb / 1024);
            }
        }
        None
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_returns_at_least_one_core() {
        let s = SystemSpecs::detect();
        assert!(s.logical_cores >= 1);
    }

    #[test]
    fn auto_concurrency_clamps_low() {
        let s = SystemSpecs {
            logical_cores: 2,
            total_mem_mib: Some(8192),
        };
        assert_eq!(auto_render_concurrency(&s), 1);
    }

    #[test]
    fn auto_concurrency_scales_with_cores() {
        let s = SystemSpecs {
            logical_cores: 16,
            total_mem_mib: Some(32_768),
        };
        assert_eq!(auto_render_concurrency(&s), 4);
    }

    #[test]
    fn auto_concurrency_caps_high() {
        let s = SystemSpecs {
            logical_cores: 128,
            total_mem_mib: Some(1_048_576),
        };
        assert_eq!(auto_render_concurrency(&s), 8);
    }
}
