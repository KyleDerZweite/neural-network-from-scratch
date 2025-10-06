use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

#[derive(Default, Clone)]
struct ProfileStat {
    calls: u64,
    total: Duration,
    max: Duration,
}

static PROFILER: Lazy<Mutex<HashMap<&'static str, ProfileStat>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

pub struct ProfileGuard {
    label: &'static str,
    start: Instant,
}

impl ProfileGuard {
    pub fn new(label: &'static str) -> Self {
        Self {
            label,
            start: Instant::now(),
        }
    }
}

impl Drop for ProfileGuard {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        let mut profiler = PROFILER.lock().expect("profiling mutex poisoned");
        let stat = profiler.entry(self.label).or_default();
        stat.calls += 1;
        stat.total += elapsed;
        if elapsed > stat.max {
            stat.max = elapsed;
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProfileEntry {
    pub label: &'static str,
    pub calls: u64,
    pub total: Duration,
    pub max: Duration,
}

pub fn snapshot() -> Vec<ProfileEntry> {
    let profiler = PROFILER.lock().expect("profiling mutex poisoned");
    let mut entries: Vec<ProfileEntry> = profiler
        .iter()
        .map(|(&label, stat)| ProfileEntry {
            label,
            calls: stat.calls,
            total: stat.total,
            max: stat.max,
        })
        .collect();

    entries.sort_by(|a, b| b.total.cmp(&a.total));
    entries
}

pub fn clear() {
    PROFILER
        .lock()
        .expect("profiling mutex poisoned")
        .clear();
}

pub fn print_report() {
    let entries = snapshot();
    if entries.is_empty() {
        println!("\n[profiling] no measurements collected");
        return;
    }

    println!("\n[profiling] runtime hotspots:");
    println!("{:<28} {:>12} {:>16} {:>16}", "label", "calls", "total (ms)", "avg (µs)");

    for entry in entries {
        let total_ms = entry.total.as_secs_f64() * 1_000.0;
        let avg_us = if entry.calls > 0 {
            entry.total.as_secs_f64() * 1_000_000.0 / entry.calls as f64
        } else {
            0.0
        };
        println!(
            "{:<28} {:>12} {:>16.3} {:>16.1}",
            entry.label,
            entry.calls,
            total_ms,
            avg_us
        );
    }
}
