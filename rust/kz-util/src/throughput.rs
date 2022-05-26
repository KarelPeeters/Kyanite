use std::time::Instant;

#[derive(Debug)]
pub struct PrintThroughput {
    name: String,
    total_count: u64,
    delta_count: u64,
    update_count: u64,
    last_print: Instant,
}

impl PrintThroughput {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            total_count: 0,
            delta_count: 0,
            update_count: 0,
            last_print: Instant::now(),
        }
    }

    pub fn total_count(&self) -> u64 {
        self.total_count
    }

    pub fn update_delta(&mut self, count: u64) -> bool {
        self.delta_count += count;
        self.total_count += count;
        self.update_count += 1;

        let now = Instant::now();
        let delta = now - self.last_print;

        let print = delta.as_secs() >= 1 && self.update_count >= 10;
        if print {
            self.print_tp(now);
        }
        print
    }

    pub fn update_total(&mut self, count: u64) -> bool {
        assert!(count >= self.total_count, "Count must be increasing");
        self.update_delta(count - self.total_count)
    }

    fn print_tp(&mut self, now: Instant) {
        let delta = now - self.last_print;
        let throughput = self.delta_count as f32 / delta.as_secs_f32();
        println!(
            "{:.3} {}/s => {:.3} {}",
            throughput, self.name, self.total_count, self.name
        );

        self.last_print = now;
        self.delta_count = 0;
        self.update_count = 0;
    }
}

impl Drop for PrintThroughput {
    fn drop(&mut self) {
        self.print_tp(Instant::now());
    }
}
