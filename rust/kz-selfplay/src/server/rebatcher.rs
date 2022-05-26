use itertools::{Itertools, zip};

use kz_core::network::job_channel::{Job, job_pair, JobClient, JobServer};

pub struct Rebatcher<X, Y> {
    batch_size_in: usize,
    batch_size_out: usize,

    server_in: JobServer<Vec<X>, Vec<Y>>,
    client_out: JobClient<Vec<X>, Vec<Y>>,
}

impl<X, Y: 'static> Rebatcher<X, Y> {
    pub fn new(
        capacity: usize,
        batch_size_in: usize,
        batch_size_out: usize,
    ) -> (Rebatcher<X, Y>, JobClient<Vec<X>, Vec<Y>>, JobServer<Vec<X>, Vec<Y>>) {
        // TODO remove this constraint by allowing buffer carryover
        assert_eq!(
            batch_size_out % batch_size_in,
            0,
            "Output batch size {} should be multiple of input {}",
            batch_size_out,
            batch_size_in
        );

        // TODO where should the capacity be? we shouldn't really need both of them
        let (client_in, server_in) = job_pair(capacity);
        let (client_out, server_out) = job_pair(capacity);

        let rebatcher = Rebatcher {
            batch_size_in,
            batch_size_out,
            server_in,
            client_out,
        };

        (rebatcher, client_in, server_out)
    }

    pub fn run_loop(self) {
        loop {
            let mut all_x = vec![];
            let mut senders = vec![];

            // collect inputs
            while all_x.len() < self.batch_size_out {
                let Job { mut x, sender } = self.server_in.receiver().recv().unwrap();
                assert_eq!(x.len(), self.batch_size_in);
                all_x.append(&mut x);
                senders.push(sender);
            }

            // map
            assert_eq!(all_x.len(), self.batch_size_out);
            let mut all_y = self.client_out.map_blocking(all_x);

            // distribute outputs
            let all_y_split = all_y.drain(..).chunks(self.batch_size_in);
            for (sender, y) in zip(senders, &all_y_split) {
                sender.send(y.collect_vec()).unwrap();
            }
        }
    }
}
