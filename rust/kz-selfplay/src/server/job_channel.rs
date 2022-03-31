use crossbeam::channel::{Receiver, Sender};

#[derive(Debug)]
pub struct JobClient<X, Y> {
    sender: Sender<Job<X, Y>>,
}

#[derive(Debug)]
pub struct JobServer<X, Y> {
    receiver: Receiver<Job<X, Y>>,
}

#[derive(Debug)]
pub struct Job<X, Y> {
    pub x: X,
    pub sender: Sender<Y>,
}

impl<X, Y> Job<X, Y> {
    pub fn run(self, mut f: impl FnMut(X) -> Y) {
        let Job { x, sender } = self;
        let y = f(x);
        sender.send(y).unwrap();
    }
}

pub fn job_pair<X, Y>(cap: usize) -> (JobClient<X, Y>, JobServer<X, Y>) {
    let (sender, receiver) = crossbeam::channel::bounded(cap);

    let client = JobClient { sender };
    let server = JobServer { receiver };

    (client, server)
}

impl<X, Y> JobClient<X, Y> {
    pub fn map(&self, x: X) -> Receiver<Y> {
        let (sender, receiver) = crossbeam::channel::bounded(1);
        let item = Job { x, sender };
        self.sender.send(item).unwrap();
        receiver
    }

    pub fn map_blocking(&self, x: X) -> Y {
        self.map(x).recv().unwrap()
    }
}

impl<X, Y> JobServer<X, Y> {
    /// Get the underlying receiver, useful for manually implementing the server loop.
    pub fn receiver(&self) -> &Receiver<Job<X, Y>> {
        &self.receiver
    }

    pub fn run_loop(&self, mut f: impl FnMut(X) -> Y) {
        loop {
            let job = self.receiver.recv().unwrap();
            job.run(&mut f);
        }
    }
}

impl<X, Y> Clone for JobClient<X, Y> {
    fn clone(&self) -> Self {
        JobClient {
            sender: self.sender.clone(),
        }
    }
}

impl<X, Y> Clone for JobServer<X, Y> {
    fn clone(&self) -> Self {
        JobServer {
            receiver: self.receiver.clone(),
        }
    }
}
