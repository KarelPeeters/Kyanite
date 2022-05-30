use std::future::Future;

use flume::{Receiver, Sender};
use futures::FutureExt;

use kz_util::sequence::VecExtSingle;

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
    pub x: Vec<X>,
    pub sender: Sender<Vec<Y>>,
}

pub fn job_pair<X, Y>(cap: usize) -> (JobClient<X, Y>, JobServer<X, Y>) {
    let (sender, receiver) = flume::bounded(cap);

    let client = JobClient { sender };
    let server = JobServer { receiver };

    (client, server)
}

impl<X, Y: 'static> JobClient<X, Y> {
    pub fn map(&self, x: Vec<X>) -> Receiver<Vec<Y>> {
        let (sender, receiver) = flume::bounded(1);

        if x.len() == 0 {
            // easy short-circuit case that avoids additional channel communication
            let _ = sender.send(vec![]);
        } else {
            let item = Job { x, sender };
            self.sender.send(item).unwrap();
        }

        receiver
    }

    pub fn map_blocking(&self, x: Vec<X>) -> Vec<Y> {
        self.map(x).recv().unwrap()
    }

    pub fn map_async(&self, x: Vec<X>) -> impl Future<Output = Vec<Y>> {
        self.map(x).into_recv_async().map(Result::unwrap)
    }

    pub fn map_async_single(&self, x: X) -> impl Future<Output = Y> {
        // Universal Function Call Syntax to help IDE type inference
        FutureExt::map(self.map_async(vec![x]), |y: Vec<Y>| y.single().unwrap())
    }
}

impl<X, Y> JobServer<X, Y> {
    pub fn receiver(&self) -> &Receiver<Job<X, Y>> {
        &self.receiver
    }

    pub fn into_receiver(self) -> Receiver<Job<X, Y>> {
        self.receiver
    }
}

// implement clone for any X, Y
impl<X, Y> Clone for JobClient<X, Y> {
    fn clone(&self) -> Self {
        JobClient {
            sender: self.sender.clone(),
        }
    }
}

// implement clone for any X, Y
impl<X, Y> Clone for JobServer<X, Y> {
    fn clone(&self) -> Self {
        JobServer {
            receiver: self.receiver.clone(),
        }
    }
}
