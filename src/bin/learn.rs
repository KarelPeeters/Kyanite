use std::cmp::{max, min};

use derive_more::From;
use itertools::Itertools;
use mnist::MnistBuilder;
use ndarray::{Array2, azip};
use ndarray_rand::RandomExt;
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand_distr::{Distribution, StandardNormal};

//TODO remove mutable state from layers
//  have a NetworkBuilder that gets a list of layers and checks them, then build a TrainNetwork instance
//  with state for the backpropagation
//  there is also a Network state that only does calculation, no actual training
//  maybe allow multiple inputs and outputs, how though?
#[derive(From, Debug)]
enum Layer {
    Relu(ReluLayer),
    Sigmoid(SigmoidLayer),
    Dense(DenseLayer),
}

impl Propagate for Layer {
    fn forward(&mut self, input: Array2<f32>) -> Array2<f32> {
        match self {
            Layer::Relu(layer) => layer.forward(input),
            Layer::Sigmoid(layer) => layer.forward(input),
            Layer::Dense(layer) => layer.forward(input)
        }
    }

    fn clear_deltas(&mut self) {
        match self {
            Layer::Relu(layer) => layer.clear_deltas(),
            Layer::Sigmoid(layer) => layer.clear_deltas(),
            Layer::Dense(layer) => layer.clear_deltas()
        }
    }

    fn backwards(&mut self, output_deriv: Array2<f32>) -> Array2<f32> {
        match self {
            Layer::Relu(layer) => layer.backwards(output_deriv),
            Layer::Sigmoid(layer) => layer.backwards(output_deriv),
            Layer::Dense(layer) => layer.backwards(output_deriv)
        }
    }

    fn apply_deltas(&mut self, factor: f32) {
        match self {
            Layer::Relu(layer) => layer.apply_deltas(factor),
            Layer::Sigmoid(layer) => layer.apply_deltas(factor),
            Layer::Dense(layer) => layer.apply_deltas(factor)
        }
    }
}

trait Propagate {
    fn forward(&mut self, input: Array2<f32>) -> Array2<f32>;

    fn clear_deltas(&mut self);
    fn backwards(&mut self, output_deriv: Array2<f32>) -> Array2<f32>;
    fn apply_deltas(&mut self, factor: f32);
}

#[derive(Debug)]
struct SigmoidLayer {
    size: usize,
    input: Array2<f32>,
}

impl SigmoidLayer {
    fn new(size: usize) -> Self {
        SigmoidLayer {
            size,
            input: Array2::zeros((size, 1)),
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl Propagate for SigmoidLayer {
    fn forward(&mut self, input: Array2<f32>) -> Array2<f32> {
        input.map(|&x| sigmoid(x))
    }

    fn clear_deltas(&mut self) {}

    fn backwards(&mut self, output_deriv: Array2<f32>) -> Array2<f32> {
        let mut result = output_deriv;
        azip!((&i in &self.input, r in &mut result) *r *= sigmoid(i) * (1.0 - sigmoid(i)) );
        result
    }

    fn apply_deltas(&mut self, _factor: f32) {}
}

#[derive(Debug)]
struct ReluLayer {
    size: usize,
    input: Array2<f32>,
}

impl ReluLayer {
    fn new(size: usize) -> Self {
        ReluLayer {
            size,
            input: Array2::zeros((size, 1)),
        }
    }
}

const RELU_LEAK: f32 = 0.1;

impl Propagate for ReluLayer {
    fn forward(&mut self, input: Array2<f32>) -> Array2<f32> {
        debug_assert!(input.dim() == (self.size, 1));

        self.input = input;
        self.input.map(|&x| if x < 0.0 { RELU_LEAK * x } else { x })
    }

    fn clear_deltas(&mut self) {}

    fn backwards(&mut self, output_deriv: Array2<f32>) -> Array2<f32> {
        debug_assert!(output_deriv.dim() == (self.size, 1));

        let mut result = output_deriv;
        azip!((&i in &self.input, r in &mut result) if i < 0.0 { *r *= RELU_LEAK } );
        result
    }

    fn apply_deltas(&mut self, _factor: f32) {}
}

#[derive(Debug)]
struct DenseLayer {
    w: Array2<f32>,
    b: Array2<f32>,

    input: Array2<f32>,
    w_delta: Array2<f32>,
    b_delta: Array2<f32>,
}

impl DenseLayer {
    fn new<R: Rng>(input: usize, output: usize, rng: &mut R) -> Self {
        DenseLayer {
            w: Array2::random_using((output, input), StandardNormal, rng),
            b: Array2::random_using((output, 1), StandardNormal, rng),
            input: Array2::zeros((input, 1)),
            w_delta: Array2::zeros((output, input)),
            b_delta: Array2::zeros((output, 1)),
        }
    }
}

impl Propagate for DenseLayer {
    fn forward(&mut self, input: Array2<f32>) -> Array2<f32> {
        self.input = input;
        self.w.dot(&self.input) + &self.b
    }

    fn clear_deltas(&mut self) {
        self.w_delta.fill(0.0);
        self.b_delta.fill(0.0);
    }

    fn backwards(&mut self, output_deriv: Array2<f32>) -> Array2<f32> {
        let input_deriv = self.w.t().dot(&output_deriv);
        let weight_deriv = output_deriv.dot(&self.input.t());

        self.w_delta += &weight_deriv;
        self.b_delta += &output_deriv;

        input_deriv
    }

    fn apply_deltas(&mut self, factor: f32) {
        self.w -= &(&self.w_delta * factor);
        self.b -= &(&self.b_delta * factor);
    }
}

#[derive(Debug)]
struct Network {
    layers: Vec<Layer>
}

impl Propagate for Network {
    fn forward(&mut self, input: Array2<f32>) -> Array2<f32> {
        self.layers.iter_mut().fold(input, |state, layer| {
            layer.forward(state)
        })
    }

    fn clear_deltas(&mut self) {
        for layer in &mut self.layers {
            layer.clear_deltas()
        }
    }

    fn backwards(&mut self, output_deriv: Array2<f32>) -> Array2<f32> {
        self.layers.iter_mut().rev().fold(output_deriv, |state, layer| {
            layer.backwards(state)
        })
    }

    fn apply_deltas(&mut self, factor: f32) {
        for layer in &mut self.layers {
            layer.apply_deltas(factor)
        }
    }
}

struct QuadraticCost;

impl QuadraticCost {
    fn eval(actual: &Array2<f32>, expected: &Array2<f32>) -> (f32, Array2<f32>) {
        let deriv: Array2<f32> = actual - expected;
        let norm = deriv.fold(0.0, |acc, x| acc + x * x);
        (norm / 2.0, deriv)
    }
}

type Entry = (Array2<f32>, Array2<f32>);

fn train_network(network: &mut Network, epochs: usize, batch_size: usize, train_factor: f32, data: &[Entry]) {
    assert!(batch_size <= data.len(), "batch size larger than data length");

    let mut rng = SmallRng::from_entropy();

    for epoch in 0..epochs {
        print!("Epoch {} ... ", epoch);

        let mut total_cost = 0.0;
        network.clear_deltas();

        for (input, expected_output) in data.choose_multiple(&mut rng, batch_size) {
            let output = network.forward(input.clone());
            let (cost, output_deriv) = QuadraticCost::eval(&output, expected_output);
            total_cost += cost;
            network.backwards(output_deriv);
        }

        network.apply_deltas(train_factor / batch_size as f32);
        println!("done, cost: {}", total_cost / batch_size as f32);
    }
}

fn load_mnist() -> (Vec<Entry>, Vec<Entry>) {
    let mnist = MnistBuilder::new()
        .base_path("mnist")
        .finalize();

    fn convert(labels: &Vec<u8>, images: &Vec<u8>) -> Vec<Entry> {
        assert_eq!(labels.len() * 28 * 28, images.len());

        labels.iter().zip(images.chunks_exact(28 * 28)).map(|(&digit, image)| {
            let input = Array2::from_shape_fn((28 * 28, 1), |(i, _)| {
                image[i] as f32 / 255.0
            });
            let output = Array2::from_shape_fn((10, 1), |(i, _)| {
                (i as u8 == digit) as u8 as f32
            });
            (input, output)
        }).collect_vec()
    }

    (convert(&mnist.trn_lbl, &mnist.trn_img), convert(&mnist.tst_lbl, &mnist.tst_img))
}

fn _manual_test() {
    // let train_data = (0..1000).map(|_| {
    //     let input = StandardNormal.sample(&mut rng);
    //     let output = 3.0 + 6.0 * input;
    //     (Array2::from_elem((1, 1), input), Array2::from_elem((1, 1), output))
    // }).collect_vec();

    let mut rng = SmallRng::from_entropy();

    let mut network = Network {
        layers: vec![
            DenseLayer::new(1, 1, &mut rng).into(),
            ReluLayer::new(1).into(),
        ]
    };

    let input = Array2::from_elem((1, 1), 2.0);
    println!("input: {:?}", input);

    network.clear_deltas();

    let output = network.forward(input);
    println!("output: {:?}", output);
    let output_deriv = Array2::from_elem((1, 1), 0.5);
    println!("output_deriv: {:?}", output_deriv);

    let input_deriv = network.backwards(output_deriv);
    println!("input_deriv: {:?}", input_deriv);

    println!("network: {:#?}", network);

    network.apply_deltas(0.1);

    println!("network: {:#?}", network);
}

fn main() {
    let mut rng = SmallRng::from_entropy();

    let mut network = Network {
        layers: vec![
            DenseLayer::new(28*28, 128, &mut rng).into(),
            ReluLayer::new(128).into(),
            DenseLayer::new(128, 10, &mut rng).into(),
        ]
    };

    let (train_data, test_data) = load_mnist();

    train_network(&mut network, 100000, 100, 0.001, &train_data);

    let output = network.forward(train_data[1].0.clone());
    println!("output: {:?}", output)
}
