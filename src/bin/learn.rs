use std::io;
use std::io::Write;

use itertools::{Itertools, zip};
use mnist::MnistBuilder;
use ndarray::{IntoNdProducer, Zip};
use ndarray_rand::RandomExt;
use ordered_float::OrderedFloat;
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand_distr::{Normal, StandardNormal};

type Matrix = ndarray::Array2<f32>;
type Vector = ndarray::Array1<f32>;

type Entry = (Vector, Vector);

fn load_mnist() -> (Vec<Entry>, Vec<Entry>) {
    let mnist = MnistBuilder::new()
        .base_path("mnist")
        .finalize();

    fn convert(labels: &Vec<u8>, images: &Vec<u8>) -> Vec<Entry> {
        assert_eq!(labels.len() * 28 * 28, images.len());

        labels.iter().zip(images.chunks_exact(28 * 28)).map(|(&digit, image)| {
            let input = Vector::from_shape_fn(28 * 28, |i| {
                image[i] as f32 / 255.0
            });
            let output = Vector::from_shape_fn(10, |i| {
                (i as u8 == digit) as u8 as f32
            });
            (input, output)
        }).collect_vec()
    }

    (convert(&mnist.trn_lbl, &mnist.trn_img), convert(&mnist.tst_lbl, &mnist.tst_img))
}

trait CostFunc {
    fn eval(&self, expected: &Vector, actual: &Vector) -> (f32, Vector);
}

struct QuadraticCost;

impl CostFunc for QuadraticCost {
    fn eval(&self, expected: &Vector, actual: &Vector) -> (f32, Vector) {
        let delta: Vector = actual - expected;
        let cost = delta.fold(0.0, |a, x| a + x * x);
        (cost, delta)
    }
}

#[derive(Debug)]
struct Network {
    weights: Vec<Matrix>,
    biases: Vec<Vector>,
}

fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

fn sigmoid_prime(z: f32) -> f32 {
    sigmoid(z) * (1.0 - sigmoid(z))
}

fn sigmoid_arr(mut x: Vector) -> Vector {
    x.map_inplace(|x| *x = sigmoid(*x));
    x
}

fn sigmoid_prime_arr(mut x: Vector) -> Vector {
    x.map_inplace(|x| *x = sigmoid_prime(*x));
    x
}

impl Network {
    fn new<R: Rng>(input_size: usize, layer_sizes: &[usize], rng: &mut R) -> Self {
        let (weights, biases) = layer_sizes.iter().scan(input_size, |i, &o| {
            let weight_distr = Normal::new(0.0, 1.0 / (input_size as f32).sqrt()).unwrap();
            let weight = Matrix::random_using((o, *i), weight_distr, rng);
            let bias = Vector::random_using(o, StandardNormal, rng);
            *i = o;
            Some((weight, bias))
        }).unzip();

        Network {
            weights,
            biases,
        }
    }

    fn forward(&self, input: &Vector) -> Vector {
        let mut a = input.clone();
        for (w, b) in zip(&self.weights, &self.biases) {
            a = sigmoid_arr(w.dot(&a) + b);
        }
        a
    }

    fn train(&mut self, cost_fn: &impl CostFunc, data: &mut [Entry], test_data: Option<&[Entry]>, epochs: usize, batch_size: usize, learning_rate: f32, weight_decay: f32) {
        println!("{}", batch_size);
        println!("{}", data.len());
        assert!(batch_size <= data.len());
        let mut rng = SmallRng::from_entropy();

        for epoch in 0..epochs {
            print!("Starting epoch {} ... ", epoch);
            io::stdout().flush().unwrap();

            let mut total_cost = 0.0;

            data.shuffle(&mut rng);
            for batch in data.chunks_exact(batch_size) {
                // print!("Starting batch ... ");
                io::stdout().flush().unwrap();
                let batch_cost = self.train_batch(cost_fn, batch, learning_rate, weight_decay);
                total_cost += batch_cost;
                // println!("cost {}", batch_cost / batch_size as f32)
            }

            let batch_count = data.len() / batch_size;
            let avg_train_cost = total_cost / batch_count as f32;
            print!("train cost {}", avg_train_cost);

            if let Some(test_data) = test_data {
                let (avg_test_cost, test_correct) = self.evaluate(cost_fn, &test_data);
                println!(", test cost {} correct {}", avg_test_cost, test_correct);
            } else {
                println!();
            }
        }
    }

    fn evaluate(&mut self, cost_fn: &impl CostFunc, data: &[Entry]) -> (f32, f32) {
        let mut total_score = 0.0;
        let mut correct = 0.0;
        for (input, expected_output) in data {
            let output = self.forward(input);
            total_score += cost_fn.eval(expected_output, &output).0;

            let expected_i = expected_output.iter().position_max_by_key(|&&x| OrderedFloat::from(x));
            let actual_i = output.iter().position_max_by_key(|&&x| OrderedFloat::from(x));

            if expected_i == actual_i {
                correct += 1.0;
            }
        }
        let len_factor = data.len() as f32;
        (total_score / len_factor, correct / len_factor)
    }

    fn train_batch(&mut self, cost_fn: &impl CostFunc, batch: &[Entry], learning_rate: f32, weight_decay: f32) -> f32 {
        let mut w_deltas = self.weights.iter().map(|w| Matrix::zeros(w.raw_dim())).collect_vec();
        let mut b_deltas = self.biases.iter().map(|b| Vector::zeros(b.raw_dim())).collect_vec();

        let batch_len_factor = batch.len() as f32;
        let mut total_cost = 0.0;

        //collect deltas
        for (input, expected_output) in batch {
            total_cost += self.backprop(cost_fn, input, expected_output, &mut w_deltas, &mut b_deltas);
        }

        //apply deltas
        for (w, w_delta) in zip(&mut self.weights, w_deltas) {
            *w *= 1.0 - learning_rate * weight_decay / batch_len_factor;
            *w -= &(learning_rate / batch_len_factor * w_delta);
        }
        for (b, b_delta) in zip(&mut self.biases, b_deltas) {
            *b -= &(learning_rate / batch_len_factor * b_delta);
        }

        total_cost / batch_len_factor
    }

    fn backprop(&self, cost_fn: &impl CostFunc, input: &Vector, expected_output: &Vector, w_deltas: &mut [Matrix], b_deltas: &mut [Vector]) -> f32 {
        let mut a = input.clone();
        let mut zs = vec![];
        let mut activations = vec![a.clone()];

        // forward pass
        for (w, b) in zip(&self.weights, &self.biases) {
            let z = w.dot(&a) + b;
            a = sigmoid_arr(z.clone());

            zs.push(z);
            activations.push(a.clone());
        }

        //backwards pass
        let (cost, mut a_delta) = cost_fn.eval(expected_output, &a);

        for i in (0..self.weights.len()).rev() {
            let z_delta: Vector = &a_delta * &sigmoid_prime_arr(zs[i].clone());
            b_deltas[i] += &z_delta;

            let activation = &activations[i];

            //in-place outer product
            w_deltas[i].indexed_iter_mut().for_each(|((r, c), w)| {
                *w += z_delta[r] * activation[c]
            });

            a_delta = self.weights[i].t().dot(&z_delta);
        }

        cost
    }
}

fn test_deriv(network: &mut Network, cost_fn: &impl CostFunc, data: &[Entry]) {
    let mut w_backprop_deltas = network.weights.iter().map(|w| Matrix::zeros(w.raw_dim())).collect_vec();
    let mut b_backprop_deltas = network.biases.iter().map(|b| Vector::zeros(b.raw_dim())).collect_vec();

    let mut w_test_deltas = network.weights.iter().map(|w| Matrix::zeros(w.raw_dim())).collect_vec();
    let mut b_test_deltas = network.biases.iter().map(|b| Vector::zeros(b.raw_dim())).collect_vec();

    let mut orig_cost = 0.0;
    let mut second_cost = 0.0;

    const EPS: f32 = 0.01;

    for entry in data {
        orig_cost += network.backprop(cost_fn, &entry.0, &entry.1, &mut w_backprop_deltas, &mut b_backprop_deltas);
        second_cost += cost_fn.eval(&entry.1, &network.forward(&entry.0)).0;

        for layer in 0..network.weights.len() {
            for wi in 0..network.weights[layer].len() {
                let slice = network.weights[layer].as_slice_mut().unwrap();
                let orig_weight = slice[wi];
                slice[wi] += EPS;

                let output = network.forward(&entry.0);
                let (cost, _) = cost_fn.eval(&entry.1, &output);

                let test_delta = (cost - orig_cost) / EPS;
                w_test_deltas[layer].as_slice_mut().unwrap()[wi] += test_delta;

                network.weights[layer].as_slice_mut().unwrap()[wi] = orig_weight;
            }

            for bi in 0..network.biases[layer].len() {
                let slice = network.biases[layer].as_slice_mut().unwrap();
                let orig_bias = slice[bi];
                slice[bi] += EPS;

                let output = network.forward(&entry.0);
                let (cost, _) = cost_fn.eval(&entry.1, &output);

                let test_delta = (cost - orig_cost) / EPS;
                b_test_deltas[layer].as_slice_mut().unwrap()[bi] += test_delta;

                network.biases[layer].as_slice_mut().unwrap()[bi] = orig_bias;
            }
        }
    }

    let len_factor = data.len() as f32;
    orig_cost /= len_factor;
    second_cost /= len_factor;
    w_backprop_deltas.iter_mut().for_each(|w| *w /= len_factor);
    b_backprop_deltas.iter_mut().for_each(|b| *b /= len_factor);
    w_test_deltas.iter_mut().for_each(|w| *w /= len_factor);
    b_test_deltas.iter_mut().for_each(|b| *b /= len_factor);

    println!("Costs should equal: {} == {}", orig_cost, second_cost);

    println!("Backprop:");
    println!("w_deltas: {:?}", w_backprop_deltas);
    println!("b_deltas: {:?}", b_backprop_deltas);

    println!("Test:");
    println!("w_deltas: {:?}", w_backprop_deltas);
    println!("b_deltas: {:?}", b_backprop_deltas);
}

fn main() {
    // let mut rng = SmallRng::from_seed([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    let mut rng = SmallRng::from_entropy();

    let mut network = Network::new(28 * 28, &[30, 10], &mut rng);

    let (mut train_data, test_data) = load_mnist();

    // test_deriv(&mut network, &train_data[0..10]);
    // return;

    let output = network.forward(&train_data[0].0);
    println!("{}", output);

    network.train(&QuadraticCost, &mut train_data, Some(&test_data), 30, 10, 3.0, 0.0);
    let output = network.forward(&train_data[0].0);
    println!("{}", output);
}
