use std::io;
use std::io::Write;

use itertools::{Itertools, izip, zip};
use mnist::MnistBuilder;
use ndarray::{Dimension, Ix1, Ix2};
use ndarray::Array;
use ndarray_rand::RandomExt;
use ordered_float::OrderedFloat;
use rand::{Rng, SeedableRng, thread_rng};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand_distr::{Bernoulli, Distribution, Normal, StandardNormal};

use sttt::board::{Board, Coord, Player};
use sttt::bot_game::{Bot, RandomBot};
use sttt::mcts::{mcts_evaluate, MCTSBot};
use sttt::mcts::heuristic::ZeroHeuristic;

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

trait Trainer {
    type WeightState: From<(usize, usize)>;
    type BiasState: From<usize>;

    fn step_weight(&self, weight: &mut Matrix, state: &mut Self::WeightState, delta: Matrix);
    fn step_bias(&self, bias: &mut Vector, state: &mut Self::BiasState, delta: Vector);
}

struct GradientDescent {
    learning_rate: f32,
}

struct EmptyState;

impl From<usize> for EmptyState {
    fn from(_: usize) -> Self {
        EmptyState
    }
}

impl From<(usize, usize)> for EmptyState {
    fn from(_: (usize, usize)) -> Self {
        EmptyState
    }
}

//TODO how to combine bias and weights? they're different types :(
impl Trainer for GradientDescent {
    type WeightState = EmptyState;
    type BiasState = EmptyState;

    fn step_weight(&self, weight: &mut Matrix, _state: &mut EmptyState, delta: Matrix) {
        //TODO avoid making copy here
        //TODO why doesn't delta: &Matrix work?
        *weight -= &(self.learning_rate * delta);
    }

    fn step_bias(&self, bias: &mut Vector, _state: &mut EmptyState, delta: Vector) {
        *bias -= &(self.learning_rate * delta);
    }
}

struct Adam {
    alpha: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
}

impl Default for Adam {
    fn default() -> Self {
        Adam { alpha: 0.001, beta1: 0.9, beta2: 0.999, eps: 1e-8 }
    }
}

struct AdamState<D> {
    m: Array<f32, D>,
    v: Array<f32, D>,
    //TODO this doesn't need to be duplicated for every weight, maybe add some global state?
    //   or maybe just bail and let the Trainer manage its own state?
    t: usize,
}

impl From<usize> for AdamState<Ix1> {
    fn from(sh: usize) -> Self {
        AdamState {
            m: Vector::zeros(sh),
            v: Vector::zeros(sh),
            t: 0,
        }
    }
}

impl From<(usize, usize)> for AdamState<Ix2> {
    fn from(sh: (usize, usize)) -> Self {
        AdamState {
            m: Matrix::zeros(sh),
            v: Matrix::zeros(sh),
            t: 0,
        }
    }
}

impl Adam {
    fn step<D: Dimension>(&self, weight: &mut Array<f32, D>, state: &mut AdamState<D>, delta: Array<f32, D>) {
        state.t += 1;
        let t = state.t as i32;

        state.m = self.beta1 * &state.m + (1.0 - self.beta1) * &delta;
        state.v = self.beta2 * &state.v + (1.0 - self.beta2) * &delta.map(|x| x * x);

        let alpha_t = self.alpha * (1.0 - self.beta2.powi(t)).sqrt() / (1.0 - self.beta1.powi(t));
        *weight -= &(alpha_t * &state.m / state.v.map(|x| x.sqrt() + self.eps))
    }
}

impl Trainer for Adam {
    type WeightState = AdamState<Ix2>;
    type BiasState = AdamState<Ix1>;

    fn step_weight(&self, weight: &mut Matrix, state: &mut Self::WeightState, delta: Matrix) {
        self.step(weight, state, delta);
    }

    fn step_bias(&self, bias: &mut Vector, state: &mut Self::BiasState, delta: Vector) {
        self.step(bias, state, delta);
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

    fn train(
        &mut self,
        cost_fn: &impl CostFunc,
        trainer: &impl Trainer,
        data: &mut [Entry],
        test_data: Option<&[Entry]>,
        epochs: usize,
        batch_size: usize,
    ) {
        println!("{}", batch_size);
        println!("{}", data.len());
        assert!(batch_size <= data.len());
        let mut rng = SmallRng::from_entropy();

        let mut w_states = self.weights.iter().map(Matrix::dim).map_into().collect_vec();
        let mut b_states = self.biases.iter().map(Vector::dim).map_into().collect_vec();

        for epoch in 0..epochs {
            print!("Starting epoch {} ... ", epoch);
            io::stdout().flush().unwrap();

            let mut total_cost = 0.0;

            data.shuffle(&mut rng);
            for batch in data.chunks_exact(batch_size) {
                // print!("Starting batch ... ");
                io::stdout().flush().unwrap();
                let batch_cost = self.train_batch(cost_fn, trainer, batch, &mut w_states, &mut b_states);
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

    fn train_batch<T: Trainer>(
        &mut self,
        cost_fn: &impl CostFunc,
        trainer: &T,
        batch: &[Entry],
        w_states: &mut [T::WeightState],
        b_states: &mut [T::BiasState],
    ) -> f32 {
        let mut w_deltas = self.weights.iter().map(|w| Matrix::zeros(w.raw_dim())).collect_vec();
        let mut b_deltas = self.biases.iter().map(|b| Vector::zeros(b.raw_dim())).collect_vec();

        let batch_len_factor = batch.len() as f32;
        let mut total_cost = 0.0;

        //collect deltas
        for (input, expected_output) in batch {
            total_cost += self.backprop(cost_fn, input, expected_output, &mut w_deltas, &mut b_deltas);
        }

        //apply deltas
        //TODO move this into trainer struct
        for (w, w_state, mut w_delta) in izip!(&mut self.weights, w_states, w_deltas) {
            w_delta /= batch_len_factor;
            trainer.step_weight(w, w_state, w_delta);
        }
        for (b, b_state, mut b_delta) in izip!(&mut self.biases, b_states, b_deltas) {
            b_delta /= batch_len_factor;
            trainer.step_bias(b, b_state, b_delta);
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

fn _old_main() {
    // let mut rng = SmallRng::from_seed([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    let mut rng = SmallRng::from_entropy();

    let mut network = Network::new(28 * 28, &[30, 10], &mut rng);

    let (mut train_data, test_data) = load_mnist();

    // test_deriv(&mut network, &train_data[0..10]);
    // return;

    let output = network.forward(&train_data[0].0);
    println!("{}", output);

    network.train(
        &QuadraticCost,
        &GradientDescent { learning_rate: 3.0 },
        &mut train_data,
        Some(&test_data),
        30,
        10,
    );
    let output = network.forward(&train_data[0].0);
    println!("{}", output);
}

fn gen_boards_from_mcts_self_play(count: usize, iterations: usize, explore_prob: f64) -> Vec<(Board, f32)> {
    let mut result = Vec::new();
    let mut rand = thread_rng();
    let explore_prob = Bernoulli::new(explore_prob).unwrap();

    let mut prev_progress = usize::MAX;

    print!("Generating games ... ");

    while result.len() < count {
        //print progress
        let progress = result.len() / (count / 10);
        if progress != prev_progress {
            print!("{}", progress);
            std::io::stdout().flush().unwrap();
            prev_progress = progress;
        }

        let mut board = Board::new();

        while !board.is_done() {
            let evaluation = mcts_evaluate(&board, iterations, &ZeroHeuristic, &mut rand);

            //save the board
            result.push((board.clone(), evaluation.value));

            //play a move
            if explore_prob.sample(&mut rand) {
                board.play(board.random_available_move(&mut rand).unwrap());
            } else {
                board.play(evaluation.best_move.unwrap());
            }
        }

        //add final board too
        let value = value_winner(board.won_by.unwrap(), board.next_player);
        result.push((board, value));
    }

    result.truncate(count);

    println!();
    result
}

fn value_winner(winner: Player, next_player: Player) -> f32 {
    if winner == next_player { 1.0 } else if winner == Player::Neutral { 0.5 } else { 0.0 }
}

fn gen_done_boards(count: usize) -> Vec<(Board, f32)> {
    let mut rand = thread_rng();

    (0..count).map(|_| {
        let mut board = Board::new();
        while let Some(mv) = board.random_available_move(&mut rand) {
            board.play(mv);
        }

        //TODO this never contains a won board!
        let value = value_winner(board.won_by.unwrap(), board.next_player);
        (board, value)
    }).collect_vec()
}

fn main() {
    /*
    //do some statistics
    let mut board = Board::new();
    let mut rand = thread_rng();

    // for _ in 0..10 {
    //     board.play(board.random_available_move(&mut rand).unwrap());
    // }

    let samples = (0..1000)
        .map(|_| mcts_evaluate(&board, 10000, &ZeroHeuristic, &mut rand).value)
        .collect_vec();

    println!("Samples {:?}", samples);
    let mean: f32 = samples.iter().fold(0.0, |a, &x| a + x) / samples.len() as f32;
    let sigma: f32 = (samples.iter().fold(0.0, |a, &x| a + (x - mean) * (x - mean)) / samples.len() as f32).sqrt();

    println!("Mean {}", mean);
    println!("Sigma {}", sigma);
    */

    //start training

    // TODO why is this piece of shit neural network not able to learn whether a position is won or lost?
    //   it literally only needs to look at the macro board
    //   IDEA: try without any other inputs first
    //   IDEA: just try to get it to detect won/not won at first


    const INPUT_SIZE: usize = /*81 * 4 +*/ 9 /* 2*/;
    const MCTS_ITERATIONS: usize = 5000;
    const SUPER_EPOCH_SIZE: usize = 100_000;

    let mut network = Network::new(INPUT_SIZE, &[9, 8, 3, 1], &mut SmallRng::from_entropy());

    println!("Generating boards");
    let boards = gen_done_boards(SUPER_EPOCH_SIZE);
    println!("Embedding boards");
    let mut all_data = boards.iter().map(|(board, evaluation)| {
        let mut input = Vec::with_capacity(INPUT_SIZE);

        /*for coord in Coord::all() {
            let p = board.tile(coord);
            input.push(((p == board.next_player) as u8) as f32);
            input.push(((p == board.next_player.other()) as u8) as f32);
            input.push(((Some(coord) == board.last_move) as u8) as f32);
            input.push(((board.is_available_move(coord)) as u8) as f32);
        }*/

        //TODO shouldn't the network be able to learn this on its own?
        for om in 0..9 {
            let p = board.macr(om);

            input.push(((p == board.next_player) as u8) as f32);
            // input.push(((p == board.next_player.other()) as u8) as f32);
        }

        assert_eq!(INPUT_SIZE, input.len());

        let input = Vector::from(input);
        let output = Vector::from(vec![*evaluation]);

        println!("Evaluation {}", evaluation);

        (input, output)
    }).collect_vec();

    // for super_epoch in 0.. {
    //     println!("Starting super_epoch {}", super_epoch);

    println!("Start training");
    network.train(
        &QuadraticCost,
        &GradientDescent { learning_rate: 3.0 },
        &mut all_data,
        None,
        1000,
        500,
    )
    // }
}