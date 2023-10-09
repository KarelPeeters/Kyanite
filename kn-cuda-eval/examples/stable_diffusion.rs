use std::path::PathBuf;

use bytemuck::cast_slice;
use clap::ArgGroup;
use clap::Parser;
use image::{ImageBuffer, Rgb, RgbImage};
use itertools::Itertools;
use ndarray::Axis;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

use kn_cuda_eval::runtime::Runtime;
use kn_cuda_sys::wrapper::handle::Device;
use kn_graph::dtype::{DTensor, DType, Tensor};
use kn_graph::graph::{BinaryOp, Graph, SliceRange};
use kn_graph::ndarray::Array;
use kn_graph::onnx::load_graph_from_onnx_path;
use kn_graph::optimizer::optimize_graph;
use kn_graph::{ndarray, shape};

use crate::ndarray::{Array1, IxDyn, Slice};
use crate::scheduler::PNDMSScheduler;

#[derive(clap::Parser)]
#[clap(group = ArgGroup::new("device").required(true))]
struct Args {
    #[clap(long)]
    network_path: PathBuf,
    #[clap(long)]
    output_folder: PathBuf,

    #[clap(long, group = "device")]
    cpu: bool,
    #[clap(long, group = "device")]
    gpu: Option<usize>,

    #[clap(long)]
    steps: usize,
    #[clap(long)]
    prompt: String,
    #[clap(long)]
    prompt_avoid: Option<String>,
    #[clap(long, default_value_t = 7.5)]
    guidance_scale: f32,

    #[clap(long)]
    no_save_intermediate: bool,
    #[clap(long)]
    no_save_latents: bool,

    #[clap(long)]
    seed: Option<u64>,
    #[clap(long)]
    seed_latents_path: Option<PathBuf>,
}

const EMBED_LENGTH: usize = 77;
const START_TOKEN: u32 = 49406;
const END_TOKEN: u32 = 49407;

const LATENT_CHANNELS: usize = 4;
const IMAGE_SIZE: usize = 512;
const LATENT_SIZE: usize = IMAGE_SIZE / 8;

fn main() -> std::io::Result<()> {
    println!("Parsing arguments");
    let args: Args = Args::parse();
    let tokens_prompt = str_to_tokens(&args.prompt);
    let tokens_prompt_avoid = args.prompt_avoid.as_ref().map_or_else(Vec::new, |s| str_to_tokens(s));
    std::fs::create_dir_all(&args.output_folder)?;

    let device = if args.cpu {
        println!("  Using CPU");
        None
    } else {
        let device = Device::new(0);
        println!("  Using GPU {:?}", device);
        Some(device)
    };

    println!("Loading graphs");
    let path_text_encoder = args.network_path.join("text_encoder.onnx");
    let path_unet = args.network_path.join("unet/unet.onnx");
    // let path_encoder = args.network_path.join("vae_encoder.onnx");
    let path_decoder = args.network_path.join("vae_decoder.onnx");

    let graph_text_encoder = load_graph_from_onnx_path(path_text_encoder, true).unwrap();
    let graph_unet = load_graph_from_onnx_path(path_unet, true).unwrap();
    // let graph_encoder = load_graph_from_onnx_path(path_encoder, true).unwrap();
    let graph_decoder = load_graph_from_onnx_path(path_decoder, true).unwrap();

    println!("Optimizing graphs");
    let graph_text_encoder = optimize_graph(&graph_text_encoder, Default::default());
    let graph_unet = optimize_graph(&graph_unet, Default::default());
    // let graph_encoder = optimize_graph(&graph_encoder, Default::default());
    let graph_decoder = optimize_graph(&graph_decoder, Default::default());

    println!("Preparing runtime");
    let mut runtime = Runtime::new(device);
    let runtime_text_encoder = runtime.prepare(graph_text_encoder, 0);
    let runtime_unet = runtime.prepare(graph_unet, 0);
    // let runtime_encoder = runtime.prepare(graph_encoder, 0);
    let runtime_decoder = runtime.prepare(graph_decoder, 0);

    println!("Embedding text");
    let tokens_prompt = tokens_to_tensor(&tokens_prompt);
    let tokens_uncond = tokens_to_tensor(&tokens_prompt_avoid);

    let emb_prompt = runtime.eval(runtime_text_encoder, &[tokens_prompt]).single();
    let emb_prompt = emb_prompt.unwrap_f32().unwrap();
    let emb_uncond = runtime.eval(runtime_text_encoder, &[tokens_uncond]).single();
    let emb_uncond = emb_uncond.unwrap_f32().unwrap();
    let emb_all = ndarray::concatenate![Axis(0), emb_uncond.clone(), emb_prompt.clone()].into_shared();

    println!("Initializing latents");
    let latent_shape = (1, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE);

    let mut latent = if let Some(seed_latents_path) = &args.seed_latents_path {
        println!("  Loading from disk");
        let data = std::fs::read(seed_latents_path)?;
        let data_float = cast_slice::<u8, f32>(&data).to_vec();
        Array::from_shape_vec(latent_shape, data_float)
            .unwrap()
            .into_dyn()
            .into_shared()
    } else {
        println!("  Generating random");

        let mut rng = if let Some(seed) = args.seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_entropy()
        };

        Array::from_shape_simple_fn(latent_shape, || rng.sample::<f32, _>(StandardNormal))
            .into_dyn()
            .into_shared()
    };

    println!("Initializing schedule");
    let mut schedule = PNDMSScheduler::default();
    schedule.init();
    schedule.set_timesteps(args.steps, 1);

    println!("Starting diffusion process");
    let timesteps = schedule.timesteps();
    for (i, &t) in timesteps.iter().enumerate() {
        println!("  Diffusion step {i}/{}, t={t}", timesteps.len());

        if !args.no_save_latents {
            println!("    Saving intermediate latents");
            latent_to_image(&latent)
                .save(args.output_folder.join(format!("latent_{i}.png")))
                .unwrap();
        }

        if !args.no_save_intermediate {
            println!("    Decoding intermediate image");
            let image = runtime.eval(runtime_decoder, &[DTensor::F32(latent.clone())]).single();
            let image = image.unwrap_f32().unwrap();

            println!("    Saving intermediate image");
            tensor_to_image(image)
                .save(args.output_folder.join(format!("image_{i}.png")))
                .unwrap();
        }

        let t_tensor = Array::from_shape_fn((), |()| t as f32).into_dyn().into_shared();
        let latent_input = ndarray::concatenate![Axis(0), latent, latent].into_shared();
        let unet_inputs = [
            DTensor::F32(latent_input),
            DTensor::F32(t_tensor),
            DTensor::F32(emb_all.clone()),
        ];

        println!("    Running unet");
        let noise_pred_all = runtime.eval(runtime_unet, &unet_inputs).single();
        let noise_pred_all = noise_pred_all.unwrap_f32().unwrap();

        println!("    Shuffling outputs");
        let noise_pred_uncond = noise_pred_all.slice_axis(Axis(0), Slice::from(0..1));
        let noise_pred_prompt = noise_pred_all.slice_axis(Axis(0), Slice::from(1..2));

        let noise_pred =
            (&noise_pred_uncond + args.guidance_scale * (&noise_pred_prompt - &noise_pred_uncond)).into_shared();

        println!("    Running step");
        latent = schedule.step_plms(noise_pred, t, latent);
    }

    println!("Saving final latents");
    latent_to_image(&latent)
        .save(args.output_folder.join("latent_final.png"))
        .unwrap();

    println!("Decoding final image");
    let image = runtime.eval(runtime_decoder, &[DTensor::F32(latent.clone())]).single();
    let image = image.unwrap_f32().unwrap();

    println!("Saving final image");
    tensor_to_image(image)
        .save(args.output_folder.join("image_final.png"))
        .unwrap();

    Ok(())
}

fn str_to_tokens(s: &str) -> Vec<u32> {
    s.split(',').map(|x| x.parse::<u32>().unwrap()).collect_vec()
}

fn tokens_to_tensor(tokens: &[u32]) -> DTensor {
    assert!(tokens.len() + 2 < EMBED_LENGTH);

    let array = Array::from_shape_fn((1, EMBED_LENGTH), |(_, i)| {
        if i == 0 {
            START_TOKEN
        } else if i - 1 < tokens.len() {
            tokens[i - 1]
        } else {
            END_TOKEN
        }
    });

    DTensor::U32(array.into_dyn().into_shared())
}

fn latent_to_image(latent: &Tensor<f32>) -> RgbImage {
    let latent_image = latent
        .clone()
        .permuted_axes(IxDyn(&[0, 2, 1, 3]))
        .reshape((1, 1, LATENT_SIZE, 4 * LATENT_SIZE))
        .broadcast((1, 3, LATENT_SIZE, 4 * LATENT_SIZE))
        .unwrap()
        .to_owned()
        .into_dyn()
        .into_shared();
    tensor_to_image(&(latent_image / 3.0 + 0.5))
}

fn tensor_to_image(tensor: &Tensor<f32>) -> RgbImage {
    let shape = tensor.shape();

    let shape = if shape.len() == 4 {
        assert_eq!(shape[0], 1, "Rank 4 tensor only allowed if leading size is 1");
        &shape[1..]
    } else {
        shape
    };

    assert_eq!(shape.len(), 3, "Expected shape (height, width, color), got {:?}", shape);
    assert_eq!(shape[0], 3, "Expected 3 color channels");
    let height = shape[1];
    let width = shape[2];

    ImageBuffer::from_fn(width as u32, height as u32, |x, y| {
        let x = x as usize;
        let y = y as usize;
        let map = |f: f32| (((f + 1.0) / 2.0).clamp(0.0, 1.0) * 255.0) as u8;
        Rgb([
            map(tensor[[0, 0, y, x]]),
            map(tensor[[0, 1, y, x]]),
            map(tensor[[0, 2, y, x]]),
        ])
    })
}

#[allow(dead_code)]
fn tensor_from_image(image: &RgbImage) -> Tensor<f32> {
    Array::from_shape_fn(
        (1, 3, image.height() as usize, image.width() as usize),
        |(_, c, y, x)| {
            let p = image.get_pixel(x as u32, y as u32).0[c];
            p as f32 / 255.0 * 2.0 - 1.0
        },
    )
    .into_shared()
    .into_dyn()
}

#[allow(dead_code)]
fn fuse_autoencoder_graphs(graph_encoder: &Graph, graph_decoder: &Graph) -> Graph {
    const LATENT_SCALAR: f32 = 5.489980697631836;

    let mut graph = Graph::new();
    let input = graph.input(shape![1, 3, 512, 512], DType::F32);
    let moments = graph.call(&graph_encoder, &[input]).single();
    let latents_raw = graph.slice(moments, 1, SliceRange::simple(0, 4));
    let latent_scalar = graph.scalar(LATENT_SCALAR);
    let latents = graph.binary(BinaryOp::Div, latents_raw, latent_scalar);
    let result = graph.call(&graph_decoder, &[latents]).single();
    graph.output(result);
    graph
}

mod scheduler {
    use std::cmp::max;

    use itertools::Itertools;

    use kn_graph::dtype::Tensor;

    use crate::{Array1, Axis, VecExt};

    /// Direct Python-to-Rust translation of the [HuggingFace diffusers implementation][1],
    /// with only the parts that are used with the default settings.
    ///
    /// [1]: (https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_pndm.py)
    #[derive(Default)]
    pub struct PNDMSScheduler {
        // init
        betas: Array1<f32>,
        alphas: Array1<f32>,
        alphas_cumprod: Array1<f32>,
        pndm_order: usize,
        num_train_timesteps: usize,

        // set_timesteps
        num_inference_steps: usize,
        _offset: usize,
        timesteps: Vec<usize>,

        // step
        ets: Vec<Tensor<f32>>,
        counter: usize,
        cur_sample: Option<Tensor<f32>>,
    }

    impl PNDMSScheduler {
        pub fn init(&mut self) {
            let beta_start: f32 = 0.00085;
            let beta_end: f32 = 0.012;
            self.num_train_timesteps = 1000;

            self.betas = Array1::linspace(beta_start.powf(0.5), beta_end.powf(0.5), self.num_train_timesteps)
                .mapv(|x| x.powf(2.0));

            self.alphas = 1.0 - &self.betas;
            self.alphas_cumprod = self.alphas.clone();
            self.alphas_cumprod.accumulate_axis_inplace(Axis(0), |a, b| *b = a * *b);

            self.pndm_order = 4;

            self.counter = 0;
            self.cur_sample = None;
            self.ets = vec![];
        }

        pub fn set_timesteps(&mut self, num_inference_steps: usize, offset: usize) {
            self.num_inference_steps = num_inference_steps;
            let _timesteps = (0..self.num_train_timesteps)
                .step_by(self.num_train_timesteps / num_inference_steps)
                .collect_vec();

            self._offset = offset;
            let _timesteps = _timesteps.iter().map(|&t| t + self._offset).collect_vec();

            let mut plms_timesteps = _timesteps.clone();
            plms_timesteps.insert(plms_timesteps.len() - 2, _timesteps[_timesteps.len() - 2]);
            plms_timesteps.reverse();

            self.timesteps = plms_timesteps;

            self.ets = vec![];
            self.counter = 0;
        }

        pub fn timesteps(&self) -> Vec<usize> {
            self.timesteps.clone()
        }

        pub fn step_plms(
            &mut self,
            mut model_output: Tensor<f32>,
            mut timestep: usize,
            mut sample: Tensor<f32>,
        ) -> Tensor<f32> {
            let mut prev_timestep = max(
                timestep as isize - (self.num_train_timesteps / self.num_inference_steps) as isize,
                0,
            ) as usize;

            if self.counter != 1 {
                self.ets.push(model_output.clone());
            } else {
                prev_timestep = timestep;
                timestep = timestep + self.num_train_timesteps / self.num_inference_steps
            }

            if self.ets.len() == 1 && self.counter == 0 {
                model_output = model_output;
                self.cur_sample = Some(sample.clone());
            } else if self.ets.len() == 1 && self.counter == 1 {
                model_output = ((model_output + self.ets.signed_index(-1)) / 2.0).into_shared();
                sample = self.cur_sample.take().unwrap();
            } else if self.ets.len() == 2 {
                model_output = ((3.0 * self.ets.signed_index(-1) - self.ets.signed_index(-2)) / 2.0).into_shared();
            } else if self.ets.len() == 3 {
                model_output = ((23.0 * self.ets.signed_index(-1) - 16.0 * self.ets.signed_index(-2)
                    + 5.0 * self.ets.signed_index(-3))
                    / 12.0)
                    .into_shared();
            } else {
                model_output = ((1.0 / 24.0)
                    * (55.0 * self.ets.signed_index(-1) - 59.0 * self.ets.signed_index(-2)
                        + 37.0 * self.ets.signed_index(-3)
                        - 9.0 * self.ets.signed_index(-4)))
                .into_shared();
            }

            let prev_sample = self.get_prev_sample(sample, timestep, prev_timestep, model_output);
            self.counter += 1;

            prev_sample
        }

        fn get_prev_sample(
            &mut self,
            sample: Tensor<f32>,
            timestep: usize,
            timestep_prev: usize,
            model_output: Tensor<f32>,
        ) -> Tensor<f32> {
            let alpha_prod_t = self.alphas_cumprod[timestep + 1 - self._offset];
            let alpha_prod_t_prev = self.alphas_cumprod[timestep_prev + 1 - self._offset];

            let beta_prod_t = 1.0 - alpha_prod_t;
            let beta_prod_t_prev = 1.0 - alpha_prod_t_prev;

            let sample_coeff = (alpha_prod_t_prev / alpha_prod_t).powf(0.5);

            let model_output_denom_coeff =
                alpha_prod_t * beta_prod_t_prev.powf(0.5) + (alpha_prod_t * beta_prod_t * alpha_prod_t_prev).powf(0.5);

            let prev_sample =
                sample_coeff * sample - (alpha_prod_t_prev - alpha_prod_t) * model_output / model_output_denom_coeff;
            prev_sample
        }
    }
}

trait VecExt {
    type Item;
    fn signed_index(&self, index: isize) -> &Self::Item;
    fn single(self) -> Self::Item;
}

impl<T> VecExt for Vec<T> {
    type Item = T;

    fn signed_index(&self, index: isize) -> &Self::Item {
        if index >= 0 {
            &self[index as usize]
        } else {
            &self[self.len() - ((-index) as usize)]
        }
    }

    fn single(mut self) -> Self::Item {
        assert_eq!(self.len(), 1, "Vec::single length must be 1");
        self.remove(0)
    }
}
