use std::cmp::Ordering;
use std::collections::HashSet;
use std::fs::create_dir_all;
use std::path::{Path, PathBuf};

use image::{ImageBuffer, Rgb};
use ndarray::ArrayView4;
use palette::{LinSrgb, Srgb};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use nn_graph::graph::{Graph, Value};
use nn_graph::onnx::load_graph_from_onnx_path;

pub type Image = ImageBuffer<Rgb<u8>, Vec<u8>>;

fn main() {
    let range = 15..3634;

    range.into_par_iter().for_each(|ei| {
        let network_path = format!(
            "C:/Documents/Programming/STTT/AlphaZero/data/loop/chess/16x128/training/gen_{}/network.onnx",
            ei
        );
        if !Path::new(&network_path).exists() {
            return;
        }

        println!("Saving filters for network {}", ei);

        let graph = load_graph_from_onnx_path(network_path);

        for (_, &v) in ordered_values(&graph).iter().enumerate() {
            if let Some(data) = graph.as_const(v) {
                let shape = graph[v].shape.unwrap_fixed("constant shape");

                if shape.size() <= 1 {
                    continue;
                };

                let (k, c, h, w) = match &*shape.dims {
                    &[k, c, h, w] => (k, c, h, w),
                    &[k, c] => (k, c, 1, 1),
                    _ => continue,
                };

                let image_path = PathBuf::from(format!(
                    "D:/Documents/A0/filters/16x128/{}_{}/{}.png",
                    v.index(),
                    shape,
                    ei
                ));
                create_dir_all(image_path.parent().unwrap()).unwrap();
                if image_path.exists() {
                    continue;
                }

                println!("Saving filters for layer {}_{}", v.index(), shape);

                let mut sorted = data.to_owned();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Less));
                let q_min = sorted[sorted.len() / 100];
                let q_max = sorted[99 * sorted.len() / 100];
                let range = f32::max(q_max, -q_min);

                let data = ArrayView4::from_shape((k, c, h, w), data).unwrap();

                let padding = if h == 1 && w == 1 { 0 } else { 1 };

                let background = Srgb::from(LinSrgb::new(0.1, 0.1, 0.1));
                let background = Rgb([background.red, background.green, background.blue]);

                let mut image = Image::from_pixel(
                    (padding + (c * (w + padding))) as u32,
                    (padding + (k * (h + padding))) as u32,
                    background,
                );

                for ki in 0..k {
                    for ci in 0..c {
                        for hi in 0..h {
                            for wi in 0..w {
                                let x = padding + ci * (w + padding) + wi;
                                let y = padding + ki * (h + padding) + hi;

                                let value = data[(ki, ci, hi, wi)];
                                let value = ((value + range) / (2.0 * range)).clamp(0.0, 1.0);

                                let color = Srgb::from(LinSrgb::new(value, value, value));
                                let p = Rgb([color.red, color.green, color.blue]);

                                image.put_pixel(x as u32, y as u32, p);
                            }
                        }
                    }
                }

                image.save(image_path).unwrap();
            }
        }
    });
}

fn ordered_values(graph: &Graph) -> Vec<Value> {
    let mut result = vec![];
    let mut visited = HashSet::default();

    for &output in graph.outputs() {
        ordered_values_impl(graph, &mut result, &mut visited, output);
    }

    result
}

fn ordered_values_impl(graph: &Graph, result: &mut Vec<Value>, visited: &mut HashSet<Value>, curr: Value) {
    if !visited.insert(curr) {
        return;
    }

    for input in graph[curr].operation.inputs() {
        ordered_values_impl(graph, result, visited, input)
    }

    result.push(curr);
}
