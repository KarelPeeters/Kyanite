use std::cmp::max;
use std::fmt::{Debug, Formatter};

use image::{ImageBuffer, Rgb};
use itertools::{Itertools, zip};
use ndarray::{ArcArray, Axis, Ix4};
use palette::{LinSrgb, Srgb};

use crate::cpu::{ExecutionInfo, Tensor};
use crate::graph::{Graph, Operation, Value};
use crate::shape::Size;

pub type Image = ImageBuffer<Rgb<u8>, Vec<u8>>;
type Tensor4 = ArcArray<f32, Ix4>;

const VERTICAL_PADDING: usize = 5;
const HORIZONTAL_PADDING: usize = 5;

pub fn visualize_graph_activations(
    graph: &Graph,
    execution: &ExecutionInfo,
    post_process_value: impl Fn(Value, Tensor) -> Option<Tensor>,
    max_images: Option<usize>,
) -> Vec<Image> {
    let batch_size = execution.batch_size;
    let image_count = max_images.map_or(batch_size, |max_images| max(max_images, batch_size));

    let mut total_width = HORIZONTAL_PADDING;
    let mut total_height = VERTICAL_PADDING;

    let mut selected = vec![];

    for value in execution.values.values() {
        let info = &graph[value.value];

        if !should_show_value(graph, value.value) {
            continue;
        }

        // check whether this is the typical intermediate shape: [B, fixed*]
        let is_intermediate_shape =
            info.shape.rank() > 0 &&
                info.shape[0] == Size::BATCH &&
                info.shape.dims[1..].iter().all(|d| d.try_unwrap_fixed().is_some());
        if !is_intermediate_shape {
            println!("Skipping value with shape {:?}", info.shape);
            continue;
        }

        let data = value.tensor.to_shared();

        selected.push((Some(value.value), data.to_shared()));
        if let Some(extra) = post_process_value(value.value, data) {
            selected.push((None, extra));
        }
    }

    let mut all_details = vec![];
    for (value, data) in selected {
        let size = data.len();

        let data: Tensor4 = match data.ndim() {
            1 => data.reshape((batch_size, 1, 1, 1)),
            2 => data.reshape((batch_size, 1, 1, size / batch_size)),
            3 => data.insert_axis(Axis(1)).into_dimensionality().unwrap(),
            4 => data.into_dimensionality().unwrap(),
            _ => {
                println!("Skipping value with (picked) shape {:?}", data.dim());
                continue;
            }
        };

        let data = if matches!(data.dim(), (_, _, 1, 1)) {
            data.reshape((batch_size, 1, 1, data.dim().1))
        } else {
            data
        };

        let (_, channels, height, width) = data.dim();

        let view_width = channels * width + (channels - 1) * HORIZONTAL_PADDING;
        let view_height = height;

        if total_height != VERTICAL_PADDING {
            total_height += VERTICAL_PADDING;
        }
        let start_y = total_height;
        total_height += view_height;

        total_width = max(total_width, HORIZONTAL_PADDING + view_width);

        let details = Details { value, data, start_y };
        all_details.push(details)
    }

    total_width += HORIZONTAL_PADDING;
    total_height += VERTICAL_PADDING;

    let background = Srgb::from(LinSrgb::new(0.01, 0.01, 0.01));
    let background = Rgb([background.red, background.green, background.blue]);

    let mut images = (0..image_count)
        .map(|_| ImageBuffer::from_pixel(total_width as u32, total_height as u32, background))
        .collect_vec();

    for details in all_details.iter() {
        let data = &details.data;
        let (_, channels, height, width) = data.dim();

        //TODO scale what by what exactly?

        let mean = data.mean().unwrap();
        let std = data.std(1.0);
        let data_norm = (data - mean) / std;

        let std_ele = data_norm.std_axis(Axis(0), 1.0);
        let std_ele_mean = std_ele.mean().unwrap();
        let std_ele_std = std_ele.std(1.0);

        for (image_i, image) in images.iter_mut().enumerate() {
            for c in 0..channels {
                for w in 0..width {
                    let x = HORIZONTAL_PADDING + c * (HORIZONTAL_PADDING + width) + w;
                    for h in 0..height {
                        let y = details.start_y + (height - 1 - h);

                        let s = (std_ele[(c, h, w)] - std_ele_mean) / std_ele_std;
                        let s_norm = ((s + 1.0) / 2.0).clamp(0.0, 1.0);

                        let f = data_norm[(image_i, c, h, w)];
                        let f_norm = ((f + 1.0) / 2.0).clamp(0.0, 1.0);

                        let color = Srgb::from(LinSrgb::new(s_norm, f_norm, f_norm));
                        let p = Rgb([color.red, color.green, color.blue]);
                        image.put_pixel(x as u32, y as u32, p);
                    }
                }
            }
        }
    }

    images
}

fn should_show_value(graph: &Graph, value: Value) -> bool {
    if graph.inputs().contains(&value) || graph.outputs().contains(&value) {
        return true;
    }

    if is_effectively_constant(graph, value) {
        return false;
    }

    let has_dummy_user = graph.values().any(|other| {
        let other_operation = &graph[other].operation;

        if other_operation.inputs().contains(&value) {
            match other_operation {
                Operation::Input { .. } | Operation::Constant { .. } => unreachable!(),
                &Operation::View { input } => {
                    // check if all commons dims at the start match, which implies the only different is trailing 1s
                    zip(&graph[input].shape.dims, &graph[other].shape.dims)
                        .all(|(l, r)| l == r)
                }
                Operation::Permute { .. }
                | Operation::Slice { .. } | Operation::Gather { .. } | Operation::Concat { .. }
                | Operation::Conv { .. } | Operation::MatMul { .. } => false,
                &Operation::Add { left, right, subtract: _ } | &Operation::Mul { left, right } => {
                    graph[left].shape != graph[right].shape
                }
                Operation::Clamp { .. } => true,
            }
        } else {
            false
        }
    });

    return !has_dummy_user;
}

fn is_effectively_constant(graph: &Graph, value: Value) -> bool {
    let operation = &graph[value].operation;
    match operation {
        Operation::Input { .. } => false,
        Operation::Constant { .. } => true,
        Operation::View { .. } | Operation::Permute { .. }
        | Operation::Slice { .. } | Operation::Gather { .. } | Operation::Concat { .. }
        | Operation::Conv { .. } | Operation::MatMul { .. }
        | Operation::Add { .. } | Operation::Mul { .. }
        | Operation::Clamp { .. } => {
            operation.inputs().iter().all(|&v| is_effectively_constant(graph, v))
        }
    }
}

struct Details {
    value: Option<Value>,
    start_y: usize,
    data: Tensor4,
}

impl Debug for Details {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Details")
            .field("value", &self.value)
            .field("start_y", &self.start_y)
            .field("shape", &self.data.dim())
            .finish()
    }
}
