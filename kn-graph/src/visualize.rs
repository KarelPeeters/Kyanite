use std::cmp::max;
use std::fmt::{Debug, Formatter};
use std::iter::zip;

use image::{ImageBuffer, Rgb};
use itertools::Itertools;
use ndarray::{ArcArray, Axis, Ix4};
use palette::{LinSrgb, Srgb};

use crate::cpu::ExecutionInfo;
use crate::dtype::{DTensor, Tensor};
use crate::graph::{Graph, Operation, Value};
use crate::shape::Size;

pub type Image = ImageBuffer<Rgb<u8>, Vec<u8>>;
type Tensor4 = ArcArray<f32, Ix4>;

const VERTICAL_PADDING: usize = 5;
const HORIZONTAL_PADDING: usize = 5;

#[derive(Debug)]
pub struct VisTensor {
    pub normalize: bool,
    pub tensor: Tensor<f32>,
}

#[derive(Debug)]
pub struct RenderTensor {
    value: Value,
    original: bool,
    vis_tensor: VisTensor,
}

pub fn visualize_graph_activations(
    graph: &Graph,
    execution: &ExecutionInfo,
    post_process_value: impl Fn(Value, &DTensor) -> Vec<VisTensor>,
    max_images: Option<usize>,
    show_variance: bool,
    print_details: bool,
) -> Vec<Image> {
    let batch_size = execution.batch_size;
    let image_count = max_images.map_or(batch_size, |max_images| max(max_images, batch_size));

    // prevent divide by zero issues later
    if image_count == 0 {
        return vec![];
    }

    let mut total_width = HORIZONTAL_PADDING;
    let mut total_height = VERTICAL_PADDING;

    let mut to_render = vec![];

    for value in execution.values.values() {
        let info = &graph[value.value];

        if !should_show_value(graph, value.value) {
            continue;
        }

        // check whether this is the typical intermediate shape: [B, fixed*]
        let is_intermediate_shape = info.shape.rank() > 0
            && info.shape[0] == Size::BATCH
            && info.shape.dims[1..].iter().all(|d| d.try_unwrap_fixed().is_some());
        if !is_intermediate_shape {
            println!("Skipping value with shape {:?}", info.shape);
            continue;
        }

        let is_input = matches!(&info.operation, Operation::Input { .. });
        let data = value
            .tensor
            .as_ref()
            .expect("Intermediate values should have been kept for visualization");

        if let DTensor::F32(data) = data {
            let vis_tensor = VisTensor {
                normalize: !is_input,
                tensor: data.to_shared(),
            };
            to_render.push(RenderTensor {
                value: value.value,
                original: true,
                vis_tensor,
            });
        }

        for extra_vis_tensor in post_process_value(value.value, data) {
            to_render.push(RenderTensor {
                value: value.value,
                original: false,
                vis_tensor: extra_vis_tensor,
            });
        }
    }

    let mut all_details = vec![];
    for render_tensor in to_render {
        let RenderTensor {
            value,
            original,
            vis_tensor,
        } = render_tensor;
        let VisTensor {
            normalize,
            tensor: data,
        } = vis_tensor;
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

        let details = Details {
            value,
            original,
            start_y,
            normalize,
            data,
        };
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
        if print_details {
            println!("{:?} {:?}", details, graph[details.value]);
        }

        let data = &details.data;
        let (_, channels, height, width) = data.dim();

        if data.iter().any(|x| !x.is_finite()) {
            eprintln!("Warning: encountered non-finite value in {:?}", details);
        }

        // TODO it's still not clear what the best way to normalize/scale/clamp/represent this stuff is
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

                        let gb = if details.normalize {
                            let f = data_norm[(image_i, c, h, w)];
                            let f_norm = ((f + 1.0) / 2.0).clamp(0.0, 1.0);
                            f_norm
                        } else {
                            data[(image_i, c, h, w)].clamp(0.0, 1.0)
                        };
                        let r = if show_variance { s_norm } else { gb };

                        let color = Srgb::from(LinSrgb::new(r, gb, gb));
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

        // TODO what are we even calculating here? mostly questionable heuristics?
        if other_operation.inputs().contains(&value) {
            match other_operation {
                Operation::Input { .. } | Operation::Constant { .. } => unreachable!(),
                &Operation::View { input } => {
                    // check if all commons dims at the start match, which implies the only different is trailing 1s
                    zip(&graph[input].shape.dims, &graph[other].shape.dims).all(|(l, r)| l == r)
                }
                Operation::Broadcast { .. }
                | Operation::Permute { .. }
                | Operation::Slice { .. }
                | Operation::Flip { .. }
                | Operation::Gather { .. }
                | Operation::Concat { .. }
                | Operation::Conv { .. }
                | Operation::MatMul { .. }
                | Operation::Softmax { .. }
                | Operation::Layernorm { .. }
                | Operation::Reduce { .. }
                | Operation::Unary { .. } => false,
                &Operation::Binary { left, right, op: _ } => graph[left].shape != graph[right].shape,
            }
        } else {
            false
        }
    });

    !has_dummy_user
}

fn is_effectively_constant(graph: &Graph, value: Value) -> bool {
    let operation = &graph[value].operation;
    match operation {
        Operation::Input { .. } => false,
        Operation::Constant { .. } => true,
        Operation::View { .. }
        | Operation::Broadcast { .. }
        | Operation::Permute { .. }
        | Operation::Slice { .. }
        | Operation::Flip { .. }
        | Operation::Gather { .. }
        | Operation::Concat { .. }
        | Operation::Conv { .. }
        | Operation::MatMul { .. }
        | Operation::Unary { .. }
        | Operation::Binary { .. }
        | Operation::Softmax { .. }
        | Operation::Layernorm { .. }
        | Operation::Reduce { .. } => operation.inputs().iter().all(|&v| is_effectively_constant(graph, v)),
    }
}

impl VisTensor {
    pub fn abs(tensor: Tensor<f32>) -> VisTensor {
        VisTensor {
            normalize: false,
            tensor,
        }
    }

    pub fn norm(tensor: Tensor<f32>) -> VisTensor {
        VisTensor {
            normalize: true,
            tensor,
        }
    }
}

struct Details {
    value: Value,
    original: bool,
    start_y: usize,

    normalize: bool,
    data: Tensor4,
}

impl Debug for Details {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Details")
            .field("value", &self.value)
            .field("original", &self.original)
            .field("start_y", &self.start_y)
            .field("shape", &self.data.dim())
            .finish()
    }
}
