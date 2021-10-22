use std::cmp::max;

use image::{ImageBuffer, Rgb};
use ndarray::{ArcArray, Axis, Ix4, s};

use crate::cpu::{ExecutionInfo, Tensor};
use crate::graph::{Graph, Value};
use crate::shape::Size;
use crate::wrap_debug::WrapDebug;

pub type Image = ImageBuffer<Rgb<u8>, Vec<u8>>;
type Tensor4 = ArcArray<f32, Ix4>;

const VERTICAL_PADDING: usize = 10;
const HORIZONTAL_PADDING: usize = 5;

pub fn visualize_graph_activations(
    graph: &Graph,
    execution: &ExecutionInfo,
    post_process_value: impl Fn(Value, Tensor) -> Option<Tensor>,
) -> Image {
    let mut total_width = 0;
    let mut total_height = 0;

    let mut selected = vec![];

    for value in execution.values().values() {
        let info = &graph[value.value];

        if should_skip_value(graph, value.value) {
            continue;
        }

        // check whether this is the typical intermediate shape: [B, fixed*]
        let is_intermediate_shape =
            info.shape.rank() > 0 &&
                info.shape[0] == Size::BATCH &&
                info.shape.dims[1..].iter().all(|d| d.try_fixed().is_some());
        if !is_intermediate_shape {
            println!("Skipping value with shape {:?}", info.shape);
            continue;
        }

        let data = value.tensor.to_shared();

        selected.push(data.to_shared());
        if let Some(extra) = post_process_value(value.value, data) {
            selected.push(extra);
        }
    }

    let mut all_details = vec![];
    for data in selected {
        let size = data.len();
        let batch_size = data.dim()[0];

        let data: Tensor4 = match data.ndim() {
            1 => data.reshape((data.len(), 1, 1, 1)),
            2 => data.reshape((batch_size, 1, size / batch_size, 1)),
            3 => data.insert_axis(Axis(1)).into_dimensionality().unwrap(),
            4 => data.into_dimensionality().unwrap(),
            _ => {
                println!("Skipping value with (picked) shape {:?}", data.dim());
                continue;
            }
        };

        let data = if matches!(data.dim(), (_, _, 1, 1)) {
            data.reshape((batch_size, 1, data.dim().1, 1))
        } else {
            data
        };

        let (_, channels, width, height) = data.dim();

        let view_width = channels * width + (channels - 1) * HORIZONTAL_PADDING;
        let view_height = height;

        if total_height != 0 {
            total_height += VERTICAL_PADDING;
        }
        let start_y = total_height;
        total_height += view_height;

        total_width = max(total_width, view_width);

        let details = Details { data: WrapDebug(data), start_y };
        all_details.push(details)
    }

    println!("{}, {}", total_width, total_height);

    let background = Rgb([60, 60, 60]);
    let mut image = ImageBuffer::from_pixel(total_width as u32, total_height as u32, background);

    for details in all_details {
        println!("{:?}", details);

        let data = details.data.inner();

        let (_, channels, width, height) = details.data.inner().dim();

        for c in 0..channels {
            for w in 0..width {
                let x = c * (HORIZONTAL_PADDING + width) + w;
                for h in 0..height {
                    let y = details.start_y + h;

                    let s = data.slice(s!(.., c, w, h)).std(1.0);
                    let s_norm = s.clamp(0.0, 2.0) / 1.0;

                    let f = data[(0, c, w, h)];
                    let f_norm = (f.clamp(-1.0, 1.0) + 1.0) / 2.0;

                    let p = Rgb([(s_norm * 255.0) as u8, (f_norm * 255.0) as u8, (f_norm * 255.0) as u8]);
                    image.put_pixel(x as u32, y as u32, p);
                }
            }
        }
    }

    image
}

fn should_skip_value(_: &Graph, _: Value) -> bool {
    false
}

#[derive(Debug)]
struct Details {
    start_y: usize,
    data: WrapDebug<Tensor4>,
}
