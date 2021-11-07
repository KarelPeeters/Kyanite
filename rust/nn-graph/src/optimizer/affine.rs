use ndarray::{Array1, Array4, ArrayView1, Data, Dimension, s};

use crate::cpu::convolution;
use crate::graph::{ConvDetails, Graph, Operation, Value};
use crate::ndarray::ArrayBase;
use crate::optimizer::Optimizer;
use crate::shape::{Shape, Size};

impl Optimizer<'_> {
    pub fn try_build_affine_group(&self, old_start: Value) -> Option<AffineGroup> {
        let output_shape = &self.old_graph[old_start].shape;

        if let &[batch, after_channels, width, height] = output_shape.dims.as_slice() {
            let after_channels = after_channels.try_unwrap_fixed()?;

            let initial_shape = AffineShape {
                batch,
                before_channels: after_channels,
                after_channels,
                width,
                height,
            };
            let mut builder = AffineGroupBuilder::new(initial_shape);

            let old_input = self.follow_if(old_start, |_, _, operation| {
                self.grow_affine_group(&mut builder, operation)
            });

            if let Some(old_input) = old_input {
                return Some(builder.finish(old_input));
            }
        }

        None
    }

    fn grow_affine_group(&self, builder: &mut AffineGroupBuilder, operation: &Operation) -> Option<Value> {
        match operation {
            &Operation::Conv { input, filter, details: conv_shape } => {
                if let Some(filter) = self.follow_const(filter) {
                    if builder.conv.is_none() && conv_shape.output_size == conv_shape.input_size {
                        builder.set_conv(ConvOperation { details: conv_shape, filter: filter.to_owned() });
                        Some(input)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            &Operation::Add { left, right, subtract } => {
                if let &[Size::ONE, channels, Size::ONE, Size::ONE] = self.old_graph[right].shape.dims.as_slice() {
                    assert_eq!(channels, Size::fixed(builder.current_channels()));

                    if let Some(data) = self.follow_const(right) {
                        builder.push_affine(AffineOperation::AddChannel { data: data.to_owned(), subtract });
                        Some(left)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            &Operation::Mul { left, right } => {
                if let &[Size::ONE, channels, Size::ONE, Size::ONE] = self.old_graph[right].shape.dims.as_slice() {
                    assert_eq!(channels, Size::fixed(builder.current_channels()));

                    if let Some(data) = self.follow_const(right) {
                        builder.push_affine(AffineOperation::ScaleChannel { data: data.to_owned() });
                        Some(left)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

#[derive(Debug)]
struct AffineGroupBuilder {
    shape: AffineShape,

    conv: Option<ConvOperation>,

    before_rev: Vec<AffineOperation>,
    after_rev: Vec<AffineOperation>,
}

impl AffineGroupBuilder {
    fn new(initial_shape: AffineShape) -> Self {
        AffineGroupBuilder {
            shape: initial_shape,
            conv: None,
            before_rev: vec![],
            after_rev: vec![],
        }
    }

    fn current_channels(&self) -> usize {
        // after a conv is set this is overwritten, before that it is initialized to after_channels
        self.shape.before_channels
    }

    fn set_conv(&mut self, conv: ConvOperation) {
        assert!(self.conv.is_none());
        self.shape.before_channels = conv.details.input_channels;
        self.conv = Some(conv);
    }

    fn push_affine(&mut self, operation: AffineOperation) {
        let target = if self.conv.is_some() { &mut self.before_rev } else { &mut self.after_rev };
        target.push(operation);
    }

    fn finish(self, old_input: Value) -> AffineGroup {
        AffineGroup {
            shape: self.shape,
            old_input,
            before: reversed(self.before_rev),
            conv: self.conv,
            after: reversed(self.after_rev),
        }
    }
}

#[derive(Debug)]
pub struct AffineGroup {
    shape: AffineShape,
    old_input: Value,

    before: Vec<AffineOperation>,
    conv: Option<ConvOperation>,
    after: Vec<AffineOperation>,
}

impl AffineGroup {
    pub fn old_input(&self) -> Value {
        self.old_input
    }

    pub fn apply_fused(self, graph: &mut Graph, input: Value) -> Value {
        if let Some(conv) = self.conv {
            let before = fuse_affine_list(self.shape.before_channels, &self.before);
            let after = fuse_affine_list(self.shape.after_channels, &self.after);

            apply_fused_conv(graph, input, before, conv, after)
        } else {
            assert!(self.before.is_empty());
            let after = fuse_affine_list(self.shape.after_channels, &self.after);

            after.apply(graph, input)
        }
    }
}

fn apply_fused_conv(graph: &mut Graph, input: Value, before: ScaleBias, conv: ConvOperation, after: ScaleBias) -> Value {
    let details = conv.details;

    let mut total_filter = Array4::from_shape_vec(
        details.kernel_shape(),
        conv.filter,
    ).unwrap();

    // fuse output scale into kernel
    for k in 0..details.output_channels {
        total_filter.slice_mut(s![k, .., .., ..]).mapv_inplace(|x| x * after.scale[k]);
    }

    // calculate total output bias
    let bias_before_as_after = pull_bias_through_conv(details, before.bias, &total_filter);
    let bias_after_shaped = after.bias.into_shape((1, details.output_channels, 1, 1)).unwrap();
    let total_bias_after: Array4<f32> = bias_before_as_after + bias_after_shaped;

    // fuse input scale into kernel
    let before_scale = before.scale;
    for c in 0..details.input_channels {
        total_filter.slice_mut(s![.., c, .., ..]).mapv_inplace(|x| x * before_scale[c]);
    }

    // put everything into the graph
    let mut curr = input;
    let value_filter = graph.constant(Shape::fixed(total_filter.shape()), total_filter.into_raw_vec());
    curr = graph.conv(curr, value_filter, details.padding);

    println!("{:?}", details);
    println!("{:?}", total_bias_after.shape());

    if !is_entirely(&total_bias_after, 0.0) {
        let value_bias = graph.constant(Shape::fixed(total_bias_after.shape()), total_bias_after.into_raw_vec());
        curr = graph.add(curr, value_bias);
    }

    curr
}

fn pull_bias_through_conv(details: ConvDetails, before: Array1<f32>, filter: &Array4<f32>) -> Array4<f32> {
    let before_shaped = before.into_shape((1, details.input_channels, 1, 1)).unwrap();

    if is_entirely(&before_shaped, 0.0) {
        // we don't need to expand the shape even if there is padding, so immediately return 0 here
        Array4::zeros((1, details.output_channels, 1, 1))
    } else if details.padding == 0 {
        // the bias will be the same for each output (x,y), so we can keep a single bias vector
        Array4::from_shape_fn((1, details.output_channels, 1, 1), |(_, k, _, _)| {
            (0..details.input_channels)
                .map(|c| filter.slice(s![k, c, .., ..]).sum() * before_shaped[(0, c, 0, 0)])
                .sum()
        })
    } else {
        // the bias will be different for the edges, so it needs to be full-sized
        let before_broadcast = before_shaped
            .broadcast((1, details.input_channels, details.input_size, details.input_size)).unwrap();

        convolution(details, before_broadcast, filter.view())
    }
}

struct ScaleBias {
    scale: Array1<f32>,
    bias: Array1<f32>,
}

impl ScaleBias {
    fn apply(self, graph: &mut Graph, input: Value) -> Value {
        let const_shape = Shape::fixed(&[1, self.scale.len(), 1, 1]);

        let mut curr = input;

        if !is_entirely(&self.scale, 1.0) {
            let scale = graph.constant(const_shape.clone(), self.scale.to_vec());
            curr = graph.mul(curr, scale);
        }

        if !is_entirely(&self.bias, 0.0) {
            let bias = graph.constant(const_shape.clone(), self.bias.to_vec());
            curr = graph.add(curr, bias);
        }

        curr
    }
}

fn is_entirely<S: Data<Elem=f32>, D: Dimension>(array: &ArrayBase<S, D>, value: f32) -> bool {
    array.iter().all(|&x| x == value)
}

fn fuse_affine_list<'a>(channels: usize, operations: impl IntoIterator<Item=&'a AffineOperation>) -> ScaleBias {
    let mut total_scale = Array1::ones(channels);
    let mut total_bias = Array1::zeros(channels);

    for op in operations {
        match op {
            AffineOperation::AddChannel { data, subtract } => {
                let data = ArrayView1::from_shape(channels, data).unwrap();
                if *subtract {
                    total_bias -= &data;
                } else {
                    total_bias += &data;
                }
            }
            AffineOperation::ScaleChannel { data } => {
                let data = ArrayView1::from_shape(channels, data).unwrap();

                total_scale *= &data;
                total_bias *= &data;
            }
        }
    }

    ScaleBias {
        scale: total_scale,
        bias: total_bias,
    }
}

#[derive(Debug)]
struct ConvOperation {
    details: ConvDetails,
    filter: Vec<f32>,
}

#[derive(Debug)]
struct AffineShape {
    batch: Size,
    before_channels: usize,
    after_channels: usize,
    width: Size,
    height: Size,
}

#[derive(Debug)]
enum AffineOperation {
    AddChannel { data: Vec<f32>, subtract: bool },
    ScaleChannel { data: Vec<f32> },
}

fn reversed<T>(mut v: Vec<T>) -> Vec<T> {
    v.reverse();
    v
}
