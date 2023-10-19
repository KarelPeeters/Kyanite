use std::fmt::{Debug, Formatter};

use ndarray::{ArcArray, ArcArray1, Array4, Data, Dimension, Ix4, s};

use crate::cpu::convolution;
use crate::dtype::DTensor;
use crate::graph::{BinaryOp, ConvDetails, Graph, Operation, Value};
use crate::ndarray::ArrayBase;
use crate::optimizer::{Optimizer, OptimizerSettings};
use crate::optimizer::core::VisitResult;
use crate::shape::Size;

type ArcArray4<A> = ArcArray<A, Ix4>;

impl Optimizer<'_> {
    pub fn try_build_affine_group(&self, old_start: Value) -> VisitResult<Option<AffineGroup>> {
        let output_shape = &self.old_graph[old_start].shape;

        if let &[batch, after_channels, width, height] = output_shape.dims.as_slice() {
            let after_channels = match after_channels.try_unwrap_fixed() {
                None => return Ok(None),
                Some(after_channels) => after_channels,
            };

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
            })?;

            if let Some(old_input) = old_input {
                return Ok(Some(builder.finish(old_input)));
            }
        }

        Ok(None)
    }

    fn grow_affine_group(&self, builder: &mut AffineGroupBuilder, operation: &Operation) -> VisitResult<Option<Value>> {
        match *operation {
            Operation::Conv { input, filter, details } => {
                if let Some(DTensor::F32(filter)) = self.old_graph.as_const(filter) {
                    let filter: ArcArray4<f32> = filter.into_dimensionality().unwrap();

                    if builder.conv.is_none() && details.keeps_spatial_shape() && !details.has_stride() {
                        builder.set_conv(ConvOperation { details, filter });
                        Ok(Some(input))
                    } else {
                        Ok(None)
                    }
                } else {
                    Ok(None)
                }
            }
            Operation::Binary {
                left,
                right,
                op: op @ (BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div),
            } => {
                if let &Operation::Broadcast { input: right_inner } = &self.old_graph[right].operation {
                    if let &[Size::ONE, actual_channels, Size::ONE, Size::ONE] =
                        self.old_graph[right_inner].shape.dims.as_slice()
                    {
                        let channels = builder.current_channels();
                        // TODO it could also just be a broadcasted scalar, maybe fuse this as well
                        if actual_channels == Size::fixed(channels) {
                            if let Some(DTensor::F32(data)) = self.old_graph.as_const(right_inner) {
                                let data: ArcArray4<f32> = data.into_dimensionality().unwrap();
                                assert_eq!(data.shape(), &[1, channels, 1, 1]);
                                let data: ArcArray1<f32> = data.reshape(channels);

                                // no need to skip useless ops here, the graph already does that automatically
                                let affine_op = match op {
                                    BinaryOp::Add => AffineOperation::AddChannel { data },
                                    BinaryOp::Sub => AffineOperation::AddChannel {
                                        data: data.map(|&x| -x).into_shared(),
                                    },
                                    BinaryOp::Mul => AffineOperation::ScaleChannel { data },
                                    BinaryOp::Div => AffineOperation::ScaleChannel {
                                        data: data.map(|&x| 1.0 / x).into_shared(),
                                    },
                                    _ => unreachable!(),
                                };

                                builder.push_affine(affine_op);
                                Ok(Some(left))
                            } else {
                                Ok(None)
                            }
                        } else {
                            Ok(None)
                        }
                    } else {
                        Ok(None)
                    }
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }
}

#[derive(Debug)]
struct AffineGroupBuilder {
    shape: AffineShape,

    before_rev: Vec<AffineOperation>,
    conv: Option<ConvOperation>,
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
        let target = if self.conv.is_some() {
            &mut self.before_rev
        } else {
            &mut self.after_rev
        };
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

    pub fn apply_fused(self, settings: OptimizerSettings, graph: &mut Graph, input: Value) -> Value {
        if let Some(conv) = self.conv {
            let before = fuse_affine_list(self.shape.before_channels, &self.before);
            let after = fuse_affine_list(self.shape.after_channels, &self.after);

            apply_fused_conv(settings, graph, input, before, conv, after)
        } else {
            assert!(self.before.is_empty());
            let after = fuse_affine_list(self.shape.after_channels, &self.after);

            after.apply(graph, input)
        }
    }
}

fn apply_fused_conv(
    settings: OptimizerSettings,
    graph: &mut Graph,
    input: Value,
    before: ScaleBias,
    conv: ConvOperation,
    after: ScaleBias,
) -> Value {
    let details = conv.details;
    assert!(!details.has_stride());

    let mut total_filter = conv.filter;

    // fuse output scale into kernel
    if let Some(after_scale) = after.scale {
        for k in 0..details.output_channels {
            total_filter
                .slice_mut(s![k, .., .., ..])
                .mapv_inplace(|x| x * after_scale[k]);
        }
    }

    let bias_after_shaped = after.bias.map(|bias| bias.into_shape((1, details.output_channels, 1, 1)).unwrap());

    // try to pull input bias through
    match pull_bias_through_conv(settings, details, before.bias, &total_filter) {
        Ok(bias_before_as_after) => {
            // great, combine both biases
            let total_bias_after: Option<ArcArray4<f32>> = match (bias_before_as_after, bias_after_shaped) {
                (Some(bias_before_as_after), Some(bias_after_shaped)) => Some(bias_before_as_after + bias_after_shaped),
                (Some(bias_before_as_after), None) => Some(bias_before_as_after),
                (None, Some(bias_after_shaped)) => Some(bias_after_shaped),
                (None, None) => None,
            };

            // fuse input scale into kernel
            if let Some(before_scale) = before.scale {
                for c in 0..details.input_channels {
                    total_filter
                        .slice_mut(s![.., c, .., ..])
                        .mapv_inplace(|x| x * before_scale[c]);
                }
            }

            // put everything into the graph
            let mut curr = input;

            //    conv
            let value_filter = graph.constant_tensor(DTensor::F32(total_filter.into_dyn()));
            curr = graph.conv(curr, value_filter, 1, 1, details.padding_y, details.padding_x);

            //    bias
            if let Some(total_bias_after) = total_bias_after {
                let value_bias = graph.constant_tensor(DTensor::F32(total_bias_after.into_dyn()));
                curr = graph.binary(BinaryOp::Add, curr, value_bias);
            }

            curr
        }
        Err(bias_before) => {
            // put everything into the graph
            let before = ScaleBias {
                scale: before.scale,
                bias: Some(bias_before),
            };

            //    before
            let mut curr = input;
            curr = before.apply(graph, curr);

            //    conv
            let value_filter = graph.constant_tensor(DTensor::F32(total_filter.into_dyn()));
            curr = graph.conv(curr, value_filter, 1, 1, details.padding_y, details.padding_x);

            //    bias
            if let Some(bias_after_shaped) = bias_after_shaped {
                let value_bias_after = graph.constant_tensor(DTensor::F32(bias_after_shaped.into_dyn().into_shared()));
                curr = graph.binary(BinaryOp::Add, curr, value_bias_after);
            }

            curr
        }
    }
}

fn pull_bias_through_conv(
    settings: OptimizerSettings,
    details: ConvDetails,
    before: Option<ArcArray1<f32>>,
    filter: &ArcArray4<f32>,
) -> Result<Option<ArcArray4<f32>>, ArcArray1<f32>> {
    assert!(!details.has_stride());

    let before = match before {
        Some(before) => before,
        None => return Ok(None),
    };

    if is_entirely(&before, 0.0) {
        // we don't need to expand the shape even if there is padding, so immediately return 0 here
        Ok(None)
    } else if details.padding_y == 0 && details.padding_x == 0 {
        // the bias will be the same for each output (x,y), so we can keep a single bias vector
        let before_shaped = before.reshape((1, details.input_channels, 1, 1));

        let result = Array4::from_shape_fn(
            (1, details.output_channels, 1, 1),
            |(_, k, _, _)| {
                (0..details.input_channels)
                    .map(|c| filter.slice(s![k, c, .., ..]).sum() * before_shaped[(0, c, 0, 0)])
                    .sum()
            },
        );
        Ok(Some(result.into_shared()))
    } else if settings.force_bias_through_conv {
        // the bias will be different for the edges, so it needs to be full-sized
        let before_shaped = before.into_shape((1, details.input_channels, 1, 1)).unwrap();

        let before_broadcast = before_shaped
            .broadcast((1usize, details.input_channels, details.input_h, details.input_w))
            .unwrap();

        let result = convolution(details, before_broadcast, filter.view());
        Ok(Some(result.into_shared()))
    } else {
        Err(before)
    }
}

/// Represents a scale and bias operation, in that order.
///
/// `None` means that the operation is a no-op. We could use a constant of the correct shape instead,
/// but this allows us to skip extra calculations that slow down the optimizer.
struct ScaleBias {
    scale: Option<ArcArray1<f32>>,
    bias: Option<ArcArray1<f32>>,
}

impl ScaleBias {
    fn apply(self, graph: &mut Graph, input: Value) -> Value {
        let mut curr = input;

        if let Some(scale) = self.scale {
            let scale = graph.constant_tensor(DTensor::F32(scale.reshape(vec![1, scale.len(), 1, 1])));
            curr = graph.binary(BinaryOp::Mul, curr, scale);
        }

        if let Some(bias) = self.bias {
            let bias = graph.constant_tensor(DTensor::F32(bias.reshape(vec![1, bias.len(), 1, 1])));
            curr = graph.binary(BinaryOp::Add, curr, bias);
        }

        curr
    }
}

fn is_entirely<S: Data<Elem = f32>, D: Dimension>(array: &ArrayBase<S, D>, value: f32) -> bool {
    array.iter().all(|&x| x == value)
}

fn fuse_affine_list<'a>(channels: usize, operations: &[AffineOperation]) -> ScaleBias {
    // TODO should these be shared or local?
    let mut total_scale = None;
    let mut total_bias = None;

    for op in operations {
        match op {
            AffineOperation::AddChannel { data } => {
                // bias is added
                total_bias = Some(total_bias.map_or_else(|| data.clone(), |total_bias| total_bias + data));
            }
            AffineOperation::ScaleChannel { data } => {
                // scale is multiplied
                total_scale = Some(total_scale.map_or_else(|| data.clone(), |total_scale| total_scale * data));
                // bias is multiplied, if it is nonzero
                total_bias = total_bias.map(|total_bias| total_bias * data);
            }
        }
    }

    // TODO is this really necessary?
    if let Some(total_scale) = total_scale.as_mut() {
        assert_eq!(total_scale.len(), channels);
    }
    if let Some(total_bias) = total_bias.as_mut() {
        assert_eq!(total_bias.len(), channels);
    }

    ScaleBias {
        scale: total_scale,
        bias: total_bias,
    }
}

struct ConvOperation {
    details: ConvDetails,
    filter: ArcArray4<f32>,
}

#[allow(dead_code)]
#[derive(Debug)]
struct AffineShape {
    batch: Size,
    before_channels: usize,
    after_channels: usize,
    width: Size,
    height: Size,
}

enum AffineOperation {
    AddChannel { data: ArcArray1<f32> },
    ScaleChannel { data: ArcArray1<f32> },
}

fn reversed<T>(mut v: Vec<T>) -> Vec<T> {
    v.reverse();
    v
}

impl Debug for ConvOperation {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "ConvOperation {{ details: {:?}, filter: ... }}", self.details)
    }
}

impl Debug for AffineOperation {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            AffineOperation::AddChannel { data } => write!(f, "AddChannel {{ len: {}, data: ... }}", data.len()),
            AffineOperation::ScaleChannel { data } => write!(f, "ScaleChannel {{ len: {}, data: ... }}", data.len()),
        }
    }
}
