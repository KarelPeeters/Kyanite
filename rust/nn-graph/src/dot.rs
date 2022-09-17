use std::fmt::Write as _;
use std::io::Write;

use crate::graph::{Graph, Operation};

pub fn graph_to_dot(mut f: impl Write, graph: &Graph, hide_const: bool) -> std::io::Result<()> {
    writeln!(f, "digraph {{")?;
    writeln!(f)?;

    for value in graph.values() {
        if hide_const && graph.is_const(value) {
            continue;
        }

        let info = &graph[value];

        let (color, op, mut attrs) = match info.operation {
            Operation::Input { index } => ("gray", "Input", vec![("index", format!("index = {}", index))]),
            Operation::Constant { ref data } => {
                let mut attrs = vec![];
                if data.len() == 1 {
                    attrs.push(("value", format!("{}", data[0])));
                }
                ("gray", "Constant", attrs)
            }
            Operation::View { input: _ } => ("brown", "View", vec![]),
            Operation::Broadcast { input: _ } => ("brown", "Broadcast", vec![]),
            Operation::Permute {
                input: _,
                ref permutation,
            } => {
                let attrs = vec![("Permute", format!("{:?}", permutation))];
                ("brown", "permute", attrs)
            }
            Operation::Slice { input: _, axis, range } => {
                let attrs = vec![("axis", format!("{}", axis)), ("range", format!("{}", range))];
                ("brown", "Slice", attrs)
            }
            Operation::Flip { input: _, axis } => ("brown", "Flip", vec![("axis", format!("{}", axis))]),
            Operation::Gather {
                input: _,
                axis,
                indices: _,
            } => ("yellow", "Gather", vec![("axis", format!("{}", axis))]),
            Operation::Concat { inputs: _, axis } => ("yellow", "Concat", vec![("axis", format!("{}", axis))]),
            Operation::Conv {
                input: _,
                filter: _,
                details,
            } => {
                let mut attrs = vec![("kernel", format!("{}x{}", details.kernel_h, details.kernel_w))];
                if details.has_stride() {
                    attrs.push(("stride", format!("{}x{}", details.stride_y, details.stride_x)));
                }
                if !details.keeps_spatial_shape() {
                    attrs.push(("padding", format!("{}x{}", details.padding_y, details.padding_x)));
                }
                ("blue", "Conv", attrs)
            }
            Operation::MatMul { left: _, right: _ } => ("blue", "MatMul", vec![]),
            Operation::Unary { input: _, op } => ("green", "Unary", vec![("op", format!("{:?}", op))]),
            Operation::Binary { left: _, right: _, op } => ("green", "Binary", vec![("op", format!("{:?}", op))]),
            Operation::Softmax { input: _, axis } => ("purple", "Softmax", vec![("axis", format!("{}", axis))]),
            Operation::Layernorm { input: _, axis, eps: _ } => {
                ("purple", "Layernorm", vec![("axis", format!("{}", axis))])
            }
            Operation::Reduce { input: _, ref axes, op } => (
                "purple",
                "Reduce",
                vec![("op", format!("{:?}", op)), ("axes", format!("{:?}", axes))],
            ),
        };

        attrs.insert(0, ("shape", format!("{}", info.shape)));
        if let Some(output_index) = graph.outputs().iter().position(|&v| v == value) {
            attrs.insert(1, ("output", format!("{}", output_index)));
        }

        let mut table = String::new();
        writeln!(&mut table, "<TABLE BORDER=\"0\">").unwrap();
        writeln!(&mut table, "<TR><TD>{:?}</TD><TD><B>{}</B></TD></TR>", value, op).unwrap();
        for (key, value) in attrs {
            writeln!(&mut table, "<TR><TD>{}</TD><TD>{}</TD></TR>", key, value).unwrap();
        }
        writeln!(&mut table, "</TABLE>").unwrap();

        let label = table;
        writeln!(
            f,
            "{} [label=<{}>, color={:?}, shape=box, width=2]",
            value.index(),
            label,
            color,
        )?;
    }

    writeln!(f)?;

    for value in graph.values() {
        for operand in graph[value].operation.inputs() {
            if hide_const && graph.is_const(operand) {
                continue;
            }

            writeln!(f, "{} -> {}", operand.index(), value.index())?;
        }
    }

    writeln!(f)?;
    writeln!(f, "}}")?;
    Ok(())
}
