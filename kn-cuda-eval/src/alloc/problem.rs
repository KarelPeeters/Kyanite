use std::collections::HashSet;
use std::fmt::{Debug, Formatter};

use itertools::{enumerate, Itertools};

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Buffer(usize);

#[derive(Clone)]
pub struct AllocProblem {
    buffers: Vec<BufferInfo>,

    total_size: usize,
    min_alignment: Option<usize>,
    max_alignment: Option<usize>,

    life_points: HashSet<usize>,
}

#[derive(Debug, Clone)]
pub struct BufferInfo {
    pub size: usize,
    // TODO or just enforce some minimum alignment on all buffers?
    pub alignment: usize,
    pub life_start: usize,
    pub life_end: usize,
}

#[derive(Debug, Clone)]
pub struct AllocSolution {
    offsets: Vec<usize>,
}

impl BufferInfo {
    fn assert_valid(&self) {
        assert!(self.alignment > 0);
        assert!(self.life_start <= self.life_end);
    }
}

impl AllocProblem {
    pub fn new() -> AllocProblem {
        AllocProblem {
            buffers: Vec::new(),
            total_size: 0,
            min_alignment: None,
            max_alignment: None,
            life_points: HashSet::new(),
        }
    }

    pub fn add_buffer(&mut self, info: BufferInfo) {
        info.assert_valid();

        self.total_size += info.size;
        self.min_alignment = Some(self.min_alignment.map_or(info.alignment, |x| x.min(info.alignment)));
        self.max_alignment = Some(self.max_alignment.map_or(info.alignment, |x| x.max(info.alignment)));
        self.life_points.insert(info.life_start);
        self.life_points.insert(info.life_end);

        self.buffers.push(info);
    }

    pub fn life_overlap(&self, a: Buffer, b: Buffer) -> bool {
        let a_info = &self.buffers[a.0];
        let b_info = &self.buffers[b.0];
        overlaps_inclusive(a_info.life_start, a_info.life_end, b_info.life_start, b_info.life_end)
    }

    pub fn assert_valid_solution(&self, solution: &AllocSolution) {
        assert_eq!(solution.offsets.len(), self.buffers.len());

        // check offsets
        for (i, &offset) in solution.offsets.iter().enumerate() {
            let info = &self.buffers[i];
            assert_eq!(offset % info.alignment, 0);
            assert!(offset + info.size <= self.total_size);
        }

        // check overlaps
        // TODO optimize this O(N^2) check, probably by keeping track of overlaps in the problem
        for a in 0..self.buffers.len() {
            let a_info = &self.buffers[a];
            let a_offset = solution.offsets[a];

            // exclude self and already checked pairs
            for b in (a + 1)..self.buffers.len() {
                let b_info = &self.buffers[b];
                let b_offset = solution.offsets[b];

                let overlap_life = self.life_overlap(Buffer(a), Buffer(b));
                let overlap_mem = overlaps_exclusive(
                    a_offset, a_offset + a_info.size,
                    b_offset, b_offset + b_info.size,
                );
                assert!(
                    !(overlap_life && overlap_mem),
                    "{:?} and {:?} overlap in both memory ({:?} and {:?}) and time ({:?} and {:?})",
                    Buffer(a), Buffer(b),
                    a_offset..a_offset + a_info.size, b_offset..b_offset + b_info.size,
                    a_info.life_start..=a_info.life_end, b_info.life_start..=b_info.life_end,
                );
            }
        }
    }

    #[cfg(feature = "plot")]
    pub fn plot(&self) {
        use plotters::backend::SVGBackend;
        use plotters::drawing::IntoDrawingArea;
        use plotters::coord::types::RangedCoordusize;
        use plotters::element::Rectangle;
        use plotters::prelude::Cartesian2d;
        use plotters::style::{Color, RGBColor};

        let root = SVGBackend::new("ignored/problem.svg", (1024, 768)).into_drawing_area();
        // let root = root.margin(10, 10, 10, 10);

        let (&min_life, &max_life) = self.life_points.iter().minmax().into_option().unwrap_or((&0, &1));

        let spec: Cartesian2d<RangedCoordusize, RangedCoordusize> = Cartesian2d::new(
            RangedCoordusize::from(min_life..max_life),
            RangedCoordusize::from(0..self.buffers.len()),
            root.get_pixel_range(),
        );
        let root = root.apply_coord_spec(spec);

        let colors_str = ["1f77b4", "ff7f0e", "2ca02c", "d62728", "9467bd", "8c564b", "e377c2", "7f7f7f", "bcbd22", "17becf"];
        let colors = colors_str.map(|hex| {
            let x = u32::from_str_radix(hex, 16).unwrap();
            RGBColor((x >> 16) as u8, ((x >> 8) & 0xFF) as u8, (x & 0xFF) as u8)
        });

        for (i, buffer) in enumerate(&self.buffers) {
            let start = buffer.life_start;
            let end = buffer.life_end;
            root.draw(&Rectangle::new([(start, i), (end, (i + 1))], colors[i % colors.len()].filled())).unwrap();
        }

        root.present().unwrap();
    }
}

fn overlaps_exclusive(a_start: usize, a_end: usize, b_start: usize, b_end: usize) -> bool {
    a_start < b_end && b_start < a_end
}

fn overlaps_inclusive(a_start: usize, a_end: usize, b_start: usize, b_end: usize) -> bool {
    a_start <= b_end && b_start < a_end
}

impl Debug for AllocProblem {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let AllocProblem {
            buffers,
            total_size,
            min_alignment,
            max_alignment,
            life_points,
        } = self;

        writeln!(f, "AllocProblem {{")?;
        writeln!(f, "  total_size: {}", total_size)?;
        writeln!(f, "  min_alignment: {:?}", min_alignment)?;
        writeln!(f, "  max_alignment: {:?}", max_alignment)?;
        writeln!(f, "  life_points: {}", life_points.len())?;
        writeln!(f, "  buffers: [")?;
        for (i, info) in buffers.iter().enumerate() {
            writeln!(f, "    Buffer({}): {:?}", i, info)?;
        }
        writeln!(f, "  ]")?;
        writeln!(f, "}}")?;

        Ok(())
    }
}