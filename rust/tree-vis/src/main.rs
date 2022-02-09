use std::cmp::{max, min};
use std::collections::HashSet;

use board_game::board::Board;
use board_game::games::chess::ChessBoard;
use crossterm::event::{DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEventKind};
use crossterm::execute;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use tui::{Frame, Terminal};
use tui::backend::{Backend, CrosstermBackend};
use tui::buffer::Buffer;
use tui::layout::{Constraint, Margin, Rect};
use tui::style::{Color, Modifier, Style};
use tui::widgets::{Cell, Row, Table, TableState, Widget};

use alpha_zero::mapping::chess::ChessStdMapper;
use alpha_zero::network::cudnn::CudnnNetwork;
use alpha_zero::network::dummy::DummyNetwork;
use alpha_zero::oracle::DummyOracle;
use alpha_zero::zero::node::UctWeights;
use alpha_zero::zero::step::FpuMode;
use alpha_zero::zero::tree::Tree;
use alpha_zero::zero::wrapper::ZeroSettings;

#[derive(Debug)]
struct State<B: Board> {
    tree: Tree<B>,

    prev_nodes: Vec<RenderNode>,

    expanded_nodes: HashSet<usize>,
    selected_node: usize,

    view_offset: usize,
}

#[derive(Debug, Copy, Clone)]
struct RenderNode {
    depth: u32,
    node: usize,
}

fn main() -> std::io::Result<()> {
    let mut state = State {
        prev_nodes: vec![],
        tree: build_tree(),
        expanded_nodes: HashSet::default(),
        selected_node: 0,
        view_offset: 0,
    };

    // println!("{}", state.tree.display(2, true, 200, true));
    // return Ok(());

    state.expanded_nodes.insert(0);
    state.expanded_nodes.insert(1);

    // setup terminal
    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    loop {
        terminal.draw(|f| {
            let area = f.size().inner(&Margin { horizontal: 10, vertical: 10 });
            state.prepare_render(area);
            f.render_widget(&state, area);
        })?;

        let event = crossterm::event::read()?;
        if event == Event::Key(KeyEvent::new(KeyCode::Char('q'), KeyModifiers::empty())) {
            break;
        }

        state.handle_event(event);
    }

    // restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(),LeaveAlternateScreen,DisableMouseCapture)?;
    terminal.show_cursor()?;

    Ok(())
}

impl<B: Board> State<B> {
    fn append_nodes(&self, curr: usize, depth: u32, result: &mut Vec<RenderNode>) {
        result.push(RenderNode { depth, node: curr });

        if self.expanded_nodes.contains(&curr) {
            for c in self.tree[curr].children.iter().flat_map(|r| r.iter()) {
                self.append_nodes(c, depth + 1, result);
            }
        }
    }

    fn prepare_render(&mut self, area: Rect) {
        // collect nodes
        let mut nodes = std::mem::take(&mut self.prev_nodes);
        nodes.clear();
        self.append_nodes(0, 0, &mut nodes);
        self.prev_nodes = nodes;

        // fix offset
        let selected = self.selected_index();
        let margin = 3;

        let offset = (self.view_offset as i32).clamp(
            selected as i32 - area.height as i32 + margin as i32 + 1,
            selected.saturating_sub(margin) as i32,
        );

        assert!(offset >= 0, "offset={}", offset);
        self.view_offset = offset as usize;
    }

    fn selected_index(&self) -> usize {
        self.prev_nodes.iter().position(|n| n.node == self.selected_node).unwrap()
    }

    fn handle_event(&mut self, event: Event) {
        match event {
            Event::Key(key) => {
                match key.code {
                    KeyCode::Up => {
                        let index = self.selected_index();
                        if index != 0 {
                            self.selected_node = self.prev_nodes[index - 1].node;
                        }
                    }
                    KeyCode::Down => {
                        self.selected_node = self.prev_nodes.get(self.selected_index() + 1)
                            .map_or(self.selected_node, |n| n.node);
                    }
                    KeyCode::Right => {
                        self.expanded_nodes.insert(self.selected_node);
                    }
                    KeyCode::Left => {
                        if self.expanded_nodes.contains(&self.selected_node) {
                            self.expanded_nodes.remove(&self.selected_node);
                        } else {
                            if let Some(parent) = self.tree[self.selected_node].parent {
                                self.selected_node = parent;
                                self.expanded_nodes.remove(&parent);
                            }
                        }
                    }
                    _ => (),
                }
            }
            Event::Mouse(mouse) => {
                if mouse.kind == MouseEventKind::Up(MouseButton::Left) {
                    if let Some(node) = self.prev_nodes.get(mouse.row as usize) {
                        self.selected_node = node.node;
                    }
                }
            }
            Event::Resize(_, _) => {}
        }
    }
}

impl<B: Board> Widget for &State<B> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        for y in 0..area.height {
            let full_y = area.y + y;
            let i = y as u32 + self.view_offset as u32;

            if let Some(&RenderNode { node, depth }) = self.prev_nodes.get(i as usize) {
                if node == self.selected_node {
                    let line = Rect::new(area.x, full_y, area.width, 1);
                    let style = Style::default().add_modifier(Modifier::REVERSED);
                    buf.set_style(line, style);
                }

                let state = if self.expanded_nodes.contains(&node) {
                    "v"
                } else {
                    ">"
                };

                buf.set_string(area.x + (depth * 2) as u16, full_y, format!("{} {}", state, node), Style::default());
            }
        }
    }
}

fn build_tree() -> Tree<ChessBoard> {
    let settings = ZeroSettings::new(128, UctWeights::default(), false, FpuMode::Parent);
    let visits = 1_000;

    // let path = "C:/Documents/Programming/STTT/AlphaZero/data/networks/chess_real_1859.onnx";
    // let graph = optimize_graph(&load_graph_from_onnx_path(path), Default::default());
    // let mut network = CudnnNetwork::new(ChessStdMapper, graph, settings.batch_size, Device::new(0));
    let mut network = DummyNetwork;

    let board = ChessBoard::default();
    let tree = settings.build_tree(&board, &mut network, &DummyOracle, |tree| tree.root_visits() >= visits);

    tree
}