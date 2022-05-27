use board_game::board::{Board, Outcome};
use board_game::wdl::{Flip, OutcomeWDL, POV, WDL};
use futures::executor::{block_on, ThreadPoolBuilder};
use futures::task::SpawnExt;
use std::fmt::{Display, Formatter};
use tabled::{Margin, Style};

use kz_core::bot::AsyncBot;

pub type BoxBotFn<B> = Box<dyn Fn() -> BoxBot<B>>;
pub type BoxBot<B> = Box<dyn AsyncBot<B> + Send>;

#[derive(Debug)]
pub struct Tournament<B: Board> {
    pub bot_names: Vec<String>,
    pub pos_count: usize,
    pub rounds: Vec<Round<B>>,

    pub total_wdl: Vec<WDL<usize>>,
    pub grid_wdl: Vec<Vec<WDL<usize>>>,
}

#[derive(Debug)]
pub struct Round<B: Board> {
    pub i: usize,
    pub j: usize,
    pub start: B,
    pub moves: Vec<B::Move>,
    outcome: Outcome,
}

pub fn box_bot<B: Board, T: AsyncBot<B> + Send + 'static>(
    f: impl Fn() -> T + 'static,
) -> Box<dyn Fn() -> Box<dyn AsyncBot<B> + Send>> {
    Box::new(move || Box::new(f()))
}

pub fn run_tournament<S: Display, B: Board>(
    bots: Vec<(S, BoxBotFn<B>)>,
    start_positions: Vec<B>,
    thread_count: usize,
    self_games: bool,
) -> Tournament<B> {
    let bot_count = bots.len();
    let pos_count = start_positions.len();

    let (bot_names, bots) = bots.into_iter().map(|(s, b)| (s.to_string(), b)).unzip();
    let rounds = run_rounds(bots, &start_positions, thread_count, self_games);

    let mut total_wdl = vec![WDL::<usize>::default(); bot_count];
    let mut grid_wdl = vec![vec![WDL::<usize>::default(); bot_count]; bot_count];

    for round in &rounds {
        let wdl_i = round.outcome_i().to_wdl();
        let wdl_j = round.outcome_i().flip().to_wdl();

        total_wdl[round.i] += wdl_i;
        total_wdl[round.j] += wdl_j;
        grid_wdl[round.i][round.j] += wdl_i;
        grid_wdl[round.j][round.i] += wdl_j;
    }

    Tournament {
        bot_names,
        pos_count,
        rounds,
        total_wdl,
        grid_wdl,
    }
}

fn run_rounds<B: Board>(
    bots: Vec<BoxBotFn<B>>,
    start_positions: &Vec<B>,
    thread_count: usize,
    self_games: bool,
) -> Vec<Round<B>> {
    let pool = ThreadPoolBuilder::new()
        .name_prefix("tournament")
        .pool_size(thread_count)
        .create()
        .unwrap();

    let mut handles = vec![];

    // this nested loop automatically includes flipped bots
    for i in 0..bots.len() {
        for j in 0..bots.len() {
            for start in start_positions {
                if (i == j) ^ self_games {
                    break;
                }

                let start = start.clone();
                let bot_i = bots[i]();
                let bot_j = bots[j]();

                let handle = pool
                    .spawn_with_handle(async move { run_game(start, i, j, bot_i, bot_j).await })
                    .unwrap();
                handles.push(handle);
            }
        }
    }

    let handle = pool
        .spawn_with_handle(async move {
            let mut rounds = vec![];
            for handle in handles {
                rounds.push(handle.await);
            }
            rounds
        })
        .unwrap();

    block_on(handle)
}

async fn run_game<B: Board>(start: B, i: usize, j: usize, mut bot_i: BoxBot<B>, mut bot_j: BoxBot<B>) -> Round<B> {
    let mut board = start.clone();
    let mut moves = vec![];

    let outcome = loop {
        if let Some(outcome) = board.outcome() {
            break outcome;
        }

        let mv_i = bot_i.select_move(&board).await;
        moves.push(mv_i);
        board.play(mv_i);

        if let Some(outcome) = board.outcome() {
            break outcome;
        }

        let mv_j = bot_j.select_move(&board).await;
        moves.push(mv_j);
        board.play(mv_j);
    };

    Round {
        i,
        j,
        start,
        moves,
        outcome,
    }
}

impl<B: Board> Round<B> {
    fn outcome_i(&self) -> OutcomeWDL {
        self.outcome.pov(self.start.next_player())
    }
}

impl<B: Board> Tournament<B> {
    fn bot_count(&self) -> usize {
        self.bot_names.len()
    }
}

impl<B: Board> Display for Tournament<B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let bot_count = self.bot_count();
        let format_wdl = |wdl: WDL<usize>| {
            let wdl = wdl.cast::<f32>().normalized();

            if wdl.sum().is_nan() {
                "".to_string()
            } else {
                format!("{:.2}/{:.2}/{:.2}", wdl.win, wdl.draw, wdl.loss)
            }
        };

        let mut cols = vec!["", "total"];
        for j in 0..bot_count {
            cols.push(&self.bot_names[j]);
        }

        let mut table = tabled::builder::Builder::default().set_columns(cols);

        for i in 0..bot_count {
            let mut row = vec![self.bot_names[i].clone(), format_wdl(self.total_wdl[i])];
            for j in 0..bot_count {
                row.push(format_wdl(self.grid_wdl[i][j]));
            }
            table = table.add_record(row);
        }

        let table = table.build().with(Margin::new(4, 0, 0, 0)).with(Style::modern());

        writeln!(f, "Tournament {{")?;
        writeln!(f, "  pos_count: {}", self.pos_count)?;
        writeln!(f, "  total_games: {}", self.rounds.len())?;
        writeln!(f, "  results:")?;
        writeln!(f, "{}", table)?;
        writeln!(f, "}}")?;

        Ok(())
    }
}
