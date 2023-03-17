use std::fmt::{Display, Formatter};
use std::time::Instant;

use board_game::board::{Board, Outcome};
use board_game::pov::{NonPov, Pov};
use board_game::util::rating::elo_from_wdl;
use board_game::wdl::{OutcomeWDL, WDL};
use flume::Sender;
use futures::executor::{block_on, ThreadPoolBuilder};
use futures::task::SpawnExt;
use futures::StreamExt;
use itertools::Itertools;
use tabled::builder::Builder;
use tabled::{Margin, Style, Table};

use kz_core::bot::AsyncBot;

pub type BoxBotFn<B> = Box<dyn Fn() -> BoxBot<B>>;
pub type BoxBot<B> = Box<dyn AsyncBot<B> + Send>;

// TODO write a better NN tournament runner that:
//   * prefers playing on games that are the least far along
//   * has some limit to the amount of concurrent games
//   * automatically picks and changes the batch size of the NNs
//   * prefers running the NN with the most evals or the most full batch

#[derive(Debug)]
pub struct Tournament<B: Board> {
    pub bot_names: Vec<String>,
    pub pos_count: usize,
    pub self_games: bool,
    pub flip_games: bool,

    pub rounds: Vec<Round<B>>,

    pub grid_wdl: Vec<Vec<WDL<usize>>>,
    pub grid_wdl_decisive: Option<Vec<Vec<WDL<usize>>>>,
}

#[derive(Debug)]
pub struct Round<B: Board> {
    pub id: RoundId,
    pub start: B,
    pub moves: Vec<B::Move>,
    pub outcome: Outcome,
}

#[derive(Debug, Copy, Clone)]
pub struct RoundId {
    pub i: usize,
    pub j: usize,
    pub s: usize,
    pub global: usize,
}

#[derive(Debug, Copy, Clone)]
enum Message {
    FinishedRound { id: RoundId },
    FinishedMove { id: RoundId, mv: usize },
}

pub fn box_bot<B: Board, T: AsyncBot<B> + Send + 'static>(
    f: impl Fn() -> T + 'static,
) -> Box<dyn Fn() -> Box<dyn AsyncBot<B> + Send>> {
    Box::new(move || Box::new(f()))
}

pub fn run_tournament<S: Display, B: Board, F: FnMut() + Send + 'static>(
    bots: Vec<(S, BoxBotFn<B>)>,
    start_positions: Vec<B>,
    limit_threads: Option<usize>,
    self_games: bool,
    flip_games: bool,
    on_print: F,
) -> Tournament<B> {
    let bot_count = bots.len();
    let pos_count = start_positions.len();

    let (bot_names, bots) = bots.into_iter().map(|(s, b)| (s.to_string(), b)).unzip();
    let rounds = run_rounds(bots, &start_positions, limit_threads, self_games, flip_games, on_print);

    let mut grid_wdl = vec![vec![WDL::default(); bot_count]; bot_count];
    let mut grid_wdl_decisive = if flip_games {
        Some(vec![vec![WDL::default(); bot_count]; bot_count])
    } else {
        None
    };

    for round in &rounds {
        let id = round.id;
        let wdl_i = round.outcome_i().to_wdl();
        let wdl_j = round.outcome_i().flip().to_wdl();

        grid_wdl[id.i][id.j] += wdl_i;
        grid_wdl[id.j][id.i] += wdl_j;

        if let Some(grid_wdl_decisive) = &mut grid_wdl_decisive {
            let other = rounds.iter().find(|other| RoundId::is_flipped_pair(id, other.id));
            if let Some(other) = other {
                if round.outcome != other.outcome {
                    grid_wdl_decisive[id.i][id.j] += wdl_i;
                    grid_wdl_decisive[id.j][id.i] += wdl_j;
                }
            }
        }
    }

    Tournament {
        bot_names,
        pos_count,
        self_games,
        flip_games,
        rounds,
        grid_wdl,
        grid_wdl_decisive,
    }
}

fn run_rounds<B: Board>(
    bots: Vec<BoxBotFn<B>>,
    start_positions: &Vec<B>,
    limit_threads: Option<usize>,
    self_games: bool,
    flip_games: bool,
    mut on_print: impl FnMut() + Send + 'static,
) -> Vec<Round<B>> {
    let mut builder = ThreadPoolBuilder::new();
    builder.name_prefix("tournament");
    if let Some(thread_count) = limit_threads {
        builder.pool_size(thread_count);
    }
    let pool = builder.create().unwrap();

    let mut handles = vec![];
    let (sender, receiver) = flume::bounded(8);

    let mut started_games = 0;
    for i in 0..bots.len() {
        let j_start = if flip_games { 0 } else { i };
        for j in j_start..bots.len() {
            for (s, start) in start_positions.iter().enumerate() {
                if (i == j) && !self_games {
                    break;
                }
                let global = started_games;
                let id = RoundId { i, j, s, global };
                started_games += 1;

                let sender = sender.clone();
                let start = start.clone();
                let bot_i = bots[i]();
                let bot_j = bots[j]();

                let handle = pool
                    .spawn_with_handle(async move { run_round(id, bot_i, bot_j, start, sender).await })
                    .unwrap();
                handles.push(handle);
            }
        }
    }

    let started_games = started_games;
    drop(bots);
    drop(sender);

    pool.spawn_ok(async move {
        let mut finished_games = 0;
        let mut moves = vec![0; started_games];

        let mut last_time = Instant::now();
        let mut delta_moves = 0;

        while let Some(message) = receiver.stream().next().await {
            match message {
                Message::FinishedRound { id } => {
                    moves[id.global] = usize::MAX;
                    finished_games += 1;
                }
                Message::FinishedMove { id, mv } => {
                    moves[id.global] = mv;
                    delta_moves += 1;
                }
            }

            if finished_games == started_games {
                break;
            }

            let now = Instant::now();
            let delta = (now - last_time).as_secs_f32();

            if delta >= 1.0 {
                let move_tp = delta_moves as f32 / delta;
                let running_games = started_games - finished_games;

                let moves = moves.iter().copied().filter(|&m| m != usize::MAX);
                let (moves_min, moves_max) = moves.clone().minmax().into_option().unwrap();
                let moves_avg = moves.sum::<usize>() as f32 / running_games as f32;

                println!("Throughput: {} moves/s", move_tp);
                println!("  finished {}/{} games", finished_games, started_games);
                println!(
                    "  running moves per game: min {} avg {} max {}",
                    moves_min, moves_avg, moves_max
                );
                on_print();

                last_time = now;
                delta_moves = 0;
            }
        }
    });

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

async fn run_round<B: Board>(
    id: RoundId,
    mut bot_i: BoxBot<B>,
    mut bot_j: BoxBot<B>,
    start: B,
    sender: Sender<Message>,
) -> Round<B> {
    let mut board = start.clone();
    let mut moves = vec![];

    let outcome = loop {
        if let Some(outcome) = board.outcome() {
            break outcome;
        }

        let mv_i = bot_i.select_move(&board).await;
        moves.push(mv_i);
        board.play(mv_i);

        sender
            .send_async(Message::FinishedMove { id, mv: moves.len() })
            .await
            .unwrap();

        if let Some(outcome) = board.outcome() {
            break outcome;
        }

        let mv_j = bot_j.select_move(&board).await;
        moves.push(mv_j);
        board.play(mv_j);

        sender
            .send_async(Message::FinishedMove { id, mv: moves.len() })
            .await
            .unwrap();
    };

    sender.send_async(Message::FinishedRound { id }).await.unwrap();

    Round {
        id,
        start,
        moves,
        outcome,
    }
}

impl RoundId {
    fn is_flipped_pair(left: Self, right: Self) -> bool {
        let RoundId {
            i: li,
            j: lj,
            s: ls,
            global: _,
        } = left;
        let RoundId {
            i: ri,
            j: rj,
            s: rs,
            global: _,
        } = right;

        // intentionally mismatched i/j
        li == rj && lj == ri && ls == rs
    }
}

impl<B: Board> Round<B> {
    fn outcome_i(&self) -> OutcomeWDL {
        self.outcome.pov(self.start.next_player())
    }
}

impl<B: Board> Tournament<B> {
    pub fn bot_count(&self) -> usize {
        self.bot_names.len()
    }
}

impl<B: Board> Display for Tournament<B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Tournament {{")?;
        writeln!(f, "  pos_count: {}", self.pos_count)?;
        writeln!(f, "  self_games: {}", self.self_games)?;
        writeln!(f, "  flip_games: {}", self.flip_games)?;

        writeln!(f, "  results:")?;
        write!(
            f,
            "{}",
            display_build_table(&self.bot_names, &self.grid_wdl, self.grid_wdl_decisive.as_deref())
        )?;

        writeln!(f, "}}")?;

        Ok(())
    }
}

fn display_build_table(
    bot_names: &[String],
    grid_wdl: &[Vec<WDL<usize>>],
    grid_wdl_decisive: Option<&[Vec<WDL<usize>>]>,
) -> Table {
    let mut table = Builder::default();

    table = display_add_rows(table, "ALL", bot_names, grid_wdl);
    if let Some(grid_wdl_decisive) = grid_wdl_decisive {
        table = display_add_rows(table, "DECISIVE", bot_names, grid_wdl_decisive);
    }

    table.build().with(Margin::new(4, 0, 0, 0)).with(Style::modern())
}

fn display_add_rows(mut table: Builder, name: &str, bot_names: &[String], grid_wdl: &[Vec<WDL<usize>>]) -> Builder {
    let bot_count = grid_wdl.len();

    let total_wdl_per_bot = grid_wdl
        .iter()
        .map(|line| line.iter().sum::<WDL<usize>>())
        .collect_vec();
    let total_wdl = total_wdl_per_bot.iter().sum::<WDL<usize>>();

    let total_wdl_str = format!(
        "{}: {}/{}/{}",
        total_wdl.sum(),
        total_wdl.win,
        total_wdl.draw,
        total_wdl.loss,
    );

    let mut header = vec![name.to_owned(), total_wdl_str, "total".to_owned()];
    header.extend_from_slice(bot_names);
    table = table.add_record(header);

    for i in 0..bot_count {
        let bot_wdl = total_wdl_per_bot[i];
        let mut row = vec![
            bot_names[i].clone(),
            format!("{}", bot_wdl.sum()),
            display_format_wdl_elo(bot_wdl),
        ];
        for j in 0..bot_count {
            row.push(display_format_wdl_elo(grid_wdl[i][j]));
        }
        table = table.add_record(row);
    }

    table
}

fn display_format_wdl_elo(wdl: WDL<usize>) -> String {
    let wdl = wdl.cast::<f32>().normalized();

    if wdl.sum().is_nan() {
        "".to_string()
    } else {
        let elo = elo_from_wdl(wdl);
        format!("{:.2}/{:.2}/{:.2} => {:.2}", wdl.win, wdl.draw, wdl.loss, elo)
    }
}
