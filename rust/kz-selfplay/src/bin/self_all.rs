use board_game::board::Board;
use board_game::games::ttt::TTTBoard;
use internal_iterator::InternalIterator;
use kz_core::mapping::ttt::TTTStdMapper;
use kz_core::mapping::BoardMapper;
use kz_core::network::dummy::{uniform_policy, uniform_values};
use kz_core::network::ZeroEvaluation;
use kz_selfplay::binary_output::BinaryOutput;
use kz_selfplay::simulation::{Position, Simulation};
use std::borrow::Cow;

fn main() -> std::io::Result<()> {
    let mapper = TTTStdMapper;
    let board = TTTBoard::default();

    let mut output = BinaryOutput::new(r#"C:\Documents\Programming\STTT\AlphaZero\data\all\ttt"#, "ttt", mapper)?;

    visit(&board, &mut vec![], &mut output)?;

    output.finish()?;

    Ok(())
}

fn visit<B: Board, M: BoardMapper<B>>(
    board: &B,
    positions: &mut Vec<Position<B>>,
    output: &mut BinaryOutput<B, M>,
) -> std::io::Result<()> {
    if board.is_done() {
        let simulation = Simulation {
            positions: std::mem::take(positions),
            final_board: board.clone(),
        };
        output.append(&simulation)?;
        println!("Appended game {}", output.game_count());
        *positions = simulation.positions;
        return Ok(());
    }

    let eval = ZeroEvaluation {
        values: uniform_values(),
        policy: Cow::Owned(uniform_policy(board.available_moves().count())),
    };

    board.available_moves().for_each(|mv: B::Move| {
        let position = Position {
            board: board.clone(),
            is_full_search: true,
            played_mv: mv,
            zero_visits: 0,
            zero_evaluation: eval.clone(),
            net_evaluation: eval.clone(),
        };

        positions.push(position);
        let next = board.clone_and_play(mv);
        visit(&next, positions, output).unwrap();
        positions.pop().unwrap();
    });

    Ok(())
}
