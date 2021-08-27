#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Command<'a> {
    UAI,
    IsReady,
    NewGame,
    Quit,
    Position(Position<'a>),
    Go(GoTimeSettings),
    SetOption { name: &'a str, value: &'a str },
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum GoTimeSettings {
    Move(u32),
    Clock {
        b_time: u32,
        w_time: u32,
        b_inc: u32,
        w_inc: u32,
    },
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Position<'a> {
    StartPos,
    Fen(&'a str),
}

impl<'a> Command<'a> {
    pub fn parse(input: &'a str) -> Result<Command, nom::Err<nom::error::Error<&str>>> {
        parse::command(input).map(|(left, command)| {
            assert!(left.is_empty());
            command
        })
    }
}

mod parse {
    use nom::branch::alt;
    use nom::bytes::complete::{tag, take_until, take_while};
    use nom::character::complete::digit1;
    use nom::combinator::{eof, map, value};
    use nom::IResult;
    use nom::sequence::{preceded, terminated, tuple};

    use crate::uai::command::{Command, GoTimeSettings, Position};

    pub fn command(input: &str) -> IResult<&str, Command> {
        let int = || map(digit1, |s: &str| s.parse().unwrap());

        let move_time = preceded(
            tag("movetime "),
            map(int(), |n| GoTimeSettings::Move(n)),
        );

        let clock_time = map(
            tuple((
                tag("btime "), int(),
                tag(" wtime "), int(),
                tag(" binc "), int(),
                tag(" winc "), int(),
            )),
            |(_, b_time, _, w_time, _, b_inc, _, w_inc)| {
                GoTimeSettings::Clock { b_time, w_time, b_inc, w_inc }
            },
        );

        let go = preceded(
            tag("go "),
            map(alt((move_time, clock_time)), Command::Go),
        );

        let position = preceded(
            tag("position "),
            map(
                alt((
                    value(Position::StartPos, tag("startpos")),
                    preceded(tag("fen "), map(take_while(|_| true), Position::Fen)),
                )),
                Command::Position,
            ),
        );

        let set_option = preceded(
            tag("setoption "),
            map(
                tuple((
                    tag("name "),
                    take_until(" "),
                    tag(" value "),
                    take_while(|_| true)
                )),
                |(_, name, _, value)| {
                    Command::SetOption { name, value }
                },
            ),
        );

        let main = alt((
            value(Command::NewGame, tag("uainewgame")),
            value(Command::UAI, tag("uai")),
            value(Command::IsReady, tag("isready")),
            value(Command::Quit, tag("quit")),
            position,
            go,
            set_option,
        ));

        let mut complete = terminated(main, eof);

        complete(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basics() {
        assert_eq!(Command::UAI, Command::parse("uai").unwrap());
        assert_eq!(Command::IsReady, Command::parse("isready").unwrap());
        assert_eq!(Command::NewGame, Command::parse("uainewgame").unwrap());
        assert_eq!(Command::Quit, Command::parse("quit").unwrap());
    }
}