use sttt::board::{Outcome, Player};

/// A board evaluation, either as returned by the network or as the final output of a zero tree search.
#[derive(Debug)]
pub struct ZeroEvaluation {
    /// The win, draw and loss probabilities, after normalization.
    pub wdl: WDL,

    /// The policy "vector", only containing the available moves in the order they are yielded by `available_moves`.
    pub policy: Vec<f32>,
}

#[derive(Default, Debug, Copy, Clone)]
pub struct WDL {
    pub win: f32,
    pub draw: f32,
    pub loss: f32,
}

impl WDL {
    pub fn nan() -> WDL {
        WDL { win: f32::NAN, draw: f32::NAN, loss: f32::NAN }
    }

    pub fn from_outcome(outcome: Outcome, pov: Player) -> WDL {
        WDL {
            win: (outcome == Outcome::WonBy(pov)) as u8 as f32,
            draw: (outcome == Outcome::Draw) as u8 as f32,
            loss: (outcome == Outcome::WonBy(pov.other())) as u8 as f32,
        }
    }

    pub fn value(self) -> f32 {
        self.win - self.loss
    }
}

impl std::ops::Neg for WDL {
    type Output = WDL;

    fn neg(self) -> WDL {
        WDL { win: self.loss, draw: self.draw, loss: self.win }
    }
}

impl std::ops::Add<WDL> for WDL {
    type Output = WDL;

    fn add(self, rhs: WDL) -> WDL {
        WDL {
            win: self.win + rhs.win,
            draw: self.draw + rhs.draw,
            loss: self.loss + rhs.loss,
        }
    }
}

impl std::ops::Div<f32> for WDL {
    type Output = WDL;

    fn div(self, rhs: f32) -> WDL {
        WDL {
            win: self.win / rhs,
            draw: self.draw / rhs,
            loss: self.loss / rhs,
        }
    }
}

impl std::ops::AddAssign<WDL> for WDL {
    fn add_assign(&mut self, rhs: WDL) {
        *self = *self + rhs
    }
}