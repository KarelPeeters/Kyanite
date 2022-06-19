use board_game::board::{Outcome, Player};
use board_game::pov::{NonPov, Pov, ScalarAbs, ScalarPov};
use board_game::wdl::{OutcomeWDL, WDLAbs, WDL};
use std::fmt::{Display, Formatter};

#[derive(Debug, Copy, Clone, Default)]
pub struct ZeroValuesAbs {
    pub value_abs: ScalarAbs<f32>,
    pub wdl_abs: WDLAbs<f32>,
    pub moves_left: f32,
}

#[derive(Debug, Copy, Clone, Default)]
pub struct ZeroValuesPov {
    pub value: ScalarPov<f32>,
    pub wdl: WDL<f32>,
    pub moves_left: f32,
}

impl NonPov for ZeroValuesAbs {
    type Output = ZeroValuesPov;

    fn pov(self, pov: Player) -> Self::Output {
        ZeroValuesPov {
            value: self.value_abs.pov(pov),
            wdl: self.wdl_abs.pov(pov),
            moves_left: self.moves_left,
        }
    }
}

impl Pov for ZeroValuesPov {
    type Output = ZeroValuesAbs;

    fn un_pov(self, pov: Player) -> Self::Output {
        ZeroValuesAbs {
            value_abs: self.value.un_pov(pov),
            wdl_abs: self.wdl.un_pov(pov),
            moves_left: self.moves_left,
        }
    }
}

impl ZeroValuesAbs {
    pub fn from_outcome(outcome: Outcome, moves_left: f32) -> Self {
        ZeroValuesAbs {
            value_abs: outcome.sign(),
            wdl_abs: outcome.to_wdl_abs(),
            moves_left,
        }
    }

    pub fn nan() -> Self {
        ZeroValuesAbs {
            value_abs: ScalarAbs::new(f32::NAN),
            wdl_abs: WDLAbs::nan(),
            moves_left: f32::NAN,
        }
    }

    /// The value that should be accumulated in the parent node of this value.
    pub fn parent(&self) -> Self {
        ZeroValuesAbs {
            value_abs: self.value_abs,
            wdl_abs: self.wdl_abs,
            moves_left: self.moves_left + 1.0,
        }
    }
}

impl ZeroValuesPov {
    pub fn from_outcome(outcome: OutcomeWDL, moves_left: f32) -> Self {
        ZeroValuesPov {
            value: ScalarPov::new(outcome.sign()),
            wdl: outcome.to_wdl(),
            moves_left,
        }
    }

    pub fn nan() -> Self {
        ZeroValuesPov {
            value: ScalarPov::new(f32::NAN),
            wdl: WDL::nan(),
            moves_left: f32::NAN,
        }
    }

    pub fn parent_flip(self) -> ZeroValuesPov {
        ZeroValuesPov {
            value: self.value.flip(),
            wdl: self.wdl.flip(),
            moves_left: self.moves_left + 1.0,
        }
    }

    pub fn to_slice(self) -> [f32; 5] {
        [
            self.value.value,
            self.wdl.win,
            self.wdl.draw,
            self.wdl.loss,
            self.moves_left,
        ]
    }
}

impl std::ops::Add<Self> for ZeroValuesAbs {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        ZeroValuesAbs {
            value_abs: self.value_abs + rhs.value_abs,
            wdl_abs: self.wdl_abs + rhs.wdl_abs,
            moves_left: self.moves_left + rhs.moves_left,
        }
    }
}

impl std::ops::Add<Self> for ZeroValuesPov {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        ZeroValuesPov {
            value: self.value + rhs.value,
            wdl: self.wdl + rhs.wdl,
            moves_left: self.moves_left + rhs.moves_left,
        }
    }
}

impl std::ops::AddAssign<Self> for ZeroValuesAbs {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs
    }
}

impl std::ops::AddAssign<Self> for ZeroValuesPov {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs
    }
}

impl std::ops::Div<f32> for ZeroValuesAbs {
    type Output = ZeroValuesAbs;

    fn div(self, rhs: f32) -> Self::Output {
        ZeroValuesAbs {
            value_abs: self.value_abs / rhs,
            wdl_abs: self.wdl_abs / rhs,
            moves_left: self.moves_left / rhs,
        }
    }
}

impl std::ops::Div<f32> for ZeroValuesPov {
    type Output = ZeroValuesPov;

    fn div(self, rhs: f32) -> Self::Output {
        ZeroValuesPov {
            value: self.value / rhs,
            wdl: self.wdl / rhs,
            moves_left: self.moves_left / rhs,
        }
    }
}

impl ZeroValuesAbs {
    pub const FORMAT_SUMMARY: &'static str = "v a/d/b m";
}

impl Display for ZeroValuesAbs {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:.3}, {:.3}/{:.3}/{:.3}, {:.3}",
            self.value_abs.value_a, self.wdl_abs.win_a, self.wdl_abs.draw, self.wdl_abs.win_b, self.moves_left
        )
    }
}

impl ZeroValuesPov {
    pub const FORMAT_SUMMARY: &'static str = "v w/d/l m";
}

impl Display for ZeroValuesPov {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:.3}, {:.3}/{:.3}/{:.3}, {:.3}",
            self.value.value, self.wdl.win, self.wdl.draw, self.wdl.loss, self.moves_left
        )
    }
}
