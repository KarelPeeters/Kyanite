use crate::board::{Outcome, Player};

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum OutcomeWDL {
    Win,
    Draw,
    Loss,
}

#[derive(Default, Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct WDL<V> {
    pub win: V,
    pub draw: V,
    pub loss: V,
}

impl Outcome {
    pub fn pov(self, pov: Player) -> OutcomeWDL {
        match self {
            Outcome::WonBy(player) => if player == pov {
                OutcomeWDL::Win
            } else {
                OutcomeWDL::Loss
            },
            Outcome::Draw => OutcomeWDL::Draw,
        }
    }
}

impl OutcomeWDL {
    pub fn to_wdl<V: num::One + num::Zero>(self) -> WDL<V> {
        match self {
            OutcomeWDL::Win => WDL { win: V::one(), draw: V::zero(), loss: V::zero() },
            OutcomeWDL::Draw => WDL { win: V::zero(), draw: V::one(), loss: V::zero() },
            OutcomeWDL::Loss => WDL { win: V::zero(), draw: V::zero(), loss: V::one() },
        }
    }

    pub fn sign<V: num::Zero + num::One + std::ops::Neg<Output=V>>(self) -> V {
        match self {
            OutcomeWDL::Win => V::one(),
            OutcomeWDL::Draw => V::zero(),
            OutcomeWDL::Loss => -V::one(),
        }
    }
}

impl<V: num::Float> WDL<V> {
    pub fn nan() -> WDL<V> {
        WDL { win: V::nan(), draw: V::nan(), loss: V::nan() }
    }
}

impl<V: Copy> WDL<V> {
    pub fn other(self) -> Self {
        WDL { win: self.loss, draw: self.draw, loss: self.win }
    }
}

impl<V: Copy + std::ops::Sub<V, Output=V>> WDL<V> {
    pub fn value(self) -> V {
        self.win - self.loss
    }
}

impl<V: Copy + std::ops::Add<V, Output=V>> WDL<V> {
    pub fn sum(self) -> V {
        self.win + self.draw + self.loss
    }
}

impl<V: Copy + std::ops::Add<V, Output=V>> std::ops::Add<WDL<V>> for WDL<V> {
    type Output = WDL<V>;

    fn add(self, rhs: WDL<V>) -> Self::Output {
        WDL {
            win: self.win + rhs.win,
            draw: self.draw + rhs.draw,
            loss: self.loss + rhs.loss,
        }
    }
}

impl<V: Copy + std::ops::Add<V, Output=V>> std::ops::AddAssign<WDL<V>> for WDL<V> {
    fn add_assign(&mut self, rhs: WDL<V>) {
        *self = *self + rhs;
    }
}

impl<V: Copy + std::ops::Div<V, Output=V>> std::ops::Div<V> for WDL<V> {
    type Output = WDL<V>;

    fn div(self, rhs: V) -> Self::Output {
        WDL { win: self.win / rhs, draw: self.draw / rhs, loss: self.loss / rhs }
    }
}
