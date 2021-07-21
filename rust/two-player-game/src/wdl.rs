use cast_trait::Cast;

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

pub trait POV {
    type Output;
    fn pov(self, pov: Player) -> Self::Output;
}

pub trait Flip {
    fn flip(self) -> Self;
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

    /// Combine outcomes together in minimax-style.. `None` means unknown.
    pub fn best(children: impl IntoIterator<Item=Option<OutcomeWDL>>) -> Option<OutcomeWDL> {
        let mut any_unknown = false;
        let mut all_known_are_loss = true;

        for outcome in children {
            match outcome {
                None => {
                    any_unknown = true;
                }
                Some(OutcomeWDL::Win) => {
                    //early exit, we've found a win
                    return Some(OutcomeWDL::Win);
                }
                Some(OutcomeWDL::Draw) => {
                    all_known_are_loss = false;
                }
                Some(OutcomeWDL::Loss) => {}
            }
        }

        if any_unknown {
            None
        } else if all_known_are_loss {
            Some(OutcomeWDL::Loss)
        } else {
            Some(OutcomeWDL::Draw)
        }
    }
}

impl<V: num::Float> WDL<V> {
    pub fn nan() -> WDL<V> {
        WDL { win: V::nan(), draw: V::nan(), loss: V::nan() }
    }
}

impl<V: Copy> WDL<V> {
    pub fn cast<W>(self) -> WDL<W> where V: Cast<W> {
        WDL {
            win: self.win.cast(),
            draw: self.draw.cast(),
            loss: self.loss.cast(),
        }
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

impl<I: POV> POV for Option<I> {
    type Output = Option<I::Output>;
    fn pov(self, pov: Player) -> Option<I::Output> {
        self.map(|inner| inner.pov(pov))
    }
}

impl<I: Flip> Flip for Option<I> {
    fn flip(self) -> Self {
        self.map(|inner| inner.flip())
    }
}

impl POV for Outcome {
    type Output = OutcomeWDL;
    fn pov(self, pov: Player) -> OutcomeWDL {
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

impl Flip for OutcomeWDL {
    fn flip(self) -> Self {
        match self {
            OutcomeWDL::Win => OutcomeWDL::Loss,
            OutcomeWDL::Draw => OutcomeWDL::Draw,
            OutcomeWDL::Loss => OutcomeWDL::Win,
        }
    }
}

impl<V: Copy> Flip for WDL<V> {
    fn flip(self) -> Self {
        WDL { win: self.loss, draw: self.draw, loss: self.win }
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
