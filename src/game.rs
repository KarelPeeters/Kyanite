use rand::Rng;

pub trait Game {
    type P: Eq + Copy;
    type M: Eq + Copy;
    type S: State<Self::P, Self::M>;

    fn initial_state() -> Self::S;
}

pub trait State<P, M>: Clone + Eq {
    type MoveIter: Iterator<Item=M>;

    fn new() -> Self;

    fn is_done(&self);
    fn winner(&self) -> Option<P>;
    fn next_player(&self) -> Option<P>;

    fn available_moves(&self) -> Self::MoveIter;
    fn random_available_move<R: Rng>(&self, rand: &mut R) -> Option<M>;

    fn play(&mut self, mv: M);
}
