pub mod move_selector;
pub mod simulation;

pub mod protocol;
pub mod server;

pub mod commander;
pub mod collector;
pub mod generator;

// TODO fix startup behaviour, only release games in the order they were started to ensure a constant
//  distribution of game lengths

// TODO re-implement caching if it turns out to actually be worth it