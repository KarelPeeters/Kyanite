#![warn(missing_debug_implementations)]

#![allow(clippy::len_without_is_empty)]
#![allow(clippy::let_and_return)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::upper_case_acronyms)]

pub mod zero;

pub mod network;

pub mod mapping;
pub mod oracle;

pub mod convert;
pub mod selfplay;

pub mod stats;

#[macro_use]
pub mod util;
