use rand::{Error, RngCore};

/// Like `Option::unwrap` but for arbitrary patterns.
/// ```
/// assert_eq!(5, unwrap_match!(Some(5), Some(x) => x));
/// ```
macro_rules! unwrap_match {
    ($value: expr, $($pattern: pat)|+ => $result: expr) => {
        match $value {
            $($pattern)|+ =>
                $result,
            ref value =>
                panic!("unwrap_match failed: `{:?}` does not match `{}`", value, stringify!($($pattern)|+)),
        }
    };
}


/// An Rng implementation that panics as soon as it is called.
/// Useful to assert that something doesn't actually use any randomness.
#[derive(Debug)]
pub struct PanicRng;

impl RngCore for PanicRng {
    fn next_u32(&mut self) -> u32 {
        panic!("Tried to use PanicRng")
    }

    fn next_u64(&mut self) -> u64 {
        panic!("Tried to use PanicRng")
    }

    fn fill_bytes(&mut self, _: &mut [u8]) {
        panic!("Tried to use PanicRng")
    }

    fn try_fill_bytes(&mut self, _: &mut [u8]) -> Result<(), Error> {
        panic!("Tried to use PanicRng")
    }
}