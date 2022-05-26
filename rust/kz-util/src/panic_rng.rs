use rand::{Error, RngCore};

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
