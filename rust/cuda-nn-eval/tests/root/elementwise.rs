use crate::root::runner::{test_elementwise, test_elementwise_pair};

#[test]
fn add() {
    test_elementwise_pair(|a, b| a + b, |g, a, b| g.add(a, b))
}

#[test]
fn sub() {
    test_elementwise_pair(|a, b| a - b, |g, a, b| g.sub(a, b))
}

#[test]
fn mul() {
    test_elementwise_pair(|a, b| a * b, |g, a, b| g.mul(a, b))
}

#[test]
fn clamp() {
    for min in [f32::NEG_INFINITY, 0.0] {
        for max in [f32::INFINITY, 6.0] {
            println!("Testing clamp({}, {})", min, max);
            test_elementwise(|a| a.clamp(min, max), |g, a| g.clamp(a, min, max))
        }
    }
}
