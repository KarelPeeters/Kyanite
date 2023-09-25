/// Trick to avoid stackoverflow on deep graphs:
/// We don't recuse at all, and if we would have wanted to to recurse we return `Err` instead.
/// This function then explicitly recuses using a heap-allocation stack by repeatedly calling `f`.
///
/// `f` _must_ be deterministic and have built-in caching.
pub fn heap_recurse<X: Copy, Y: Clone>(x: X, mut f_cached: impl FnMut(X) -> Result<Y, X>) -> Y {
    let mut stack = vec![x];

    loop {
        let curr = *stack.last().unwrap();

        match f_cached(curr) {
            Ok(y) => {
                stack.pop().unwrap();
                if stack.is_empty() {
                    return y;
                }
            }
            Err(other_value) => stack.push(other_value),
        }
    }
}
