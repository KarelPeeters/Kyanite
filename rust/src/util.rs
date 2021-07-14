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
