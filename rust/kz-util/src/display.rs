use std::fmt::{Display, Formatter};

pub fn display_option<T: Display>(value: Option<T>) -> impl Display {
    struct Wrapper<T>(Option<T>);
    impl<T: Display> Display for Wrapper<T> {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            match &self.0 {
                None => write!(f, "None"),
                Some(value) => write!(f, "Some({})", value),
            }
        }
    }
    Wrapper(value)
}

pub fn display_option_empty<T: Display>(value: Option<T>) -> impl Display {
    struct Wrapper<T>(Option<T>);
    impl<T: Display> Display for Wrapper<T> {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            match &self.0 {
                None => write!(f, ""),
                Some(value) => write!(f, "{}", value),
            }
        }
    }
    Wrapper(value)
}
