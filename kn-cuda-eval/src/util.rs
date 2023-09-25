use std::fmt::{Debug, Formatter};

pub fn debug_vec_multiline<'a, T: Debug>(prefix: &'a str, values: &'a [T]) -> impl Debug + 'a {
    struct Wrapper<'a, T>(&'a str, &'a [T]);
    impl<'a, T: Debug> Debug for Wrapper<'a, T> {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            writeln!(f, "[")?;
            for v in self.1 {
                writeln!(f, "{}{}{:?},", self.0, self.0, v)?;
            }
            write!(f, "{}]", self.0)?;
            Ok(())
        }
    }
    Wrapper(prefix, values)
}
