mod library {
    pub struct Foo(i32);

    impl Foo {
        pub fn new(x: i32) -> Foo {
            Foo(x)
        }
    }

    pub struct Bar<'a>(&'a Foo);

    impl Bar<'_> {
        pub fn new(foo: &Foo) -> Bar {
            Bar(foo)
        }
    }
}

#[macro_use]
extern crate rental;

use library::{Foo, Bar};
use rental::rental;

rental! {
    mod my_rentals {
        use super::*;

        #[rental]
        pub struct Container {
            foo: Box<Foo>,
            bar: Bar<'foo>,
        }
    }
}

pub use my_rentals::Container;

pub fn build() -> Container {
    let foo = Foo::new(5);

    my_rentals::Container::new(
        Box::new(foo),
        |foo| Bar::new(foo),
    )
}


fn main() {}