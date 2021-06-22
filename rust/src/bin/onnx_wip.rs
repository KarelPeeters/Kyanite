#[macro_use]
extern crate rental;

use rental::rental;

use library::{Bar, Foo};
pub use my_rentals::Container;

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

pub fn build() -> Container {
    let foo = Foo::new(5);

    my_rentals::Container::new(
        Box::new(foo),
        |foo| Bar::new(foo),
    )
}


fn main() {}

/*
fn main() -> onnxruntime::Result<()> {
    let env = Environment::builder()
        .with_name("test_env")
        .with_log_level(LoggingLevel::Verbose)
        .build()?;

    let mut session = env.new_session_builder()?
        // .with_optimization_level(GraphOptimizationLevel::All)?
        // .with_number_threads(1)?
        .with_model_from_file("../data/onnx/small.onnx")?;

    let batch_size = 1000;
    let start = Instant::now();
    let mut total = 0;

    loop {
        let input = Array::from_elem((100, 5, 9, 9), 0.0f32);
        let input = vec![input];
        let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(input)?;

        total += batch_size;
        let delta = (Instant::now() - start).as_secs_f32();
        let throughput = (total as f32) / delta;
        println!("Throughput: {:.2} boards/s", throughput);
    }

    // Ok(())
}
 */