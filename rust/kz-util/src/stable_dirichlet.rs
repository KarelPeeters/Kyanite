use rand::distributions::Distribution;
use rand::Rng;
use rand_distr::Gamma;

/// Variant of [rand::distributions::Dirichlet] that never generates NaNs, even when `alpha` is low
#[derive(Debug, Copy, Clone)]
pub struct StableDirichlet {
    alpha: f32,
    len: usize,
}

#[derive(Debug, Copy, Clone)]
pub struct DirichletError;

impl StableDirichlet {
    pub fn new(alpha: f32, len: usize) -> Result<Self, DirichletError> {
        if alpha > 0.0 && len > 0 {
            Ok(Self { alpha, len })
        } else {
            Err(DirichletError)
        }
    }
}

const ALPHA_MIN: f32 = 0.1;
const SUM_MIN: f32 = 0.00000001;

impl Distribution<Vec<f32>> for StableDirichlet {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<f32> {
        assert!(self.len > 0);
        if self.len == 1 {
            return vec![1.0; self.len];
        }

        if self.alpha > ALPHA_MIN {
            // actually try generating a proper sample
            let gamma = Gamma::new(self.alpha, 1.0).unwrap();

            let mut values = Vec::with_capacity(self.len);
            let mut sum = 0.0;

            for _ in 0..self.len {
                let v = gamma.sample(rng);
                values.push(v);
                sum += v;
            }

            if sum > SUM_MIN {
                // normalize values
                for v in &mut values {
                    *v /= sum;
                }
                return values;
            }
        }

        // fallback: generate a maximally concentrated sample
        let index = rng.gen_range(0..self.len);
        let mut result = vec![0.0; self.len];
        result[index] = 1.0;
        result
    }
}
