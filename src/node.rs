use rand::{RngExt, SeedableRng, rngs::StdRng};

use crate::modelstep::ModelStep;
pub struct NeuronLayer{
    nr_in:usize,
    nr_out:usize,
    bias:Vec<f64>,
    weights:Vec<Vec<f64>>,
}

impl NeuronLayer{
    pub fn new(nr_in:usize, nr_out:usize) -> NeuronLayer{
        let mut rng = StdRng::seed_from_u64(42);
        let bias = vec![0.0;nr_out];
        let mut weights = vec![vec![0.0;nr_in];nr_out];
        let limit = (6.0 / nr_in as f64).sqrt();
        for i in 0..weights.len(){
            for j in 0..weights[0].len(){
                weights[i][j] = rng.random_range(-limit..limit);
            }
        }
        return NeuronLayer {nr_in, nr_out, bias,weights};
    }
    pub fn forward(&self,input:&Vec<f64>)->Vec<f64>{
        let mut output = vec![0.0;self.nr_out];
        for i in 0..self.nr_out{
            output[i] = self.bias[i];
            for j in 0..self.nr_in{
                output[i] += self.weights[i][j]*input[j];
            }
        }
        return output;
    }
    pub fn backprop(&mut self, prev_deriv:&Vec<f64>, last_value:&Vec<f64>,lr:f64)->Vec<f64>{
        let mut next_deriv = vec![0.0;self.nr_in];
        for i in 0..self.nr_in{
            for j in 0..self.nr_out{
                next_deriv[i] += self.weights[j][i]*prev_deriv[j];
            }
        }
        for i in 0..self.nr_out{
            for j in 0..self.nr_in{
                self.weights[i][j] -= last_value[j]*prev_deriv[i]*lr;
            }
        }
        for i in 0..self.nr_out{
            self.bias[i] -= prev_deriv[i]*lr;
        }
        return next_deriv;
    }
}

impl ModelStep for NeuronLayer{
    fn forward(&self, input:&Vec<f64>)->Vec<f64>{
        return self.forward(input);
    }
    fn backprop(&mut self, prev_deriv:&Vec<f64>, last_value:&Vec<f64>,lr:f64)->Vec<f64>{
        return self.backprop(prev_deriv, last_value, lr);
    }
}