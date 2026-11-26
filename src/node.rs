use rand::{RngExt, SeedableRng, rngs::StdRng};
use crate::modelstep::ModelStep;
pub struct NeuronLayer{
    nr_in:usize,
    nr_out:usize,
    bias:Vec<f32>,
    weights:Vec<f32>,
}

impl NeuronLayer{
    pub fn new(nr_in:usize, nr_out:usize) -> NeuronLayer{
        let mut rng = StdRng::seed_from_u64(42);
        let bias = vec![0.0;nr_out];
        let mut weights = vec![0.0;nr_out*nr_in];
        let limit = (6.0 / nr_in as f32).sqrt();
        for i in 0..weights.len(){
            weights[i] = rng.random_range(-limit..limit);
        }
        return NeuronLayer {nr_in, nr_out, bias,weights};
    }
    pub fn forward(&self,input:&Vec<f32>)->Vec<f32>{
        let mut output = vec![0.0;self.nr_out];
        for i in 0..self.nr_out{
            output[i] = input.iter().zip(self.weights[i*self.nr_in..(i+1)*self.nr_in].iter()).fold(self.bias[i],
                |s,(inp,w)| s.algebraic_add(inp.algebraic_mul(*w)));
        }
        return output;
    }
    pub fn backprop(&mut self, prev_deriv:&Vec<f32>, last_value:&Vec<f32>,lr:f32,last_layer:bool)->Vec<f32>{
        let mut next_deriv:Vec<f32> = vec![0.0;self.nr_in];
        if !last_layer{ 
            for i in 0..self.nr_out{
                if prev_deriv[i] == 0.0{
                    continue;
                }
                for j in 0..self.nr_in{
                    next_deriv[j] = next_deriv[j].algebraic_add(self.weights[i*self.nr_in + j].algebraic_mul(prev_deriv[i]));
                }
            }
        }
        for i in 0..self.nr_out{
            if prev_deriv[i] == 0.0{
                continue;
            }
            let s = prev_deriv[i]*lr;
            for (w,v) in self.weights[i*self.nr_in..(i+1)*self.nr_in].iter_mut().zip(last_value.iter()){
                *w -= v*s;
            }
        }
        for i in 0..self.nr_out{
            self.bias[i] -= prev_deriv[i]*lr;
        }
        return next_deriv;
    }
}

impl ModelStep for NeuronLayer{
    fn forward(&self, input:&Vec<f32>)->Vec<f32>{
        return self.forward(input);
    }
    fn backprop(&mut self, prev_deriv:&Vec<f32>, last_value:&Vec<f32>,lr:f32,last_layer:bool)->Vec<f32>{
        return self.backprop(prev_deriv, last_value, lr,last_layer);
    }
}