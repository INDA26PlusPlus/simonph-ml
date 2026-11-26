pub trait ModelStep{
    fn forward(&self, input:&Vec<f32>)->Vec<f32>;
    fn backprop(&mut self, prev_deriv:&Vec<f32>, last_value:&Vec<f32>,lr:f32,last_layer:bool)->Vec<f32>;
}
