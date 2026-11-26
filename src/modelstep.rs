pub trait ModelStep{
    fn forward(&self, input:&Vec<f64>)->Vec<f64>;
    fn backprop(&mut self, prev_deriv:&Vec<f64>, last_value:&Vec<f64>,lr:f64,last_layer:bool)->Vec<f64>;
}
