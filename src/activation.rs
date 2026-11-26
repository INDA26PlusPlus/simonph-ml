use crate::modelstep::ModelStep;

pub struct ReLu{}

impl ReLu{
    pub fn new() -> ReLu{
        return ReLu {};
    }
    pub fn forward(&self, input:&Vec<f64>)->Vec<f64>{
        let mut ret = input.clone();
        for i in 0..ret.len(){
            if ret[i] < 0.0{
                ret[i] = 0.0;
            }
        }
        return ret;
    }
    pub fn backprop(&mut self, prev_deriv:&Vec<f64>, last_value:&Vec<f64>,_:f64)->Vec<f64>{
        let mut ret = prev_deriv.clone();
        for i in 0..ret.len(){
            if last_value[i] < 0.0{
                ret[i] = 0.0;
            }
            else{
                ret[i] = prev_deriv[i];
            }
        }
        return ret;
    }
}

impl ModelStep for ReLu{
    fn forward(&self, input:&Vec<f64>)->Vec<f64>{
        return self.forward(input);
    }
    fn backprop(&mut self, prev_deriv:&Vec<f64>, last_value:&Vec<f64>,lr:f64,_:bool)->Vec<f64>{
        return self.backprop(prev_deriv, last_value, lr);
    }
}