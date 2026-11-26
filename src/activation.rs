use crate::modelstep::ModelStep;

pub struct ReLu{}

impl ReLu{
    pub fn new() -> ReLu{
        return ReLu {};
    }
    pub fn forward(&self, input:&Vec<f32>)->Vec<f32>{
        let mut ret = input.clone();
        for i in 0..ret.len(){
            if ret[i] < 0.0{
                ret[i] = 0.0;
            }
        }
        return ret;
    }
    pub fn backprop(&mut self, prev_deriv:&Vec<f32>, last_value:&Vec<f32>,_:f32)->Vec<f32>{
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
    fn forward(&self, input:&Vec<f32>)->Vec<f32>{
        return self.forward(input);
    }
    fn backprop(&mut self, prev_deriv:&Vec<f32>, last_value:&Vec<f32>,lr:f32,_:bool)->Vec<f32>{
        return self.backprop(prev_deriv, last_value, lr);
    }
}