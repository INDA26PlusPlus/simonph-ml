use super::modelstep::ModelStep;
pub struct Model{
    pub layers:Vec<Box<dyn ModelStep>>,
    pub last_output:Vec<Vec<f32>>
}

impl Model{
    pub fn new(layers:Vec<Box<dyn ModelStep>>)->Model{
        return Model{layers, last_output:Vec::new()};
    }
    pub fn forward(&mut self, input:Vec<f32>) -> Vec<f32>{
        self.last_output.clear();
        self.last_output.push(input);
        for layer in self.layers.iter(){
            self.last_output.push(layer.forward(self.last_output.last().unwrap()));
        }
        return self.last_output.last().unwrap().clone();
    }
    pub fn backward(&mut self, mut loss:Vec<f32>,lr:f32){
        for i in (0..self.layers.len()).rev(){
            loss = self.layers[i].backprop(&loss, &self.last_output[i], lr,i==0)
        }
    }
}