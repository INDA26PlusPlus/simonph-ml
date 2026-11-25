use crate::node::NeuronLayer;

mod node;
mod model;
mod loss;
fn main() {
    let mut model = model::Model::new(vec![NeuronLayer::new(2, 1)]);
    let nr_epoch = 20;
    let data: [(Vec<f64>, Vec<f64>); 4] = [(vec![0.0,0.0],vec![0.0]),(vec![1.0,0.0],vec![0.0]),(vec![0.0,1.0],vec![0.0]),(vec![1.0,1.0],vec![1.0])];
    for epoch in 0..nr_epoch{
        let mut tot_loss = 0.0;
        for (x,y) in data.iter(){
            let pred = model.forward(x.clone());
            let (loss_sum, loss) = loss::MSEloss::get_loss(&pred,&y);
            model.backward(loss, 0.01);
            tot_loss += loss_sum;
        }
        println!("Epoch: {epoch}, loss={tot_loss}");
    }
    for(x,_) in data.iter(){
        println!("{:?}",x);
        println!("{:?}", model.forward(x.clone()));
    }
}
