

mod node;
use node::NeuronLayer;
mod model;
mod loss;
mod activation;
use activation::ReLu;
mod modelstep;
fn main() {
    let mut model = model::Model::new(vec![
        Box::new(NeuronLayer::new(2, 4)), 
        Box::new(ReLu::new()),
        Box::new(NeuronLayer::new(4,1))
    ]);
    let nr_epoch = 500;
    //let and_data: [(Vec<f64>, Vec<f64>); 4] = [(vec![0.0,0.0],vec![0.0]),(vec![1.0,0.0],vec![0.0]),(vec![0.0,1.0],vec![0.0]),(vec![1.0,1.0],vec![1.0])];
    let xor_data: [(Vec<f64>, Vec<f64>); 4] = [(vec![0.0,0.0],vec![0.0]),(vec![1.0,0.0],vec![1.0]),(vec![0.0,1.0],vec![1.0]),(vec![1.0,1.0],vec![0.0])];
    for epoch in 0..nr_epoch{
        let mut tot_loss = 0.0;
        for (x,y) in xor_data.iter(){
            let pred = model.forward(x.clone());
            let (loss_sum, loss) = loss::MSEloss::get_loss(&pred,&y);
            model.backward(loss, 0.01);
            tot_loss += loss_sum;
        }
        println!("Epoch: {epoch}, loss={tot_loss}");
    }
    for(x,_) in xor_data.iter(){
        println!("{:?}",x);
        println!("{:?}", model.forward(x.clone()));
    }
}
