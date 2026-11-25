

mod node;
use node::NeuronLayer;
mod model;
mod loss;
mod activation;
use activation::ReLu;
mod modelstep;
use mnist::*;
// fn main() {
//     let mut model = model::Model::new(vec![
//         Box::new(NeuronLayer::new(2, 4)), 
//         Box::new(ReLu::new()),
//         Box::new(NeuronLayer::new(4,1))
//     ]);
//     let nr_epoch = 500;
//     //let and_data: [(Vec<f64>, Vec<f64>); 4] = [(vec![0.0,0.0],vec![0.0]),(vec![1.0,0.0],vec![0.0]),(vec![0.0,1.0],vec![0.0]),(vec![1.0,1.0],vec![1.0])];
//     let xor_data: [(Vec<f64>, Vec<f64>); 4] = [(vec![0.0,0.0],vec![0.0]),(vec![1.0,0.0],vec![1.0]),(vec![0.0,1.0],vec![1.0]),(vec![1.0,1.0],vec![0.0])];
//     for epoch in 0..nr_epoch{
//         let mut tot_loss = 0.0;
//         for (x,y) in xor_data.iter(){
//             let pred = model.forward(x.clone());
//             let (loss_sum, loss) = loss::MSEloss::get_loss(&pred,&y);
//             model.backward(loss, 0.01);
//             tot_loss += loss_sum;
//         }
//         println!("Epoch: {epoch}, loss={tot_loss}");
//     }
//     for(x,_) in xor_data.iter(){
//         println!("{:?}",x);
//         println!("{:?}", model.forward(x.clone()));
//     }
// }
fn main(){
    let Mnist {
       trn_img,
       trn_lbl,
       tst_img,
       tst_lbl,
       ..
    } = MnistBuilder::new()
        .base_path("mnist")
       .label_format_one_hot()
       .training_set_length(50000)
       .validation_set_length(1_000)
       .test_set_length(10000)
       .finalize();
    let train_data:Vec<Vec<f64>> = trn_img.chunks(28*28).map(|img| img.iter().map(|x| *x as f64/255.0).collect()).collect();
    let train_labels:Vec<Vec<f64>> = trn_lbl.chunks(10).map(|img| img.iter().map(|x| *x as f64).collect()).collect();

    let test_data:Vec<Vec<f64>> = tst_img.chunks(28*28).map(|img| img.iter().map(|x| *x as f64/255.0).collect()).collect();
    let test_labels:Vec<Vec<f64>> = tst_lbl.chunks(10).map(|img| img.iter().map(|x| *x as f64).collect()).collect(); 

    let mut model = model::Model::new(vec![
        Box::new(NeuronLayer::new(28*28, 128)), 
        Box::new(ReLu::new()),
        Box::new(NeuronLayer::new(128,10))
    ]);

    let nr_epochs = 3;
    println!("starting trainig");
    for epoch in 0..nr_epochs{
        let mut tot_loss = 0.0;
        for i in 0..train_data.len(){
            let x = train_data[i].clone();
            let y = train_labels[i].clone();
            let preds = model.forward(x);
            let (loss_sum,loss) = loss::MSEloss::get_loss(&preds, &y);
            model.backward(loss, 0.001);
            tot_loss += loss_sum;
        }
        println!("Epoch: {epoch}, loss={}",tot_loss/train_data.len() as f64);
    }
    println!("training done");
    let mut nrcorrect = 0;
    for i in 0..test_data.len(){
        let preds = model.forward(test_data[i].clone());
        let mut mx = 0;
        for i in 1..preds.len(){
            if preds[i] > preds[mx]{
                mx = i;
            }
        }
        if test_labels[i][mx] == 1.0{
            nrcorrect += 1;
        }
    }
    println!("nrcorrect: {nrcorrect}, accuracy: {}", nrcorrect as f64 / test_data.len() as f64);
}
