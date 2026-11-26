

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
fn validate(model:&mut model::Model, data:&Vec<Vec<f64>>, labels:&Vec<Vec<f64>>)->(i32,f64){
    let mut nrcorrect = 0;
    for i in 0..data.len(){
        let preds = model.forward(data[i].clone());
        let mut mx = 0;
        for i in 1..preds.len(){
            if preds[i] > preds[mx]{
                mx = i;
            }
        }
        if labels[i][mx] == 1.0{
            nrcorrect += 1;
        }
    }
    return (nrcorrect, nrcorrect as f64/data.len() as f64);
}
fn main(){
    let Mnist {
       trn_img,
       trn_lbl,
       tst_img,
       tst_lbl,
       val_img,
       val_lbl
    } = MnistBuilder::new()
        .base_path("mnist")
       .label_format_one_hot()
       .training_set_length(50000)
       .validation_set_length(10000)
       .test_set_length(10000)
       .finalize();
    let train_data:Vec<Vec<f64>> = trn_img.chunks(28*28).map(|img| img.iter().map(|x| *x as f64/255.0).collect()).collect();
    let train_labels:Vec<Vec<f64>> = trn_lbl.chunks(10).map(|img| img.iter().map(|x| *x as f64).collect()).collect();

    let test_data:Vec<Vec<f64>> = tst_img.chunks(28*28).map(|img| img.iter().map(|x| *x as f64/255.0).collect()).collect();
    let test_labels:Vec<Vec<f64>> = tst_lbl.chunks(10).map(|img| img.iter().map(|x| *x as f64).collect()).collect(); 

    let val_data:Vec<Vec<f64>> = val_img.chunks(28*28).map(|img| img.iter().map(|x| *x as f64/255.0).collect()).collect();
    let val_labels:Vec<Vec<f64>> = val_lbl.chunks(10).map(|img| img.iter().map(|x| *x as f64).collect()).collect(); 

    let mut model = model::Model::new(vec![
        Box::new(NeuronLayer::new(28*28, 128)), 
        Box::new(ReLu::new()),
        Box::new(NeuronLayer::new(128,10))
    ]);
    let mut best_val = 0;
    let patiense = 4;
    let mut cur_failed = 0;
    let nr_epochs = 30;
    let mut lr = 0.01;
    println!("starting trainig");
    for epoch in 0..nr_epochs{
        let mut tot_loss = 0.0;
        for i in 0..train_data.len(){
            let x = train_data[i].clone();
            let y = train_labels[i].clone();
            let preds = model.forward(x);
            let (loss_sum,loss) = loss::CrossEntropyLoss::get_loss(&preds, &y);
            model.backward(loss, lr);
            tot_loss += loss_sum;
        }
        let (val_correct, val_accuracy) = validate(&mut model, &val_data, &val_labels);
        
        println!("Epoch: {epoch}, loss={} val_accuracy: {val_accuracy}",tot_loss/train_data.len() as f64);
        if val_correct >= best_val{
            best_val = val_correct;
        }
        else{
            cur_failed += 1;
            if cur_failed >= patiense{
                println!("stopping early");
                break;
            }
        }
        lr *= 0.9;
    }
    println!("training done");
    let (nrcorrect, accuracy) = validate(&mut model, &test_data, &test_labels);
    println!("nrcorrect: {nrcorrect}, accuracy: {}", accuracy);
}
