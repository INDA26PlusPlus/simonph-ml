pub struct MSEloss;

impl MSEloss{
    pub fn get_loss(preds:&Vec<f64>, y:&Vec<f64>)->(f64, Vec<f64>){
        let mut ret = vec![0.0;preds.len()];
        let mut sum = 0.0;
        for i in 0..preds.len(){
            let diff = preds[i] - y[i];
            sum += diff*diff;
            ret[i] = diff*2.0;
        }
        return (sum/preds.len() as f64,ret);
    }
}