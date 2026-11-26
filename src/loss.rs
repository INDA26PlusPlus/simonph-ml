pub struct MSEloss;

impl MSEloss{
    pub fn get_loss(preds:&Vec<f32>, y:&Vec<f32>)->(f32, Vec<f32>){
        let mut ret = vec![0.0;preds.len()];
        let mut sum = 0.0;
        for i in 0..preds.len(){
            let diff = preds[i] - y[i];
            sum += diff*diff;
            ret[i] = diff*2.0;
        }
        return (sum/preds.len() as f32,ret);
    }
}

pub struct CrossEntropyLoss;

impl CrossEntropyLoss{
    pub fn get_loss(preds:&Vec<f32>, y:&Vec<f32>)->(f32,Vec<f32>){
        let mut p:Vec<f32> = preds.iter().map(|x| x.exp()).collect();
        let combined:f32 = p.iter().sum();
        p = p.iter().map(|x| x/combined).collect();
        let ret = p.iter().zip(y.iter()).map(|(x,y)| x - y).collect();
        let sum = -p.iter().zip(y.iter()).map(|(p,y)| y*(p + 1e-12).ln()).sum::<f32>();
        return (sum,ret);
    }
}