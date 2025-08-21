use std::fs::File;
use std::io::Write;
use csv::Reader;

fn normalize(v: &mut Vec<f32>, range: &mut Vec<f32>) {
    for i in 0..v.len() {
        if v[i] < range[0] {
            range[0] = v[i];
        }
        if v[i] > range[1] {
            range[1] = v[i];
        }
    }
    let len = range[1] - range[0];
    for i in 0..v.len() {
        v[i] = (v[i] - range[0]) / len;
    }
}

fn denorm_value(value: f32, range: & Vec<f32>) -> f32 {

    return value * (range[1] - range[0]) + range[0];
}

fn denorm_theta(m_range: &Vec<f32>, p_range: &Vec<f32>, theta: &Vec<f32>, o_theta: &mut Vec<f32>) {

    let x0 = denorm_value(0.0, &m_range);
    let y0 = denorm_value(theta[0], &p_range);
    let x1 = denorm_value(1.0, &m_range);
    let y1 = denorm_value(theta[0] + theta[1], &p_range);

    o_theta[1] = (y1 - y0) / (x1 - x0);
    o_theta[0] = y1 - o_theta[1] * x1;
}

fn cal_pred(theta: &Vec<f32>, mileage: f32) -> f32 {

    return theta[0] + theta[1] * mileage;
}

fn cal_err(theta: &Vec<f32>, mileage: f32, price: f32) -> f32 {

    return cal_pred(&theta, mileage) - price;
}

fn cal_msqe(mileages: &Vec<f32>, prices: &Vec<f32>, theta: &Vec<f32>) -> f32 {

    let mut sum : f32 = 0.0;
    let mut err;

    for i in 0..mileages.len() {
        err = cal_err(&theta, mileages[i], prices[i]);
        sum += err * err;
    }
    return (sum / (mileages.len() as f32)) as f32;
}

fn cal_grad0(mileages: &Vec<f32>, prices: &Vec<f32>, theta: &Vec<f32>) -> f32 {

    let mut sum : f32 = 0.0;

    for i in 0..mileages.len() {
        sum += cal_err(&theta, mileages[i], prices[i]);
    }
    return (sum / (mileages.len() as f32)) as f32;
}

fn cal_grad1(mileages: &Vec<f32>, prices: &Vec<f32>, theta: &Vec<f32>) -> f32 {

    let mut sum : f32 = 0.0;

    for i in 0..mileages.len() {
        sum += cal_err(&theta, mileages[i], prices[i]) * mileages[i];
    }
    return (sum / (mileages.len() as f32)) as f32;
}

fn save_weight(o_theta: &Vec<f32>) {
    match File::create("weight.txt") {
        Ok(mut file) => {
            let weight = format!("{} {}", o_theta[0], o_theta[1]);
            match file.write_all(weight.as_bytes()) {
                Ok(()) => {}
                Err(e) => {
                    println!("Error: {}", e);
                }
            }
        }   
        Err(e) => {
            println!("Error: {}", e);
        }
    }
}

fn training(mileages: &mut Vec<f32>, prices: &mut Vec<f32>, rate: f32) -> Result<(), String> {

    let mut m_range: Vec<f32> = vec![f32::INFINITY, f32::NEG_INFINITY];
    let mut p_range: Vec<f32> = vec![f32::INFINITY, f32::NEG_INFINITY];
    let mut theta: Vec<f32> = vec![0.0, 0.0];
    let mut o_theta: Vec<f32> = vec![0.0, 0.0];

    normalize(mileages, &mut m_range);
    normalize(prices, &mut p_range);

    let mut i = 0;
    let mut msqe: f32;
    let mut last_msque: f32 = f32::INFINITY;
    
    while i < 50000 {
        msqe = cal_msqe(&mileages, &prices, &theta);

        if (msqe - last_msque).abs() < 0.00000001 {
            break;
        }

        theta[0] -= rate * cal_grad0(&mileages, &prices, &theta);
        theta[1] -= rate * cal_grad1(&mileages, &prices, &theta);

        if i % 100 == 0 {

            denorm_theta(&m_range, &p_range, &theta, &mut o_theta);
            println!("{} {} {}", o_theta[0], o_theta[1], msqe);
        }
        i += 1;
        last_msque = msqe;
    }

    save_weight(&o_theta);

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {

    let mut mileages: Vec<f32> = vec![];
    let mut prices: Vec<f32> = vec![];

    let args : Vec<String> = std::env::args().collect();

    if args.len() != 2 {
        return Err("Invalid number of arguments, usage: linear_regression data.csv".into());
    }

    let mut datacsv = Reader::from_path(&args[1])?;

    for data in datacsv.records() {
        let info = data?;

        mileages.push(info[0].parse::<f32>()?);
        prices.push(info[1].parse::<f32>()?);
    }

    match training(&mut mileages, &mut prices, 0.001) {
        Ok(()) => {}
        Err(e) => {
            println!("Error: {}", e);
        }
    }

    Ok(())
}
