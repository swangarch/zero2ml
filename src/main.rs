fn cal_err(k: f64, b: f64, mileage: f64, price: f64) -> f64 {
    let err = (k * mileage + b) - price;

    return err;
}

fn cal_msq(vec_mileage: Vec<f64>, vec_price: Vec<f64>, k: f64, b: f64) -> f64 {

    let mut sum : f64 = 0.0;

    for i in 0..vec_mileage.len() {
        sum += cal_err(k, b, vec_mileage[i], vec_price[i]);
    }

    return (sum / (vec_mileage.len() as f64)) as f64;
}   

fn main() -> Result<(), Box<dyn std::error::Error>> {

    let mileage: Vec<f64> = vec![
        5000.0,  10000.0, 15000.0, 20000.0, 25000.0,
        30000.0, 40000.0, 50000.0, 60000.0, 70000.0,
    ];

    let price: Vec<f64> = vec![
        19120.0, 17950.0, 17110.0, 15870.0, 15220.0,
        13980.0, 12150.0, 10120.0,  8350.0,  6150.0,
    ];

    if mileage.len() != mileage.len() {
        println!("Error: Mismatch length.");
    }

    for i in 0..mileage.len() {
        println!("mileage: {}  price: {}", mileage[i], price[i]);
    }

    println!();

    let k = 1.0;
    let b = 2.0;

    for i in 0..mileage.len() {
        println!("Error {}", cal_err(k, b, mileage[i], price[i]));
    }

    println!("MSE: {}", cal_msq(mileage, price, k, b));

    Ok(())
}
