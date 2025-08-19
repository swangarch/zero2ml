fn cal_err(k: f32, b: f32, mileage: f32, price: f32) -> f32 {
    let err = (k * mileage + b) - price;

    return err;
}

fn cal_me(vec_mileage: &Vec<f32>, vec_price: &Vec<f32>, k: f32, b: f32) -> f32 {

    let mut sum : f32 = 0.0;

    for i in 0..vec_mileage.len() {
        sum += cal_err(k, b, vec_mileage[i], vec_price[i]);
    }
    return (sum / (vec_mileage.len() as f32)) as f32;
}

fn cal_grad(vec_mileage: &Vec<f32>, vec_price: &Vec<f32>, k: f32, b: f32) -> f32 {

    let mut sum : f32 = 0.0;

    for i in 0..vec_mileage.len() {
        sum += cal_err(k, b, vec_mileage[i], vec_price[i]) * vec_mileage[i];
    }
    return (sum / (vec_mileage.len() as f32)) as f32;
}

fn update_params(mileages: &Vec<f32>, prices: &Vec<f32>, rate: f32, params: &mut Vec<f32>) {

    let mut i = 0;

    while i < 100 {
        params[0] = rate * cal_me(&mileages, &prices, params[0], params[1]);
        params[1] = rate * cal_grad(&mileages, &prices, params[0], params[1]);

        println!("param0: {}\nparam1: {}\n-----------{}----------", params[0], params[1], i);
        i += 1;
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {

    let mileage: Vec<f32> = vec![
        5000.0,  10000.0, 15000.0, 20000.0, 25000.0,
        30000.0, 40000.0, 50000.0, 60000.0, 70000.0,
    ];

    let price: Vec<f32> = vec![
        19120.0, 17950.0, 17110.0, 15870.0, 15220.0,
        13980.0, 12150.0, 10120.0,  8350.0,  6150.0,
    ];

    // if mileage.len() != mileage.len() {
    //     println!("Error: Mismatch length.");
    // }

    // for i in 0..mileage.len() {
    //     println!("mileage: {}  price: {}", mileage[i], price[i]);
    // }

    // println!();

    // let k = 1.0;
    // let b = 2.0;

    // for i in 0..mileage.len() {
    //     println!("E: {}", cal_err(k, b, mileage[i], price[i]));
    // }

    // println!("ME: {}", cal_me(&mileage, &price, k, b));

    println!("_______________________________________________________");
    
    let mut params: Vec<f32> = vec![0.0, 0.0];

    update_params(&mileage, &price, 0.0001, &mut params);

    Ok(())
}
