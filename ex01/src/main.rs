fn normalization (v: &mut Vec<f32>, range: &mut Vec<f32>) {
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

fn denormalization(value: f32, range: &mut Vec<f32>) {
    return value * (range[1] - range[0]) + range[0];
}

fn cal_pred(theta0: f32, theta1: f32, mileage: f32) -> f32 {

    return theta0 + theta1 * mileage;
}

fn cal_err(theta0: f32, theta1: f32, mileage: f32, price: f32) -> f32 {

    return cal_pred(theta0, theta1, mileage) - price;
}

fn cal_msqe(mileages: &Vec<f32>, prices: &Vec<f32>, theta0: f32, theta1: f32) -> f32 {

    let mut sum : f32 = 0.0;
    let mut err;

    for i in 0..mileages.len() {
        err = cal_err(theta0, theta1, mileages[i], prices[i]);
        sum += err * err;
    }
    return (sum / (mileages.len() as f32)) as f32;
}

fn cal_grad0(mileages: &Vec<f32>, prices: &Vec<f32>, theta0: f32, theta1: f32) -> f32 {

    let mut sum : f32 = 0.0;

    for i in 0..mileages.len() {
        sum += cal_err(theta0, theta1, mileages[i], prices[i]);
    }
    return (sum / (mileages.len() as f32)) as f32;
}

fn cal_grad1(mileages: &Vec<f32>, prices: &Vec<f32>, theta0: f32, theta1: f32) -> f32 {

    let mut sum : f32 = 0.0;

    for i in 0..mileages.len() {
        sum += cal_err(theta0, theta1, mileages[i], prices[i]) * mileages[i];
    }
    return (sum / (mileages.len() as f32)) as f32;
}

fn training(mileages: &Vec<f32>, prices: &Vec<f32>, rate: f32, theta: &mut Vec<f32>) {

    let mut i = 0;
    let mut msqe: f32;
    let mut last_msque: f32 = f32::INFINITY;

    while i < 50000 {

        msqe = cal_msqe(&mileages, &prices, theta[0], theta[1]);

        if (msqe - last_msque).abs() < 0.00000001 {
            break;
        }
        
        theta[0] -= rate * cal_grad0(&mileages, &prices, theta[0], theta[1]);
        theta[1] -= rate * cal_grad1(&mileages, &prices, theta[0], theta[1]);

        if i % 100 == 0 {
            println!("{} {} {}", theta[0], theta[1], msqe);
        }
        i += 1;
        last_msque = msqe;
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {

    let mut mileages: Vec<f32> = vec![240000.0, 139800.0, 150500.0, 185530.0, 176000.0, 114800.0, 166800.0, 89000.0, 144500.0, 84000.0, 82029.0, 63060.0, 74000.0, 97500.0, 67000.0, 76025.0, 48235.0, 93000.0, 60949.0, 65674.0, 54000.0, 68500.0, 22899.0, 61789.0];

    let mut prices: Vec<f32> = vec![3650.0, 3800.0, 4400.0, 4450.0, 5250.0, 5350.0, 5800.0, 5990.0, 5999.0, 6200.0, 6390.0, 6390.0, 6600.0, 6800.0, 6800.0, 6900.0, 6900.0, 6990.0, 7490.0, 7555.0, 7990.0, 7990.0, 7990.0, 8290.0];

    let mut theta: Vec<f32> = vec![0.0, 0.0];
    let mut m_range: Vec<f32> = vec![f32::INFINITY, f32::NEG_INFINITY];
    let mut p_range: Vec<f32> = vec![f32::INFINITY, f32::NEG_INFINITY];

    normalization(&mut mileages, &mut m_range);
    normalization(&mut prices, &mut p_range);

    // println!("mileages: {:#?}", mileages);
    // println!("prices  : {:#?}", prices);

    training(&mileages, &prices, 0.001, &mut theta);

    // println!("theta  : {:#?}", theta);

    // let real_mileage = 65000.0;

    // let map_mileage = map(real_mileage, &mut m_range);
    // let map_price = cal_pred(theta[0], theta[1], map_mileage);
    // let real_price = map_price * (p_range[1] - p_range[0]) + p_range[0];

    // println!("Prediction price for {} mileage is {}", real_mileage, real_price);

    Ok(())
}
