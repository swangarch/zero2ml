use std::io::{stdin};
use std::fs;

fn cal_pred(theta0: f32, theta1: f32, mileage: f32) -> f32 {

    let res = theta0 + theta1 * mileage;

    if res < 0.0 {
        return 0.0;
    }
    return res;
}

fn read_prompt() -> Result<f32, String> {

    println!("Plase enter your car's mileage: ");
    let mut mileage = String::new();

    stdin()
        .read_line(& mut mileage)
        .map_err(|_| "Failed to read mileage.".to_string())?;
    
    let fmileage: f32 = mileage
        .trim()
        .parse()
        .map_err(|_| "Please enter a valid number.".to_string())?;

    Ok(fmileage)
}

fn load_weight(path: &String, theta: &mut Vec<f32>) -> Result<(), String> {
    match fs::read_to_string(&path) {
            Ok(text) => {

                if text.lines().count() != 1 {
                    println!("Error: Invalid weight file, use default: [0.0] [0.0]\n----------------------------------------------");
                }
                else {
                    for line in text.lines() {
                        let parts: Vec<&str> = line.split_whitespace().collect();

                        if parts.len() != 2 {
                            println!("Error: Invalid weights, use default: [0.0] [0.0]\n----------------------------------------------");
                            break;
                        }
                        theta[0] = parts[0].parse().map_err(|_| "Invalid float number".to_string())?;
                        theta[1] = parts[1].parse().map_err(|_| "Invalid float number".to_string())?;
                        println!("Weights loaded from file: [{}] [{}]\n----------------------------------------------", theta[0], theta[1]);
                    }
                }
            }
            Err(e) => {
                println!("Error: Failed to read or parse weight data because: {}, use default: [0.0] [0.0]\n----------------------------------------------", e);
            }
        }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {

    let args : Vec<String> = std::env::args().collect();

    if args.len() != 1 && args.len() != 2 {
        return Err("Invalid number of arguments, accept 1 or 2.".into());
    }

    let mut theta: Vec<f32> = vec![0.0, 0.0];

    if args.len() == 1 {
        println!("No weights, use default weights: [0.0] [0.0]\n----------------------------------------------");
    }
    else if args.len() == 2 {
        if let Err(e) = load_weight(&args[1], &mut theta) {
            println!("Error: {}, use default weights: [0.0] [0.0]\n----------------------------------------------", e);
        }
    }
    
    match read_prompt() {
        Ok(mileage) => {

            if mileage < 0.0 {
                println!("Error: please enter a non negative number.");
            }
            else {
                println!("Your car worth around {}", cal_pred(theta[0], theta[1], mileage));
            }
        }
        Err(e) => {
            println!("Error: {}", e);
        }
    }

    Ok(())
}
