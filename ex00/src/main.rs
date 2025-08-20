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

fn main() -> Result<(), Box<dyn std::error::Error>> {

    let args : Vec<String> = std::env::args().collect();

    if args.len() != 1 && args.len() != 2 {
        println!("Error: the program only accept 1 or 2 argument.");
        return Err("Invalid number of arguments.".into());
    }

    let mut theta: Vec<f32> = vec![0.0, 0.0];

    if args.len() == 2 {
        match fs::read_to_string(&args[1]) {
            Ok(text) => {

                if text.lines().count() != 1 {
                    return Err("Error: Invalid weight file.".into());
                }

                for line in text.lines() {
                    let parts: Vec<&str> = line.split_whitespace().collect();

                    if parts.len() != 2 {
                        println!("Error: Invalid weights file");
                        continue;
                    }

                    theta[0] = parts[0].parse()?;
                    theta[1] = parts[0].parse()?;
                }
            }
            Err(_e) => {
                
            }
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
