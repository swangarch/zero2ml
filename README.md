# zero2ml

**zero2ml** is a **practice project** to learn machine learning **from zero**,  
by re-implementing fundamental machine learning algorithms.

The purpose of this repository is **educational**:  
to understand how classic ML models work internally,  
not to build a production-ready library.

---

## ✨ Highlights

- **From Zero** → no external ML frameworks, everything implemented from scratch  
- **Educational** → designed as a step-by-step learning exercise  

---

## 📌 Current Progress

- [x] Linear Regression (done 🎉 with Rust)  
- [ ] Logistic Regression (WIP)  
- [ ] Multilayer Perceptron (planned)
- [ ] More to come... 🚀


### 1. Linear Regression ✅

Linear regression is implemented with three programs, main algorithm is implemented with Rust, visualizer with python:

#### 📖 Programs

1. **Predictor**  
   - Usage:  
     ```
     cargo run <weight.txt>
     ```  
   - Reads model parameters from `weight.txt`.  
   - If no `weight.txt` is available, it uses default weights `(0, 0)`.  
   - Outputs the predicted price for the given `km` value.  

2. **Trainer**  
   - Usage:  
     ```
     cargo run <data.csv>
     ```  
   - Takes a CSV file containing training data (`km`, `price`).  
   - Trains the linear regression model using gradient descent.  
   - Saves the final model parameters into `weight.txt`.  

3. **Visualizer**  
   - Usage:  
     ```
     python visualize.py <path_to_training_program> <path_to_data_csv>
     ```  
   - Runs the Rust training program, collects the results,  
     and visualizes the training process and regression line using Python (matplotlib).  

---

## 🎯 Goal

By the end of this project, the repository will contain:  
1. A minimal set of ML algorithms.  
2. Python scripts & notebooks for visualizing training results  
3. A clean and educational codebase that shows **how ML works under the hood**
