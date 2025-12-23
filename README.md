a fully connected neural network implemented from scratch in C++ to recognize handwritten digits

- dataset  -  **MNIST in CSV format**
- link     -  https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

- src/main.cpp      - training 
- src/predict.cpp   - prediction
- models/           - already trained model weights
- include/          - math and the core logic behind network
- data/             - data


- saved models go into models/
- for prediction change the data in data/prediction.csv
- *make sure the paths are correct in src/main.cpp and src/predict.cpp  


how to run :

1. train the model : compile and run

       g++ src/main.cpp -o train
       ./train

2. make predictions : compile and run

       g++ src/predict.cpp -o predict
       ./predict

   
