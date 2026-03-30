#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <random>
#include <utility>
#include "../include/neuralnet.h"

//using namespace std;

struct image{
    std::vector<double> pixels;
    int label;
};

std::vector<image> load_csv(const std::string& filename){
    std::vector<image> images;
    std::ifstream file(filename);

    if(file.is_open() == 0){
        std::cout << "cannot open file" << " " << filename << std::endl;
        return images;
    }

    std::string line;
    bool header = true;

    while(std::getline(file,line)){
        if(header){
            header = false;
            continue;
        }

        std::stringstream data(line);
        std::string value;
        image sample;

        sample.pixels.reserve(784);
        if(!std::getline(data,value,',')){
            continue;
        }
        
        sample.label = std::stoi(value);
        while(std::getline(data,value,',')){
            double pixel = (std::stoi(value)/255.0*0.99)+0.01;
            sample.pixels.push_back(pixel);
        }

        images.push_back(std::move(sample));
    }
    return images;    
}

std::vector<double> target_var(int label) {
    std::vector<double> target(10, 0.01);
    target[label] = 0.99;
    return target;
}


double accuracy(neuralnet& network,const std::vector<image>& data){
    int total = data.size();
    int correct_pred_count = 0;
    for (int i = 0; i < total; i++)
    {
        std::vector<double> output = network.forward(data[i].pixels);
        int prediction = 0;
        double max_prob = output[0];
        for (int j = 1; j < 10; j++)
        {
            if(output[j]>max_prob){
                max_prob = output[j];
                prediction=j;
            }
        }
        if(prediction==data[i].label){
            correct_pred_count++;
        }
    }
    double accuracy = (double)correct_pred_count/total*100;
    return accuracy;
}

int main() {
    std::mt19937 rng(std::random_device{}());

    std::vector<image> training_data = load_csv("../data/mnist_train.csv");
    std::vector<image> test_data = load_csv("../data/mnist_test.csv");
    if (training_data.empty()){
        return 1;
    }else{
        std::cout << "training : " << training_data.size() << "images\n"; 
    }
    if(test_data.empty()){
        return 1;
    }else{
        std::cout << "testing : " << test_data.size() << "images\n";
    }

    int input_neurons=784;
    int hidden_neurons=128;
    int output_neurons=10;
    int epochs=5; 

    neuralnet nn(input_neurons,hidden_neurons,output_neurons);
    double best_accuracy = 0.0;
    int best_epoch=0;
    for (int i = 0; i < epochs; i++)
    {
        std::cout << "epoch" << i+1 << "\n";
        std::shuffle(training_data.begin(),training_data.end(),rng);
        for (int j = 0; j < training_data.size(); j++)
        {
            std::vector<double> targets = target_var(training_data[j].label);
            nn.train(training_data[j].pixels,targets);
        }
        double accuracy_after_epoch = accuracy(nn,test_data);
        if (accuracy_after_epoch >= best_accuracy)
        {
            best_accuracy = accuracy_after_epoch;
            best_epoch = i+1;
            nn.save_model("../models/model_0.txt");
        }
        
        std::cout << accuracy_after_epoch << "\n";
    }
    std::cout << "\nbest epoch : " << best_epoch << "\n";
    std::cout << "best accuracy : " << best_accuracy << "\n"; 
    return 0;
}