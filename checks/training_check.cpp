#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <algorithm> 
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

        images.push_back(sample);
    }
    return images;    
}

std::vector<double> create_target(int label) {
    std::vector<double> target(10, 0.01);
    target[label] = 0.99;
    return target;
}

int main() {
    std::srand(std::time(0)); 

    std::vector<image> training_data = load_csv("../data/mnist_train.csv");

    if (training_data.empty()){
        return 1;
    }

    int input_neurons=784;
    int hidden_neurons=128;
    int output_neurons=10;
    int epochs=5; 

    neuralnet nn(input_neurons,hidden_neurons,output_neurons);

    for (int i = 0; i < epochs; i++)
    {
        std::random_shuffle(training_data.begin(),training_data.end());
        for (int j = 0; j < training_data.size(); j++)
        {
            std::vector<double> targets = create_target(training_data[j].label);
            nn.train(training_data[j].pixels,targets);
        }
    }

    int test_index = std::rand()%training_data.size();
    image test_img = training_data[test_index];

    std::vector<double> result = nn.forward(test_img.pixels);

    int predicted_label = 0;
    double max_prob = result[0];
    
    std::cout << "probabilities\n";
    for (int i = 0; i < 10; i++)
    {
        std::cout << i << ": " << result[i] << "\n";
        if (result[i] > max_prob)
        {
            max_prob = result[i];
            predicted_label = i;
        }
    }

    std::cout << "actual : " << test_img.label << "\n";
    std::cout << "predicted : " << predicted_label << "\n";

    return 0;
}