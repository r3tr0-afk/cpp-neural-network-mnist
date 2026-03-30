#include<iostream>
#include<vector>
#include<fstream>
#include<sstream>
#include<string>
#include<algorithm>
#include<utility>
#include"../include/neuralnet.h"

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

int main(){
    int input_neurons=784;
    int hidden_neurons=128;
    int output_neurons=10;

    neuralnet nn(input_neurons,hidden_neurons,output_neurons);
    std::string model = "../models/model_0.txt";
    nn.load_model(model);
    std::vector<image> test_data = load_csv("../data/prediction.csv");
    std::cout << test_data.size() <<"\n";
    for (int i = 0; i < test_data.size(); i++)
    {
        std::cout << "image[" << i+1 << "] : \n";
        std::vector<double> output = nn.forward(test_data[i].pixels);
        int predicted = 0;
        double max_prob = output[0];
        std::cout << "  probabilities : \n";
        for (int j = 0; j < 10; j++)
        {
            std::cout << j << " : " << output[j] << "\n";
            if(output[j]>max_prob){
                max_prob=output[j];
                predicted=j;
            }
        }
       std::cout << "\n";
       std::cout << "       predicted : " << predicted << "\n";
       std::cout << "       probability : " << max_prob << "\n";
       std::cout << "       actual : " << test_data[i].label << "\n"; 
       std::cout << "\n";
    }

    return 0;
}