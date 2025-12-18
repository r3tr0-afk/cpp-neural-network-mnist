#include<iostream>
#include<vector>
#include<fstream>
#include<sstream>
#include<string>

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
            double pixel = std::stoi(value)/255.0;
            sample.pixels.push_back(pixel);
        }

        images.push_back(sample);
    }
    return images;    
}