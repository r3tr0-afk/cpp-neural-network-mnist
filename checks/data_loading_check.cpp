#include<iostream>
#include<vector>
#include<fstream>
#include<sstream>
#include<string>

using namespace std;

struct image{
    vector<double> pixels;
    int label;
};

vector<image> load_csv(const string& filename){
    vector<image> images;
    ifstream file(filename);

    if(file.is_open() == 0){
        cout << "cannot open file" << " " << filename << endl;
    }

    string line;
    bool header = true;

    while(getline(file,line)){
        if(header){
            header = false;
            continue;
        }

        stringstream data(line);
        string value;
        image sample;

        sample.pixels.reserve(784);
        if(!getline(data,value,',')){
            continue;
        }
        sample.label = stoi(value);
        while(getline(data,value,',')){
            double pixel = stoi(value)/255.0;
            sample.pixels.push_back(pixel);
        }

        images.push_back(sample);

    }
    return images;    
}


int main(){
    vector<image> data = load_csv("../data/mnist_train.csv");
    cout << "total data size : " << data.size() << endl ;
    cout << "label : " << data[0].label << endl;
    cout << "pixel count :" << data[0].pixels.size() << endl;
    cout << "pixel values : " << endl;
    for (int i = 0; i < data[0].pixels.size(); i++)
    {
        cout << data[0].pixels[i] << " ";
    } 
    cout << "\n";
    for (int i = 1; i < data.size(); i++)
    {
        if(data[i].pixels.size()!=data[i-1].pixels.size()){
            cout << "size didnt match " << endl; 
        }
    }
    
    return 0;
}