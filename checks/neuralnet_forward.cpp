#include<iostream>
#include<vector>
#include"../include/neuralnet.h"

using namespace std;

int main(){
    neuralnet network(784,128,10);
    vector<double> input(784,0.5);
    vector<double> res = network.forward(input);
    for (int i = 0; i < res.size(); i++)
    {
        cout << i << " : " << res[i] << "\n";
    }
    return 0;
}