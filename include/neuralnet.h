#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include<iostream>
#include<vector>
#include"matrix.h"
#include<stdexcept>

//using namespace std;

class neuralnet{
    private:
        int input_neurons;
        int hidden_neurons;
        int output_neurons;

        matrix w_input_hidden;
        matrix w_hidden_output;
        matrix b_hidden;
        matrix b_output;
    
    public:
        neuralnet(int input, int hidden, int output) :
            input_neurons(input),
            hidden_neurons(hidden),
            output_neurons(output),
            w_input_hidden(hidden,input),
            w_hidden_output(output,hidden),
            b_hidden(hidden,1),
            b_output(output,1){

                w_input_hidden.random_number_fill();
                w_hidden_output.random_number_fill();
                b_hidden.random_number_fill();
                b_output.random_number_fill();
            }
        
        std::vector<double> forward(std::vector<double> input_data){
            if(input_data.size()!=input_neurons){
                throw std::invalid_argument("input size didnt match with number of input neurons\n");
            }

            matrix inputs(input_neurons,1);
            for (int i = 0; i < input_neurons; i++)
            {
                inputs.atpos_modifiable(i,0) = input_data[i];
            }

            matrix hidden_inputs = w_input_hidden.matmultiply(inputs);
            matrix hidden_with_bias = hidden_inputs.matadd(b_hidden);
            matrix hidden_outputs = hidden_with_bias.sigmoid();

            matrix final_inputs = w_hidden_output.matmultiply(hidden_outputs);
            matrix final_with_bias = final_inputs.matadd(b_output);
            matrix final_outputs = final_with_bias.sigmoid();

            std::vector<double> output_data;
            for (int i = 0; i < output_neurons; i++)
            {
                output_data.push_back(final_outputs.atpos(i,0));
            }
            
            return output_data;
        }    
};
#endif