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
        
        void train(vector<double> input_data, vector<double> target_data){
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

            matrix targets(output_neurons,1);
            for (int i = 0; i < output_neurons; i++)
            {
                targets.atpos_modifiable(i,0) = target_data[i];
            }

            matrix e_output = targets.matsubtract(final_outputs);
            matrix w_hidden_output_t = w_hidden_output.mattranspose();
            matrix e_hidden = w_hidden_output_t.matmultiply(e_output);

            double learning_rate = 0.1;
            matrix output_gradients = final_outputs.sigmoid_derivative();
            output_gradients = output_gradients.element_wise_multiply(e_output);
            output_gradients = output_gradients.apply_learning_rate(learning_rate);

            matrix hidden_gradients = hidden_outputs.sigmoid_derivative();
            hidden_gradients = hidden_gradients.element_wise_multiply(e_hidden);
            hidden_gradients = hidden_gradients.apply_learning_rate(learning_rate);

            matrix hidden_outputs_t = hidden_outputs.mattranspose();
            matrix w_hidden_output_deltas = output_gradients.matmultiply(hidden_outputs_t);
            w_hidden_output = w_hidden_output.matadd(w_hidden_output_deltas);
            b_output = b_output.matadd(output_gradients);

            matrix inputs_t = inputs.mattranspose();
            matrix w_input_hidden_deltas = hidden_gradients.matmultiply(inputs_t);
            w_input_hidden = w_input_hidden.matadd(w_input_hidden_deltas);
            b_hidden = b_hidden.matadd(hidden_gradients);            
            
        }
};
#endif