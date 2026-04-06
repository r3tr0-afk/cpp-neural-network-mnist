#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include<iostream>
#include<vector>
#include"matrix.h"
#include<stdexcept>
#include<string>
#include<fstream>

//using namespace std;

class neuralnet{
    private:
        int input_neurons;
        int hidden_neurons;
        int output_neurons;
        double learning_rate;

        matrix w_input_hidden;
        matrix w_hidden_output;
        matrix b_hidden;
        matrix b_output;
    
    public:
        neuralnet(int input, int hidden, int output, double lr = 0.1) :
            input_neurons(input),
            hidden_neurons(hidden),
            output_neurons(output),
            learning_rate(lr),
            w_input_hidden(hidden,input),
            w_hidden_output(output,hidden),
            b_hidden(hidden,1),
            b_output(output,1){

                double limit1 = std::sqrt(6.0 / (input + hidden));
                w_input_hidden.random_number_fill(-limit1, limit1);

                double limit2 = std::sqrt(6.0 / (hidden + output));
                w_hidden_output.random_number_fill(-limit2, limit2);

                b_hidden.random_number_fill(0.0, 0.0);
                b_output.random_number_fill(0.0, 0.0);
            }
        
        std::vector<double> forward(const std::vector<double>& input_data){
            if(input_data.size()!=input_neurons){
                throw std::invalid_argument("input size didnt match with number of input neurons\n");
            }

            matrix inputs(input_neurons,1);
            for (int i = 0; i < input_neurons; i++)
            {
                inputs.atpos_modifiable(i,0) = input_data[i];
            }

            matrix hidden = w_input_hidden.matmultiply(inputs);
            hidden.add_inplace(b_hidden);
            hidden.sigmoid_inplace();

            matrix final_out = w_hidden_output.matmultiply(hidden);
            final_out.add_inplace(b_output);
            final_out.softmax_inplace();

            std::vector<double> output_data;
            output_data.reserve(output_neurons);
            for (int i = 0; i < output_neurons; i++)
            {
                output_data.push_back(final_out.atpos(i,0));
            }
            
            return output_data;
        } 
        
        void train(const std::vector<double>& input_data, const std::vector<double>& target_data){
            matrix inputs(input_neurons,1);
            for (int i = 0; i < input_neurons; i++)
            {
                inputs.atpos_modifiable(i,0) = input_data[i];
            }

            matrix hidden = w_input_hidden.matmultiply(inputs);
            hidden.add_inplace(b_hidden);
            hidden.sigmoid_inplace();

            matrix final_out = w_hidden_output.matmultiply(hidden);
            final_out.add_inplace(b_output);
            final_out.softmax_inplace();

            matrix e_output(output_neurons,1);
            for (int i = 0; i < output_neurons; i++)
            {
                e_output.atpos_modifiable(i,0) = target_data[i] - final_out.atpos(i,0);
            }

            matrix e_hidden = w_hidden_output.transpose_matmultiply_vec(e_output);

            matrix output_gradients = e_output;
            output_gradients.apply_learning_rate_inplace(learning_rate);

            matrix hidden_gradients = hidden;
            hidden_gradients.sigmoid_derivative_inplace();
            hidden_gradients.element_wise_multiply_inplace(e_hidden);
            hidden_gradients.apply_learning_rate_inplace(learning_rate);

            matrix w_hidden_output_deltas = matrix::outer_product(output_gradients, hidden);
            w_hidden_output.add_inplace(w_hidden_output_deltas);
            b_output.add_inplace(output_gradients);

            matrix w_input_hidden_deltas = matrix::outer_product(hidden_gradients, inputs);
            w_input_hidden.add_inplace(w_input_hidden_deltas);
            b_hidden.add_inplace(hidden_gradients);            
        }
        
        void save_model(const std::string& path){
            std::ofstream file(path);
            if (file.is_open()==0)
            {
                std::cout << "saving model - error while loading file\n";
                return;
            }
            file << w_input_hidden.numrows() << " " << w_input_hidden.numcols() << "\n";
            for (int i = 0; i < w_input_hidden.numrows(); i++)
            {
                for (int j = 0; j < w_input_hidden.numcols(); j++)
                {
                    file << w_input_hidden.atpos(i,j) << " ";
                }
                file << "\n";
            }
            file << b_hidden.numrows() << " " << b_hidden.numcols() << "\n";
            for (int i = 0; i < b_hidden.numrows(); i++)
            {
                file << b_hidden.atpos(i,0) << "\n";
            }
            file << w_hidden_output.numrows() << " " << w_hidden_output.numcols() << "\n";
            for (int i = 0; i < w_hidden_output.numrows(); i++)
            {
                for (int j = 0; j < w_hidden_output.numcols(); j++)
                {
                    file << w_hidden_output.atpos(i,j) << " ";
                }
                file << "\n";
            }
            file << b_output.numrows() << " " << b_output.numcols() << "\n";
            for (int i = 0; i < b_output.numrows(); i++)
            {
                file << b_output.atpos(i,0) << "\n";
            }

            file.close();
            std::cout << "model saved\n"; 
        }

        void load_model(const std::string& path){
            std::ifstream file(path);
            if(file.is_open()==0){
                std::cout << "loading model - error while loading file \n";
                return;
            }
            int rows;
            int cols;
            double val;

            file >> rows >> cols;
            if (rows!=hidden_neurons || cols!=input_neurons)
            {
                std::cout << "dimensions didnt match\n";
                return;
            }
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    file >> val;
                    w_input_hidden.atpos_modifiable(i,j)=val;
                }
                
            }
            file >> rows >> cols;
            for (int i = 0; i < rows; i++)
            {
                file >> val;
                b_hidden.atpos_modifiable(i,0)=val;
            }
            file >> rows >> cols;
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    file >> val;
                    w_hidden_output.atpos_modifiable(i,j)=val;
                }

            }            
            file >> rows >> cols;
            for (int i = 0; i < rows; i++)
            {
                file >> val;
                b_output.atpos_modifiable(i,0)=val;
            }            
            file.close();
            std::cout << "model loaded\n";
            
        }
};
#endif