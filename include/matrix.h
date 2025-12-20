#ifndef MATRIX_H
#define MATRIX_H

#include<iostream>
#include<vector>
#include<stdexcept>
#include<cmath>
#include<cstdlib>

using namespace std;

class matrix{
    private:
        int rows;
        int cols;
        std::vector<double> data;
      
    public :
        matrix(int r,int c):rows(r),cols(c){
            data.resize(r*c,0.0);
        }

        int numrows(){
            return rows;
        }

        int numcols(){
            return cols;
        }

        double& atpos_modifiable(int r, int c){
            return data[(r*cols)+c];
        };
        
        double atpos(int r,int c){
            return data[(r*cols)+c];
        };
        
        matrix matmultiply(matrix& other){
            if(cols!=other.rows)
            {
                throw std::invalid_argument("multiplication-dimensions didnt match\n");
            }
                matrix res(rows, other.cols);
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < other.cols; j++)
                    {
                        double sum=0;
                        for (int k = 0; k < cols; k++)
                        {
                            sum += atpos(i,k) * other.atpos(k,j);
                        }
                        res.atpos_modifiable(i,j) = sum;
                    }
                
                }
                return res;
        };

        matrix matadd(matrix& other){
            if(rows!=other.rows || cols!=other.cols)
            {
                throw std::invalid_argument("addition-dimensions didnt match\n");
            }
                matrix res(rows,cols);
                for (int i = 0; i < data.size(); i++)
                {
                    res.data[i] = data[i] + other.data[i];
                }
                return res;
        };

        matrix matsubtract(matrix& other){
            if(rows!=other.rows || cols!=other.cols)
            {
                throw std::invalid_argument("subtraction-dimensions didnt match\n");
            }
                matrix res(rows,cols);
                for (int i = 0; i < data.size(); i++)
                {
                    res.data[i] = data[i] - other.data[i];
                }
                return res;
        };

        matrix mattranspose(){
            matrix res(cols,rows);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    res.atpos_modifiable(j,i) = atpos(i,j);
                }
                
            }
            return res;
        };

        void printmat(){
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    std::cout << atpos(i,j) << " "; 
                }
                std::cout << "\n";
            }  
        };

        void random_number_fill(){
            for (int i = 0; i < data.size(); i++)
            {
                data[i] = ((double)std::rand()/RAND_MAX)*2.0-1.0;
            }
        };

        matrix sigmoid(){
            matrix res(rows,cols);
            for (int i = 0; i < data.size(); i++)
            {
                res.data[i] = 1.0 / (1.0 + std::exp(-1*data[i]));
            }
            return res;
        }; 
        
        matrix element_wise_multiply(matrix& other){
            if(rows!=other.rows || cols!=other.cols)
            {
                throw std::invalid_argument("dimensions didnt match\n");
            }
            matrix res(rows,cols);
            for (int i = 0; i < data.size(); i++)
            {
                res.data[i] = data[i]*other.data[i];
            }
            return res;
        }

        matrix sigmoid_derivative(){
            matrix res(rows,cols);
            for (int i = 0; i < data.size(); i++)
            {
                res.data[i] = data[i]*(1-data[i]);
            }
            return res;
        }

        matrix apply_learning_rate(double n){
            matrix res(rows,cols);
            for (int i = 0; i < data.size(); i++)
            {
                res.data[i] = n*data[i];
            }
            return res;
        }
};
#endif