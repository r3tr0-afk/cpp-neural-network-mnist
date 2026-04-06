#ifndef MATRIX_H
#define MATRIX_H

#include<iostream>
#include<vector>
#include<stdexcept>
#include<cmath>
#include<cstdlib>
#include<random>


class matrix{
    private:
        int rows;
        int cols;
        std::vector<double> data;
      
    public :
        matrix(int r,int c):rows(r),cols(c){
            data.resize(r*c,0.0);
        }

        int numrows() const {
            return rows;
        }

        int numcols() const {
            return cols;
        }

        double& atpos_modifiable(int r, int c){
            return data[(r*cols)+c];
        };
        
        double atpos(int r,int c) const {
            return data[(r*cols)+c];
        };
        
        matrix matmultiply(const matrix& other) const {
            if(cols!=other.rows)
            {
                throw std::invalid_argument("multiplication-dimensions didnt match\n");
            }
            
            if (other.cols == 1) {
                matrix res(rows, 1);
                for (int i = 0; i < rows; i++) {
                    double sum = 0.0;
                    int row_a = i * cols;
                    for (int k = 0; k < cols; k++) {
                        sum += data[row_a + k] * other.data[k];
                    }
                    res.data[i] = sum;
                }
                return res;
            }

                // transpose other so inner loop accesses memory contiguously
                matrix otherT = other.mattranspose();
                matrix res(rows, other.cols);
                for (int i = 0; i < rows; i++)
                {
                    int row_a = i * cols;
                    for (int j = 0; j < other.cols; j++)
                    {
                        double sum = 0.0;
                        int row_bt = j * cols;  // row j of B^T = column j of B
                        for (int k = 0; k < cols; k++)
                        {
                            sum += data[row_a + k] * otherT.data[row_bt + k];
                        }
                        res.data[i * other.cols + j] = sum;
                    }
                }
                return res;
        };

        matrix matadd(const matrix& other) const {
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

        matrix matsubtract(const matrix& other) const {
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

        matrix mattranspose() const {
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

        void printmat() const {
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    std::cout << atpos(i,j) << " "; 
                }
                std::cout << "\n";
            }  
        };

        void random_number_fill(double min = -1.0, double max = 1.0){
            static std::mt19937 rng(std::random_device{}());
            std::uniform_real_distribution<double> dist(min, max);
            for (int i = 0; i < data.size(); i++)
            {
                data[i] = dist(rng);
            }
        };

        matrix sigmoid() const {
            matrix res(rows,cols);
            for (int i = 0; i < data.size(); i++)
            {
                double x = data[i];
                if (x >= 0) {
                    res.data[i] = 1.0 / (1.0 + std::exp(-x));
                } else {
                    double ex = std::exp(x);
                    res.data[i] = ex / (1.0 + ex);
                }
            }
            return res;
        }; 

        matrix softmax() const {
            matrix res(rows, cols);
            if (data.empty()) return res;
            double max_val = data[0];
            for (int i = 1; i < data.size(); i++) {
                if (data[i] > max_val) max_val = data[i];
            }
            double sum = 0.0;
            for (int i = 0; i < data.size(); i++) {
                res.data[i] = std::exp(data[i] - max_val);
                sum += res.data[i];
            }
            for (int i = 0; i < data.size(); i++) {
                res.data[i] /= sum;
            }
            return res;
        };
        
        matrix element_wise_multiply(const matrix& other) const {
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

        matrix sigmoid_derivative() const {
            matrix res(rows,cols);
            for (int i = 0; i < data.size(); i++)
            {
                res.data[i] = data[i]*(1-data[i]);
            }
            return res;
        }

        matrix apply_learning_rate(double n) const {
            matrix res(rows,cols);
            for (int i = 0; i < data.size(); i++)
            {
                res.data[i] = n*data[i];
            }
            return res;
        }

        static matrix outer_product(const matrix& a, const matrix& b) {
            if (a.cols != 1 || b.cols != 1) throw std::invalid_argument("outer_product requires column vectors");
            matrix res(a.rows, b.rows);
            for (int i = 0; i < a.rows; i++) {
                double val_a = a.data[i];
                int row_res = i * b.rows;
                for (int j = 0; j < b.rows; j++) {
                    res.data[row_res + j] = val_a * b.data[j];
                }
            }
            return res;
        }

        matrix transpose_matmultiply_vec(const matrix& vec) const {
            if (rows != vec.rows || vec.cols != 1) throw std::invalid_argument("dimension mismatch");
            matrix res(cols, 1);
            for (int i = 0; i < cols; i++) {
                double sum = 0.0;
                for (int k = 0; k < rows; k++) {
                    sum += data[k * cols + i] * vec.data[k];
                }
                res.data[i] = sum;
            }
            return res;
        }

        void add_inplace(const matrix& other) {
            for (size_t i = 0; i < data.size(); i++) data[i] += other.data[i];
        }

        void subtract_inplace(const matrix& other) {
            for (size_t i = 0; i < data.size(); i++) data[i] -= other.data[i];
        }

        void sigmoid_inplace() {
            for (size_t i = 0; i < data.size(); i++) {
                double x = data[i];
                if (x >= 0) {
                    data[i] = 1.0 / (1.0 + std::exp(-x));
                } else {
                    double ex = std::exp(x);
                    data[i] = ex / (1.0 + ex);
                }
            }
        }

        void softmax_inplace() {
            if (data.empty()) return;
            double max_val = data[0];
            for (size_t i = 1; i < data.size(); i++) {
                if (data[i] > max_val) max_val = data[i];
            }
            double sum = 0.0;
            for (size_t i = 0; i < data.size(); i++) {
                data[i] = std::exp(data[i] - max_val);
                sum += data[i];
            }
            for (size_t i = 0; i < data.size(); i++) {
                data[i] /= sum;
            }
        }

        void sigmoid_derivative_inplace() {
            for (size_t i = 0; i < data.size(); i++) {
                data[i] = data[i] * (1.0 - data[i]);
            }
        }

        void element_wise_multiply_inplace(const matrix& other) {
            for (size_t i = 0; i < data.size(); i++) data[i] *= other.data[i];
        }

        void apply_learning_rate_inplace(double n) {
            for (size_t i = 0; i < data.size(); i++) data[i] *= n;
        }
};
#endif