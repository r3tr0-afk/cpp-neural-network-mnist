#include"../include/matrix.h"
#include<iostream>
#include<vector>

using namespace std;
int main(){
    matrix A(2,3);

    A.printmat();
    cout << "\n";
    cout << "rows :" << A.numrows() << "\ncols : " << A.numcols() << endl; 
    
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            A.atpos_modifiable(i,j) = i+j;
        }
        
    }
    A.printmat();
    cout << "\n";

    matrix transpose = A.mattranspose();
    transpose.printmat();
    cout << "\n";

    matrix add = A.matadd(A);
    add.printmat();
    cout << "\n";

    matrix B(3,2);
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            B.atpos_modifiable(i,j) = i+j;
        }
        
    }

    //matrix subtract = A.matsubtract(B);
    //subtract.printmat();
    //cout << "\n";

    matrix multiply = A.matmultiply(B);
    multiply.printmat();
    cout << "\n";

    matrix sub = A.matsubtract(A);
    sub.printmat();
    cout << "\n";

    matrix C(4,4);
    matrix D(5,5);
    C.random_number_fill();
    D.random_number_fill();
    C.printmat();
    cout << "\n";
    D.printmat();

    return 0;
}