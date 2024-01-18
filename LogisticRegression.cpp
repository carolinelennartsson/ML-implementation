#include <iostream>
using namespace std;
#include "armadillo.hpp"
using namespace arma;


class LogReg
{
private:
    mat dataX;
    mat dataY;
    mat dataXtest;
public:
    LogReg(string XfileName, string YfileName, string testfileName);
    ~LogReg();
    double euc_dist(mat v1);
    mat read_data(string fileName);
    void log_reg(mat dataXorig, mat dataY, mat dataXtest);

};

// Constructor
LogReg::LogReg(string XfileName, string YfileName, string testfileName)
{
    dataX = read_data(XfileName);
    dataY = read_data(YfileName);
    dataXtest = read_data(testfileName);
    
    log_reg(dataX, dataY, dataXtest);
}

LogReg::~LogReg()
{}

// method to read data for this purpose
mat LogReg::read_data(string fileName)
{
    ifstream file(fileName);
    mat data;
    data.load(file);
    return data;
}



void LogReg::log_reg(mat dataX, mat dataY, mat dataXtest)
{
    
    int n = dataX.n_rows;
    double e = 2.7182818284;
    
    // initialising w and w1
    mat w;
    w.set_size(dataX.n_cols);
    
    // initialising w
    mat w1;
    w1.set_size(dataX.n_cols);
    
    // defining tolerance and learning rate
    double tol = 10e-7;
    double learn = 0.9;
    
    int count = 0;
    
    do
    {
        // calc w
        w1.zeros(); // set to 0
        
        count += 1;
        
        for (int i = 0; i < n; i++)
        {
            rowvec y = dataY.row(i);
            rowvec x = dataX.row(i);
            
            w1 += ((rowvec)(y * (1.0 / (1 + exp(y * w.t()* x.t()))))).at(0) * x.t();
            
        }
        
        w1 = - (1.0 / n) * w1;
        w = w - learn * w1;

        
    } while (norm(w1) > tol);

    cout << w << endl;
    
    mat labels;
    labels.set_size(dataXtest.n_rows);
    
    
    labels = dataXtest * w;
    ofstream write_file("LogReg.dat");
    
    cout << labels << endl;
    
    for (int i = 0; i < labels.n_rows; i++)
    {
        if (labels[i] < 0)
        {
            write_file << -1 << "\n";
        }
        else {
            write_file << 1 << "\n";
        }
    }
    write_file.close();
    
}



int main(int argc, char* argv[])
{
    LogReg calculations("dataX.dat", "dataY.dat", "dataXtest.dat");
    
    return 0;
}

