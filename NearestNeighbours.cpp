#include <cmath>
#include <iostream>
using namespace std;
#include "armadillo.hpp"
using namespace arma;




class KNN
{
private:
    mat dataX;
    mat dataY;
    mat dataXtest;
public:
    KNN(string XfileName, string YfileName, string testfileName);
    ~KNN();
    mat read_data(string fileName);
    void Train(mat dataX, mat dataXtest, mat dataY, int k = 5);
};

// Constructor
KNN::KNN(string XfileName, string YfileName, string testfileName)
{
    dataX = read_data(XfileName);
    dataY = read_data(YfileName);
    dataXtest = read_data(testfileName);
    Train(dataX, dataXtest, dataY);
    
}

KNN::~KNN()
{}

// method to read data for this purpose
mat KNN::read_data(string fileName)
{
    ifstream file(fileName);
    mat data;
    data.load(file);
    return data;
}


void KNN::Train(mat dataX, mat dataXtest, mat dataY, int k)
{
    mat distanceMat;
    //distanceMat.set_size(dataX.n_rows);
    distanceMat.zeros(dataXtest.n_rows, dataXtest.n_cols);
    
    mat labels;
    labels.set_size(dataXtest.n_rows);
    int label = 0;
    
    int index;
    
    vec distancesum;
    
    for (int i = 0; i < dataXtest.n_rows; i++)
    {
        //mat sub;
        /*
        
        for(int j = 0; j < dataX.n_rows; j++)
        {
            for(int k = 0; k < dataX.n_cols; k++)
            {
                distanceMat(j, k) = pow( dataX(j,k) - dataXtest(i, k), 2);
            }
        } */

        // euclidian distance
        distanceMat = dataX.each_row() - dataXtest.row(i);
        distanceMat = square(distanceMat);
        //distanceMat.transform([](double value){return pow(value, 2);});
        distancesum = sum(distanceMat, 1);
        distancesum.transform([](double value){return sqrt(value);});
        
        
        
        for (int m = 0; m < k; m++)
        {
            //index = index_min(distancesum);
            index = distancesum.index_min();
            label += dataY(index);
            // making sure the index is not reused
            distancesum(index) = 10000;
        }
        
        if (label < 0)
        {
            labels(i) = -1;
        } else {
            labels(i) = 1;
        }
        
        // reset label
        label = 0;
    }
    
    std::ofstream write_file("NN.dat");
    for (int i = 0; i < labels.n_rows; i++)
    {
        write_file << labels[i] << "\n";
    }
    
    write_file.close();
}




int main(int argc, char* argv[])
{
    KNN calculations("dataX.dat", "dataY.dat", "dataXtest.dat");
    // string XfileName, string YfileName, string testfileName
    return 0;
}
