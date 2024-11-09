#include<iostream>
#include<fstream>
#include<Eigen/Dense>
#include<cmath>

using namespace Eigen;
using namespace std;

// MatrixXd is a data type of Eigen library to lead with matrix and array. 
MatrixXd sigmoid(const MatrixXd &Z)
{
    return 1.0 / (1.0 + (-Z.array()).exp());
}

MatrixXd sigmoide_derivative(const MatrixXd &Z)
{
    return Z.array() * (1.0 - Z.array());
}

void save_model(const MatrixXd &weights1, const MatrixXd &weights2, const string &filename)
{
    ofstream file(filename, ios::out | ios::binary);

    if (!file)
    {
        cerr << "Error to open file to salve model.";
        return;
    }

    for(int i = 0; i < weights1.size(); ++i)
    {
        file.write(reinterpret_cast<const char *>(&weights1(i)), sizeof(weights1(i)));
    }

    for(int i = 0; i < weights2.size(); ++i)
    {
        file.write(reinterpret_cast<const char *>(&weights2(i)), sizeof(weights2(i)));
    }

    file.close();
    cout << "Model save in: " << filename << endl;
}

void fit(MatrixXd &X, VectorXd &y, MatrixXd &W1, MatrixXd &W2, double learning_rate, int iterations)
{
    int m = X.rows();

    for(int i = 0; i < iterations; ++i)
    {
        // Forward
        MatrixXd Z1 = X * W1;
        MatrixXd A1 = sigmoid(Z1);
        MatrixXd Z2 = A1 * W2;
        MatrixXd A2 = Z2;

        MatrixXd error = A2 - y.replicate(1, A2.cols());

        // Backpropagation
        MatrixXd dZ2 = error;
        MatrixXd dW2 = A1.transpose() * dZ2 / m;
        MatrixXd dZ1 = (dZ2 * W2.transpose()).array() * sigmoide_derivative(A1).array();
        MatrixXd dW1 = X.transpose() * dZ1 / m;

        W1 -= learning_rate * dW1;
        W2 -= learning_rate * dW2;

        if(i % 100 == 0)
        {
            double cost = error.array().square().mean();
            cout << "Epoch: " << i << ", Error: " << cost << endl;
        }
    }
}

int main()
{
    // Example of Data Input. (4 samples, 2 features + 1 bias)
    MatrixXd X(4, 3);
    X << 1, 1, 1,
         1, 2, 1,
         2, 2, 1,
         2, 3, 1;
    
    VectorXd y(4);
    y << 6, 8, 9, 11;

    MatrixXd W1 = MatrixXd::Random(3, 3);
    MatrixXd W2 = MatrixXd::Random(3, 1);

    double learning_rate = 0.01;
    int iterations = 1000;
    fit(X, y, W1, W2, learning_rate, iterations);

    string filename = "model_cpp.bin";
    save_model(W1, W2, filename);

    return 0;
}