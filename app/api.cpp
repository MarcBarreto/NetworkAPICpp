#include "crow.h"
#include<Eigen/Dense>
#include<fstream>
#include<iostream>
#include<vector>

using namespace std;
using namespace Eigen;

VectorXd load_model(const string &filename, int size)
{
    VectorXd weights(size);

    ifstream file(filename, ios::in | ios::binary);

    if(!file)
    {
        cerr << "Error to open model file!" << endl;
        exit(1);
    }

    for(int i = 0; i < size; ++i)
    {
        file.read(reinterpret_cast<char *>(&weights[i]), sizeof(weights[i]));
    }

    file.close();
    return weights;
}

double predict(const VectorXd &weights, const VectorXd &input)
{
    return weights.dot(input);
}

int main()
{
    string model_file = "model_cpp.bin";

    VectorXd weights = load_model(model_file, 3);

    crow::SimpleApp app;

    //root route
    CROW_ROUTE(app, "/")
    ([]()
    {
        return "Running Network API";
    });

    // Predict route (endpoint)
    CROW_ROUTE(app, "/predict")
    .methods("POST"_method)
    ([&weights](const crow::request &req) 
    {
        auto json_data = crow::json::load(req.body);

        if(!json_data)
        {
            return crow::response(400, "Invalid Input Data");
        }

        double feature1 = json_data["feature1"].d();
        double feature2 = json_data["feature2"].d();

        // Input Data (features + bias)
        VectorXd input(3);

        input << feature1, feature2, 1.0;

        double prediction = predict(weights, input);

        // Return request
        crow::json::wvalue response;
        response["prediction"] = prediction;
        return crow::response(200, response);
    });

    // Starts server
    app.port(5001).multithreaded().run();
}