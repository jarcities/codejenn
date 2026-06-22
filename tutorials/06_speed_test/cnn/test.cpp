#include <iostream>
#include <chrono>
#include <array>
#include <vector>
#include <string>
#include <iomanip>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>

#include "model_big.hpp"
#include "H5Cpp.h"

static constexpr int NUM_SAMPLES = 10000;
static constexpr int INPUT_DIM = 1000;
static constexpr int INPUT_DIM_1 = 10;
static constexpr int INPUT_DIM_2 = 100;

int main()
{
    // load inputs
    std::vector<std::array<std::array<double, INPUT_DIM_2>, INPUT_DIM_1>> batch;
    batch.reserve(NUM_SAMPLES);

    try
    {
        for (int i = 0; i < NUM_SAMPLES; i++)
        {
            char path[256];
            std::snprintf(path, sizeof(path), "data/data_%04d.h5", i);

            H5::H5File file(path, H5F_ACC_RDONLY);
            H5::DataSet ds = file.openDataSet("x");
            H5::DataSpace space = ds.getSpace();

            int ndims = space.getSimpleExtentNdims();
            if (ndims != 1)
            {
                throw std::runtime_error(std::string("Dataset x is not 1D: ") + path);
            }

            hsize_t dims[1];
            space.getSimpleExtentDims(dims, nullptr);
            if (dims[0] != INPUT_DIM)
            {
                throw std::runtime_error(std::string("Dataset x wrong length: ") + path);
            }

            std::array<double, INPUT_DIM> x{};
            ds.read(x.data(), H5::PredType::NATIVE_DOUBLE);

            std::array<std::array<double, INPUT_DIM_2>, INPUT_DIM_1> input_2d{};
            for(int i = 0; i < INPUT_DIM_1; i++){
                for(int j = 0; j < INPUT_DIM_2; j++){
                    input_2d[i][j] = x[i * INPUT_DIM_2 + j];
                }
            }
            batch.push_back(input_2d);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error loading data: " << e.what() << std::endl;
        return 1;
    }

    // warmup
    {
        std::vector<double> dummy_times;
        volatile auto warm = model(batch[0], dummy_times);
        (void)warm;
    }

    // time inference loop
    std::vector<double> total_layer_times;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_SAMPLES; i++)  
    {
        std::vector<double> layer_times;
        volatile auto result = model(batch[i], layer_times);
        (void)result;

        if (total_layer_times.empty()) total_layer_times.resize(layer_times.size(), 0.0);
        for (size_t j = 0; j < layer_times.size(); ++j) {
            total_layer_times[j] += layer_times[j];
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double total_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::vector<std::string> layer_names = {
        "Prep/Flatten",
        "Layer 1 (Conv2D)",
        "Layer 2 (MaxPooling2D)",
        "Layer 3 (Conv2D)",
        "Layer 4 (Conv2DTranspose)",
        "Layer 6 (Dense)",
        "Layer 7 (Dense)",
        "Softmax"
    };

    std::cout << "\n" << std::left << std::setw(30) << "Layer" 
              << std::setw(20) << "Total Time (s)" 
              << "Percentage (%)" << std::endl;
    std::cout << std::string(65, '-') << std::endl;

    double cumulative_model_time = total_layer_times.back();
    double prev_time = 0.0;

    for (size_t i = 0; i < total_layer_times.size(); ++i) {
        double current_layer_time = total_layer_times[i] - prev_time;
        double percentage = (current_layer_time / cumulative_model_time) * 100.0;
        
        std::cout << std::left << std::setw(30) << layer_names[i]
                  << std::fixed << std::setprecision(6) << std::setw(20) << current_layer_time
                  << std::setprecision(2) << percentage << "%" << std::endl;
        
        prev_time = total_layer_times[i];
    }

    std::cout << std::string(65, '-') << std::endl;
    std::cout << std::left << std::setw(30) << "Total Inference Time" 
              << std::fixed << std::setprecision(6) << (total_us / 1'000'000.0) << " seconds!" << std::endl;

    return 0;
}