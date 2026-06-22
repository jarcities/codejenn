#include <iostream>
#include <chrono>
#include <array>
#include <vector>
#include <string>
#include <iomanip>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>

#include "model.hpp"
#include "H5Cpp.h"

static constexpr int NUM_SAMPLES = 10000;
static constexpr int INPUT_DIM = 1000;

int main()
{
    // load inputs
    std::vector<std::array<double, INPUT_DIM>> batch;
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
            batch.push_back(x);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error loading data: " << e.what() << std::endl;
        return 1;
    }

    // warmup
    {
        volatile auto warm = model(batch[0]);
        (void)warm;
    }

    // time inference loop
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_SAMPLES; i++)
    {
        auto result = model(batch[i]);
        (void)result;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << std::fixed << std::setprecision(6)
              << (us / 1'000'000.0) << " seconds!" << std::endl;

    return 0;
}