#include "parser/parser.h"
#include "parser/shape_inputs.h"
#include <iostream>

int main() {
    try{
        CSVLoader loader("data/AAAU.csv");
        const std::vector<PriceData>& data = loader.getData();
        std::cout << "Loaded " << data.size() << " rows from csv" << std::endl; 
        StockData stockData(data, 40, 20);

        size_t batchIndex = 0;
        while (stockData.hasAnotherBatch()) {
            auto [inputBatch, targetBatch] = stockData.nextBatch();
            
            // inputBatch: Eigen::Tensor<double,3> → dimensions (B, T, F)
            std::cout << "Batch " << batchIndex++
                      << ": input shape = ("
                      << inputBatch.dimension(0) << ", "
                      << inputBatch.dimension(1) << ", "
                      << inputBatch.dimension(2) << ")"
                      << ", target length = "
                      << targetBatch.size()
                      << "\n";
        }

    }catch(std::exception& ex){
        std::cerr << ex.what() << "\n";
        return 2;
    }
}
