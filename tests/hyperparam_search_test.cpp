#include "inputs/parser.h"
#include "search/hyperparam_search.h"

#include <filesystem>
#include <functional>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void expectTrue(bool cond, const std::string& msg) {
    if (!cond) {
        throw std::runtime_error(msg);
    }
}

PriceData makePrice(int dayIndex) {
    const double trend = 50.0 + 0.6 * static_cast<double>(dayIndex);
    const double wave = 0.04 * std::sin(0.25 * static_cast<double>(dayIndex));
    const double close = trend + wave;

    PriceData p{};
    p.open = close - 0.10;
    p.high = close + 0.20;
    p.low = close - 0.25;
    p.close = close;
    p.adjClose = close;
    p.volume = static_cast<uint64_t>(1000 + dayIndex * 5);
    p.date = "";
    return p;
}

std::vector<PriceData> makeChronologicalSeries(int count) {
    std::vector<PriceData> out;
    out.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        out.push_back(makePrice(i));
    }
    return out;
}

struct FileSnapshot {
    bool exists{false};
    std::filesystem::file_time_type writeTime{};
    std::uintmax_t size{0};
};

FileSnapshot snapshotFile(const std::filesystem::path& p) {
    FileSnapshot s;
    std::error_code ec;

    s.exists = std::filesystem::exists(p, ec);
    if (!ec && s.exists) {
        s.writeTime = std::filesystem::last_write_time(p, ec);
        if (!ec) {
            s.size = std::filesystem::file_size(p, ec);
        }
    }

    return s;
}

bool sameSnapshot(const FileSnapshot& a, const FileSnapshot& b) {
    if (a.exists != b.exists) {
        return false;
    }
    if (!a.exists) {
        return true;
    }
    return a.writeTime == b.writeTime && a.size == b.size;
}

void test_grid_search_writes_to_explicit_safe_path() {
    const auto full = makeChronologicalSeries(22);
    const std::vector<PriceData> train(full.begin(), full.begin() + 14);
    const std::vector<PriceData> val(full.begin() + 14, full.end());

    const std::filesystem::path testsResultsPath("tests/results.csv");
    const std::filesystem::path dataResultsPath("data/results.csv");
    const FileSnapshot beforeTestsResults = snapshotFile(testsResultsPath);
    const FileSnapshot beforeDataResults = snapshotFile(dataResultsPath);

    const std::filesystem::path safeOutputPath("build/test_outputs/hyperparam_search_results.csv");
    std::error_code ec;
    std::filesystem::remove(safeOutputPath, ec);

    const std::vector<HyperParam> params = {
        {"hiddenSize", ParamType::INTEGER, {4.0}},
    };

    gridSearch(
        params,
        /*numFeatures=*/6,
        /*hiddenSize=*/4,
        /*sequenceLength=*/3,
        /*batchSize=*/2,
        /*learningRate=*/1e-3,
        /*delta=*/1.0,
        /*windowSize=*/4,
        /*maxNorm=*/1.0,
        /*decayFactor=*/0.5,
        /*minLR=*/1e-6,
        /*lrDecayMaxTries=*/1,
        train,
        val,
        /*epochs=*/1,
        /*stoppingToleranceLoss=*/1e-6,
        /*maxEpochsWithNoImprovement=*/1,
        safeOutputPath.string()
    );

    expectTrue(std::filesystem::exists(safeOutputPath),
               "gridSearch should create configured build-local output file");
    expectTrue(std::filesystem::file_size(safeOutputPath) > 0,
               "gridSearch configured output file should be non-empty");

    std::ifstream in(safeOutputPath);
    expectTrue(static_cast<bool>(in), "configured output file should be readable");
    std::string header;
    expectTrue(static_cast<bool>(std::getline(in, header)), "configured output CSV should have header");
    expectTrue(header == "hiddenSize,sequenceLength,batchSize,learningRate,delta,windowSize,bestValLoss,epochOfBestValLoss,totalEpochs",
               "gridSearch output header should match expected schema");

    const FileSnapshot afterTestsResults = snapshotFile(testsResultsPath);
    const FileSnapshot afterDataResults = snapshotFile(dataResultsPath);
    expectTrue(sameSnapshot(beforeTestsResults, afterTestsResults),
               "gridSearch should not modify tests/results.csv when safe output path is provided");
    expectTrue(sameSnapshot(beforeDataResults, afterDataResults),
               "gridSearch should not modify data/results.csv when safe output path is provided");
}

void test_random_search_reports_not_implemented() {
    bool threw = false;
    try {
        const std::vector<HyperParam> params = {
            {"hiddenSize", ParamType::INTEGER, {4.0}},
        };
        const auto full = makeChronologicalSeries(20);
        const std::vector<PriceData> train(full.begin(), full.begin() + 12);
        const std::vector<PriceData> val(full.begin() + 12, full.end());

        randomSearch(
            params,
            /*numFeatures=*/6,
            /*hiddenSize=*/4,
            /*sequenceLength=*/3,
            /*batchSize=*/2,
            /*learningRate=*/1e-3,
            /*delta=*/1.0,
            /*windowSize=*/4,
            /*maxNorm=*/1.0,
            /*decayFactor=*/0.5,
            /*minLR=*/1e-6,
            /*lrDecayMaxTries=*/1,
            train,
            val,
            /*epochs=*/1,
            /*stoppingToleranceLoss=*/1e-6,
            /*maxEpochsWithNoImprovement=*/1,
            /*nTrials=*/1
        );
    } catch (const std::logic_error& ex) {
        threw = std::string(ex.what()).find("not implemented") != std::string::npos;
    }

    expectTrue(threw, "randomSearch should throw a clear not-implemented logic_error");
}

} // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"gridSearch writes to explicit safe output path", test_grid_search_writes_to_explicit_safe_path},
        {"randomSearch reports not implemented", test_random_search_reports_not_implemented},
    };

    std::size_t passed = 0;
    for (const auto& [name, fn] : tests) {
        try {
            fn();
            ++passed;
            std::cout << "[PASS] " << name << "\n";
        } catch (const std::exception& ex) {
            std::cout << "[FAIL] " << name << " -> " << ex.what() << "\n";
        }
    }

    std::cout << "\nHyperparameter search tests: " << passed << "/" << tests.size() << " passed\n";
    return passed == tests.size() ? 0 : 1;
}
