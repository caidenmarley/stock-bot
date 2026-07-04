#include "inputs/parser.h"
#include "model/trainer.h"

#include <filesystem>
#include <functional>
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
    const double close = 100.0 + 0.8 * static_cast<double>(dayIndex);

    PriceData p{};
    p.open = close - 0.2;
    p.high = close + 0.4;
    p.low = close - 0.5;
    p.close = close;
    p.adjClose = close;
    p.volume = static_cast<uint64_t>(1500 + dayIndex * 13);
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

TrainingResult runTinyTrainerOnce(const std::vector<PriceData>& train,
                                  const std::vector<PriceData>& val,
                                  const std::string& resultsPath) {
    Trainer trainer(
        /*numFeatures=*/6,
        /*hiddenSize=*/6,
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
        resultsPath
    );

    return trainer.run(
        /*epochs=*/1,
        /*stoppingToleranceLoss=*/1e-6,
        /*maxEpochsWithNoImprovement=*/1
    );
}

void test_trainer_results_can_be_disabled_and_redirected_safely() {
    const auto full = makeChronologicalSeries(20);
    const std::vector<PriceData> train(full.begin(), full.begin() + 12);
    const std::vector<PriceData> val(full.begin() + 12, full.end());

    const std::filesystem::path sourceTreeResultsPath("tests/results.csv");
    const FileSnapshot beforeSourceTree = snapshotFile(sourceTreeResultsPath);

    // Disabled output should not write any trainer CSV and must not touch tests/results.csv.
    const TrainingResult disabledResult = runTinyTrainerOnce(train, val, "");
    expectTrue(disabledResult.totalEpochs >= 1, "disabled-output run should complete at least one epoch");

    const FileSnapshot afterDisabled = snapshotFile(sourceTreeResultsPath);
    expectTrue(sameSnapshot(beforeSourceTree, afterDisabled),
               "disabled trainer output should not modify source-tree tests/results.csv");

    // Explicit safe path should create output under build-local test_outputs.
    const std::filesystem::path safeOutputPath("build/test_outputs/results_file_hygiene.csv");
    std::error_code ec;
    std::filesystem::remove(safeOutputPath, ec);

    const TrainingResult enabledResult = runTinyTrainerOnce(train, val, safeOutputPath.string());
    expectTrue(enabledResult.totalEpochs >= 1, "enabled-output run should complete at least one epoch");

    expectTrue(std::filesystem::exists(safeOutputPath),
               "explicit safe output path should be created by trainer");
    expectTrue(std::filesystem::file_size(safeOutputPath) > 0,
               "explicit safe output CSV should be non-empty");

    const FileSnapshot afterEnabled = snapshotFile(sourceTreeResultsPath);
    expectTrue(sameSnapshot(beforeSourceTree, afterEnabled),
               "explicit safe output should not modify source-tree tests/results.csv");
}

} // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"trainer results output can be disabled and redirected", test_trainer_results_can_be_disabled_and_redirected_safely},
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

    std::cout << "\nResults-file hygiene tests: " << passed << "/" << tests.size() << " passed\n";
    return passed == tests.size() ? 0 : 1;
}
