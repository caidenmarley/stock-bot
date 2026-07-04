#include "inputs/parser.h"
#include "model/trainer.h"

#include <cmath>
#include <filesystem>
#include <fstream>
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
    const double trend = 100.0 + 0.85 * static_cast<double>(dayIndex);
    const double wave = 0.06 * std::sin(0.33 * static_cast<double>(dayIndex));
    const double close = trend + wave;

    PriceData p{};
    p.open = close - 0.20;
    p.high = close + 0.35;
    p.low = close - 0.40;
    p.close = close;
    p.adjClose = close;
    p.volume = static_cast<uint64_t>(1300 + dayIndex * 9);
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

void assertResultLooksValid(const TrainingResult& r) {
    expectTrue(r.totalEpochs == 1, "tiny Trainer run should report one epoch");
    expectTrue(r.epochOfBestValLoss >= 1 && r.epochOfBestValLoss <= r.totalEpochs,
               "best-validation epoch should be within [1, totalEpochs]");
    expectTrue(std::isfinite(r.bestValLoss), "best validation loss should be finite");
    expectTrue(r.bestValLoss >= 0.0, "best validation loss should be non-negative");
    expectTrue(r.bestValLoss < 1e6, "best validation loss should improve from initial sentinel");
}

void test_tiny_trainer_run_with_output_disabled_keeps_source_tree_clean() {
    const auto full = makeChronologicalSeries(26);
    const std::vector<PriceData> train(full.begin(), full.begin() + 16);
    const std::vector<PriceData> val(full.begin() + 16, full.end());

    const std::filesystem::path testsResultsPath("tests/results.csv");
    const std::filesystem::path dataResultsPath("data/results.csv");
    const FileSnapshot beforeTestsResults = snapshotFile(testsResultsPath);
    const FileSnapshot beforeDataResults = snapshotFile(dataResultsPath);

    const TrainingResult result = runTinyTrainerOnce(train, val, "");
    assertResultLooksValid(result);

    const FileSnapshot afterTestsResults = snapshotFile(testsResultsPath);
    const FileSnapshot afterDataResults = snapshotFile(dataResultsPath);

    expectTrue(sameSnapshot(beforeTestsResults, afterTestsResults),
               "disabled Trainer output should not modify tests/results.csv");
    expectTrue(sameSnapshot(beforeDataResults, afterDataResults),
               "Trainer run should not modify data/results.csv");
}

void test_tiny_trainer_run_can_write_to_safe_build_local_path() {
    const auto full = makeChronologicalSeries(26);
    const std::vector<PriceData> train(full.begin(), full.begin() + 16);
    const std::vector<PriceData> val(full.begin() + 16, full.end());

    const std::filesystem::path testsResultsPath("tests/results.csv");
    const std::filesystem::path dataResultsPath("data/results.csv");
    const FileSnapshot beforeTestsResults = snapshotFile(testsResultsPath);
    const FileSnapshot beforeDataResults = snapshotFile(dataResultsPath);

    const std::filesystem::path safeOutputPath("build/test_outputs/trainer_behavior_results.csv");
    std::error_code ec;
    std::filesystem::remove(safeOutputPath, ec);

    const TrainingResult result = runTinyTrainerOnce(train, val, safeOutputPath.string());
    assertResultLooksValid(result);

    expectTrue(std::filesystem::exists(safeOutputPath),
               "safe build-local output path should be created");
    expectTrue(std::filesystem::file_size(safeOutputPath) > 0,
               "safe build-local output file should be non-empty");

    std::ifstream in(safeOutputPath);
    expectTrue(static_cast<bool>(in), "safe output CSV should be readable");

    std::string header;
    std::string firstRow;
    expectTrue(static_cast<bool>(std::getline(in, header)), "safe output CSV should contain header");
    expectTrue(static_cast<bool>(std::getline(in, firstRow)), "safe output CSV should contain one epoch row");
    expectTrue(header == "epoch,train_loss,val_loss,mae,rmse,da,ada_lr,dense_lr",
               "safe output CSV header should match Trainer schema");
    expectTrue(!firstRow.empty(), "safe output CSV first data row should be non-empty");

    const FileSnapshot afterTestsResults = snapshotFile(testsResultsPath);
    const FileSnapshot afterDataResults = snapshotFile(dataResultsPath);
    expectTrue(sameSnapshot(beforeTestsResults, afterTestsResults),
               "safe-output Trainer run should not modify tests/results.csv");
    expectTrue(sameSnapshot(beforeDataResults, afterDataResults),
               "safe-output Trainer run should not modify data/results.csv");
}

} // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"tiny Trainer run succeeds with output disabled and source tree unchanged", test_tiny_trainer_run_with_output_disabled_keeps_source_tree_clean},
        {"tiny Trainer run writes to safe build-local path", test_tiny_trainer_run_can_write_to_safe_build_local_path},
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

    std::cout << "\nTrainer behavior tests: " << passed << "/" << tests.size() << " passed\n";
    return passed == tests.size() ? 0 : 1;
}
