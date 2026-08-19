#include "inputs/parser.h"
#include "inputs/stock_data.h"
#include "model/lstm.h"
#include "model/trainer.h"
#include "search/hyperparam_search.h"
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct FoldRun {
    std::string ticker;
    std::string csvPath;
    int featureCount;
    int foldNumber;
    TrainingResult result;
};

struct RunSettings {
    int hiddenSize;
    int sequenceLength;
    int batchSize;
    double learningRate;
    double delta;
    size_t windowSize;
    double maxNorm;
    double decayFactor;
    double minLR;
    int lrDecayMaxTries;
    int epochs;
    double stoppingToleranceLoss;
    int maxEpochsWithNoImprovement;
    std::string trainerResultsPath;
    std::optional<uint32_t> denseInitSeed;
    std::optional<uint32_t> lstmInitSeed;
};

std::vector<int> defaultCutPoints(int rawSize, int sequenceLength);
void printFeatureSummary(int featureCount, const std::vector<FoldRun>& foldRuns);

static TrainingResult runFold(
    const std::vector<PriceData>& rawData,
    int splitStart,
    int numFeatures,
    int hiddenSize,
    int sequenceLength,
    int batchSize,
    double learningRate,
    double delta,
    size_t windowSize,
    double maxNorm,
    double decayFactor,
    double minLR,
    int lrDecayMaxTries,
    int epochs,
    double stoppingToleranceLoss,
    int maxEpochsWithNoImprovement,
    const std::string& trainerResultsPath,
    std::optional<uint32_t> denseInitSeed,
    std::optional<uint32_t> lstmInitSeed
) {
    if (lstmInitSeed.has_value()) {
        LSTMCell::setGlobalInitSeed(*lstmInitSeed);
    }

    std::vector<PriceData> trainingData(rawData.begin(), rawData.begin() + splitStart);
    std::vector<PriceData> validationData(rawData.begin() + splitStart, rawData.end());

    Trainer trainer(
        numFeatures, hiddenSize, sequenceLength, batchSize, learningRate, delta,
        windowSize, maxNorm, decayFactor, minLR, lrDecayMaxTries,
        trainingData, validationData,
        trainerResultsPath,
        denseInitSeed
    );

    return trainer.run(epochs, stoppingToleranceLoss, maxEpochsWithNoImprovement);
}

void writeAblationReportCsv(const std::string& outputPath, const std::vector<FoldRun>& runs) {
    const std::filesystem::path reportPath(outputPath);
    if (reportPath.has_parent_path()) {
        std::error_code mkErr;
        std::filesystem::create_directories(reportPath.parent_path(), mkErr);
    }

    std::ofstream csv(outputPath);
    if (!csv) {
        throw std::runtime_error("failed to open ablation report file: " + outputPath);
    }

        csv << "ticker,csv_path,feature_count,fold,best_val_loss,epoch_of_best_val_loss,total_epochs,model_sharpe_net,avg_turnover,"
            "benchmark_strategy,benchmark_sharpe_net,benchmark_avg_turnover,benchmark_cumulative_net_return,benchmark_num_observations\n";

    csv << std::fixed << std::setprecision(6);
    for (const FoldRun& run : runs) {
        if (run.result.finalValBenchmarkRows.empty()) {
            csv << run.ticker << ","
                << run.csvPath << ","
                << run.featureCount << ","
                << run.foldNumber << ","
                << run.result.bestValLoss << ","
                << run.result.epochOfBestValLoss << ","
                << run.result.totalEpochs << ","
                << run.result.finalValSharpeNet << ","
                << run.result.finalValAvgTurnover << ","
                << "model,"
                << run.result.finalValSharpeNet << ","
                << run.result.finalValAvgTurnover << ","
                << 0.0 << ","
                << 0 << "\n";
            continue;
        }

        for (const auto& strategyRow : run.result.finalValBenchmarkRows) {
            csv << run.ticker << ","
                << run.csvPath << ","
                << run.featureCount << ","
                << run.foldNumber << ","
                << run.result.bestValLoss << ","
                << run.result.epochOfBestValLoss << ","
                << run.result.totalEpochs << ","
                << run.result.finalValSharpeNet << ","
                << run.result.finalValAvgTurnover << ","
                << strategyRow.name << ","
                << strategyRow.sharpeNet << ","
                << strategyRow.avgTurnover << ","
                << strategyRow.cumulativeNetReturn << ","
                << strategyRow.numObservations << "\n";
        }
    }
}

bool hasCsvExtension(const std::filesystem::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return ext == ".csv";
}

std::vector<std::filesystem::path> discoverCsvFiles(const std::filesystem::path& dataDirPath) {
    std::vector<std::filesystem::path> csvFiles;
    for (const auto& entry : std::filesystem::directory_iterator(dataDirPath)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        if (hasCsvExtension(entry.path())) {
            csvFiles.push_back(entry.path());
        }
    }

    std::sort(csvFiles.begin(), csvFiles.end(), [](const std::filesystem::path& a, const std::filesystem::path& b) {
        const std::string aName = a.filename().string();
        const std::string bName = b.filename().string();
        if (aName == bName) {
            return a.string() < b.string();
        }
        return aName < bName;
    });

    return csvFiles;
}

std::string tickerFromPath(const std::filesystem::path& csvPath) {
    return csvPath.stem().string();
}

std::vector<FoldRun> runForFeatureCountOnDataset(
    const std::vector<PriceData>& rawData,
    const std::string& ticker,
    const std::string& csvPath,
    int featureCount,
    const RunSettings& settings
) {
    const std::vector<int> cutPoints = defaultCutPoints(static_cast<int>(rawData.size()), settings.sequenceLength);

    std::vector<FoldRun> foldRuns;
    foldRuns.reserve(cutPoints.size());

    for (size_t i = 0; i < cutPoints.size(); ++i) {
        const int splitStart = cutPoints[i];
        const int foldNumber = static_cast<int>(i) + 1;

        std::cout << "[DATA " << ticker << "][FOLD " << foldNumber << "][features=" << featureCount << "] train=[0,"
                  << splitStart << "] val=[" << splitStart << "," << rawData.size() << "]"
                  << std::endl;

        TrainingResult result = runFold(
            rawData,
            splitStart,
            featureCount,
            settings.hiddenSize,
            settings.sequenceLength,
            settings.batchSize,
            settings.learningRate,
            settings.delta,
            settings.windowSize,
            settings.maxNorm,
            settings.decayFactor,
            settings.minLR,
            settings.lrDecayMaxTries,
            settings.epochs,
            settings.stoppingToleranceLoss,
            settings.maxEpochsWithNoImprovement,
            settings.trainerResultsPath,
            settings.denseInitSeed,
            settings.lstmInitSeed
        );

        foldRuns.push_back({ticker, csvPath, featureCount, foldNumber, std::move(result)});
    }

    printFeatureSummary(featureCount, foldRuns);
    return foldRuns;
}

std::vector<int> defaultCutPoints(int rawSize, int sequenceLength) {
    const int maxStartSequence = rawSize - sequenceLength;
    return {
        static_cast<int>(0.6 * maxStartSequence),
        static_cast<int>(0.7 * maxStartSequence),
        static_cast<int>(0.8 * maxStartSequence),
    };
}

void printFeatureSummary(int featureCount, const std::vector<FoldRun>& foldRuns) {
    double avg = 0.0;
    double best = 1e9;

    for (const FoldRun& foldRun : foldRuns) {
        avg += foldRun.result.bestValLoss;
        best = std::min(best, foldRun.result.bestValLoss);
    }
    avg /= foldRuns.empty() ? 1.0 : static_cast<double>(foldRuns.size());

    std::cout << "[SUMMARY][features=" << featureCount << "] avg best val loss = "
              << std::fixed << std::setprecision(6) << avg
              << " | best fold loss = " << best
              << std::endl;
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        int numFeatures = stock_features::kFeatureCount;
        int hiddenSize = 64;
        int sequenceLength = 15;
        int batchSize = 10;
        double learningRate = 1e-3;
        size_t windowSize = 256;
        double maxNorm = 1.0;
        double decayFactor = 0.5;
        double minLR = 1e-6;
        int lrDecayMaxTries = 3;
        double delta = 1.0;
        int epochs = 10;
        double stoppingToleranceLoss = 1e-6;
        int maxEpochsWithNoImprovement = 3;

        int seed = 0;
        bool givenSeed = false;

        bool enableTrainerResults = true;
        std::string trainerResultsPath = "build/results/trainer_results.csv";

        bool runFeatureAblation = false;
        std::string ablationReportPath;

        std::string dataPath;
        std::string dataDir;
        std::optional<int> maxFiles;
        std::optional<int> fileOffset;

        auto requireValue = [&](int idx, const std::string& opt) {
            if (idx + 1 >= argc) {
                throw std::invalid_argument(opt + " requires a value");
            }
        };

        auto parseIntArg = [&](const std::string& raw, const std::string& opt) {
            try {
                size_t pos = 0;
                int value = std::stoi(raw, &pos);
                if (pos != raw.size()) {
                    throw std::invalid_argument("");
                }
                return value;
            } catch (...) {
                throw std::invalid_argument("invalid integer for " + opt + ": " + raw);
            }
        };

        auto parseDoubleArg = [&](const std::string& raw, const std::string& opt) {
            try {
                size_t pos = 0;
                double value = std::stod(raw, &pos);
                if (pos != raw.size()) {
                    throw std::invalid_argument("");
                }
                return value;
            } catch (...) {
                throw std::invalid_argument("invalid number for " + opt + ": " + raw);
            }
        };

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--epochs") {
                requireValue(i, arg);
                epochs = parseIntArg(argv[++i], arg);
                if (epochs <= 0) {
                    throw std::invalid_argument("--epochs must be > 0");
                }
            } else if (arg == "--early-stop-eps") {
                requireValue(i, arg);
                stoppingToleranceLoss = parseDoubleArg(argv[++i], arg);
            } else if (arg == "--early-stop-patience") {
                requireValue(i, arg);
                maxEpochsWithNoImprovement = parseIntArg(argv[++i], arg);
                if (maxEpochsWithNoImprovement <= 0) {
                    throw std::invalid_argument("--early-stop-patience must be > 0");
                }
            } else if (arg == "--seed") {
                requireValue(i, arg);
                seed = parseIntArg(argv[++i], arg);
                if (seed < 0) {
                    throw std::invalid_argument("--seed must be >= 0");
                }
                givenSeed = true;
            } else if (arg == "--results-file") {
                requireValue(i, arg);
                trainerResultsPath = argv[++i];
                if (trainerResultsPath.empty()) {
                    throw std::invalid_argument("--results-file requires a non-empty path");
                }
                enableTrainerResults = true;
            } else if (arg == "--no-results") {
                enableTrainerResults = false;
            } else if (arg == "--feature-count") {
                requireValue(i, arg);
                numFeatures = parseIntArg(argv[++i], arg);
                if (numFeatures <= 0) {
                    throw std::invalid_argument("--feature-count must be > 0");
                }
                if (numFeatures > stock_features::kFeatureCount) {
                    throw std::invalid_argument("--feature-count must be <= " + std::to_string(stock_features::kFeatureCount));
                }
                if (numFeatures != 6 && numFeatures != stock_features::kFeatureCount) {
                    throw std::invalid_argument("--feature-count supports 6 or " + std::to_string(stock_features::kFeatureCount) + " for this ablation sprint");
                }
            } else if (arg == "--feature-ablation") {
                runFeatureAblation = true;
            } else if (arg == "--ablation-report") {
                requireValue(i, arg);
                ablationReportPath = argv[++i];
                if (ablationReportPath.empty()) {
                    throw std::invalid_argument("--ablation-report requires a non-empty path");
                }
                runFeatureAblation = true;
            } else if (arg == "--data-path") {
                requireValue(i, arg);
                dataPath = argv[++i];
                if (dataPath.empty()) {
                    throw std::invalid_argument("--data-path requires a non-empty path");
                }
            } else if (arg == "--data-dir") {
                requireValue(i, arg);
                dataDir = argv[++i];
                if (dataDir.empty()) {
                    throw std::invalid_argument("--data-dir requires a non-empty path");
                }
            } else if (arg == "--max-files") {
                requireValue(i, arg);
                const int parsedMaxFiles = parseIntArg(argv[++i], arg);
                if (parsedMaxFiles <= 0) {
                    throw std::invalid_argument("--max-files must be > 0");
                }
                maxFiles = parsedMaxFiles;
            } else if (arg == "--file-offset") {
                requireValue(i, arg);
                const int parsedFileOffset = parseIntArg(argv[++i], arg);
                if (parsedFileOffset < 0) {
                    throw std::invalid_argument("--file-offset must be >= 0");
                }
                fileOffset = parsedFileOffset;
            } else if (arg == "--test") {
            } else if (arg == "--help") {
                std::cout
                    << "Usage: " << argv[0] << " [options]\n"
                    << "Options:\n"
                    << "  --epochs N                     Train up to N epochs (default 10)\n"
                    << "  --early-stop-eps X             Early-stop tol. (default 1e-3)\n"
                    << "  --early-stop-patience P        Early-stop patience (default 3)\n"
                    << "  --seed S                       Set RNG seed\n"
                    << "  --feature-count N              Use N features (supports 6 or " << stock_features::kFeatureCount << ")\n"
                    << "  --feature-ablation             Run 6-feature vs " << stock_features::kFeatureCount << "-feature comparison\n"
                    << "  --ablation-report PATH         Write ablation CSV report to PATH\n"
                    << "  --data-path PATH               Run training/eval on one CSV file\n"
                    << "  --data-dir DIR                 Run training/eval across all .csv files in DIR\n"
                    << "  --max-files N                  Process at most N sorted CSV files (data-dir mode only)\n"
                    << "  --file-offset N                Skip first N sorted CSV files (data-dir mode only)\n"
                    << "  --results-file PATH            Write trainer metrics CSV to PATH\n"
                    << "  --no-results                   Disable trainer CSV output\n"
                    << "  --test                         Run hyperparameter search\n";
                return 0;
            } else {
                throw std::invalid_argument("unknown option: " + arg + " (use --help for usage)");
            }
        }

        if (runFeatureAblation && numFeatures != stock_features::kFeatureCount) {
            throw std::invalid_argument("--feature-count cannot be combined with --feature-ablation; ablation mode runs both 6 and "
                                        + std::to_string(stock_features::kFeatureCount));
        }

        if (!dataPath.empty() && !dataDir.empty()) {
            throw std::invalid_argument("--data-path and --data-dir are mutually exclusive");
        }

        if ((maxFiles.has_value() || fileOffset.has_value()) && dataDir.empty()) {
            throw std::invalid_argument("--max-files and --file-offset require --data-dir");
        }

        if (!dataPath.empty()) {
            const std::filesystem::path inputPath(dataPath);
            if (!std::filesystem::exists(inputPath)) {
                throw std::invalid_argument("--data-path does not exist: " + dataPath);
            }
            if (!std::filesystem::is_regular_file(inputPath)) {
                throw std::invalid_argument("--data-path must be a file: " + dataPath);
            }
        }

        if (!dataDir.empty()) {
            const std::filesystem::path inputDir(dataDir);
            if (!std::filesystem::exists(inputDir)) {
                throw std::invalid_argument("--data-dir does not exist: " + dataDir);
            }
            if (!std::filesystem::is_directory(inputDir)) {
                throw std::invalid_argument("--data-dir must be a directory: " + dataDir);
            }
        }

        const std::optional<uint32_t> denseInitSeed =
            givenSeed ? std::optional<uint32_t>(static_cast<uint32_t>(seed)) : std::nullopt;
        const std::optional<uint32_t> lstmInitSeed =
            givenSeed ? std::optional<uint32_t>(static_cast<uint32_t>(seed)) : std::nullopt;

        const std::string activeTrainerResultsPath = enableTrainerResults ? trainerResultsPath : "";
        const RunSettings settings{
            hiddenSize,
            sequenceLength,
            batchSize,
            learningRate,
            delta,
            windowSize,
            maxNorm,
            decayFactor,
            minLR,
            lrDecayMaxTries,
            epochs,
            stoppingToleranceLoss,
            maxEpochsWithNoImprovement,
            activeTrainerResultsPath,
            denseInitSeed,
            lstmInitSeed,
        };

        std::vector<std::filesystem::path> datasets;
        if (!dataDir.empty()) {
            const std::vector<std::filesystem::path> discovered = discoverCsvFiles(std::filesystem::path(dataDir));
            datasets = discovered;
            if (datasets.empty()) {
                throw std::runtime_error("No CSV files found in data directory: " + dataDir);
            }

            const size_t selectedOffset = static_cast<size_t>(fileOffset.value_or(0));
            if (selectedOffset >= datasets.size()) {
                throw std::runtime_error(
                    "No CSV files selected after applying --file-offset=" + std::to_string(selectedOffset)
                    + " to " + std::to_string(datasets.size()) + " discovered file(s)"
                );
            }

            const size_t selectedMaxFiles = maxFiles.has_value()
                ? static_cast<size_t>(*maxFiles)
                : (datasets.size() - selectedOffset);
            const size_t selectedCount = std::min(selectedMaxFiles, datasets.size() - selectedOffset);

            datasets = std::vector<std::filesystem::path>(
                datasets.begin() + static_cast<std::ptrdiff_t>(selectedOffset),
                datasets.begin() + static_cast<std::ptrdiff_t>(selectedOffset + selectedCount)
            );

            if (datasets.empty()) {
                throw std::runtime_error("No CSV files selected after applying --file-offset/--max-files");
            }

            std::cout << "[DATA] discovered " << discovered.size() << " csv files in " << dataDir
                      << " | selected " << datasets.size() << " (offset=" << selectedOffset
                      << ", max-files=" << (maxFiles.has_value() ? std::to_string(*maxFiles) : std::string("all"))
                      << ")" << std::endl;
        } else if (!dataPath.empty()) {
            datasets.push_back(std::filesystem::path(dataPath));
        } else {
            datasets.push_back(std::filesystem::path("data/AAAU.csv"));
        }

        std::vector<FoldRun> allRuns;

        const bool robustDirectoryMode = !dataDir.empty();
        for (const std::filesystem::path& datasetPath : datasets) {
            const std::string csvPath = datasetPath.string();
            const std::string ticker = tickerFromPath(datasetPath);

            try {
                CSVLoader loader(csvPath);
                const std::vector<PriceData>& rawData = loader.getData();
                std::cout << "[DATA " << ticker << "] parsed " << rawData.size() << " rows from CSV" << std::endl;

                if (!runFeatureAblation) {
                    std::vector<FoldRun> runs = runForFeatureCountOnDataset(rawData, ticker, csvPath, numFeatures, settings);
                    allRuns.insert(allRuns.end(), runs.begin(), runs.end());
                    continue;
                }

                const int baselineFeatures = 6;
                const int engineeredFeatures = stock_features::kFeatureCount;

                std::vector<FoldRun> baselineRuns = runForFeatureCountOnDataset(rawData, ticker, csvPath, baselineFeatures, settings);
                std::vector<FoldRun> engineeredRuns = runForFeatureCountOnDataset(rawData, ticker, csvPath, engineeredFeatures, settings);

                allRuns.insert(allRuns.end(), baselineRuns.begin(), baselineRuns.end());
                allRuns.insert(allRuns.end(), engineeredRuns.begin(), engineeredRuns.end());
            } catch (const std::exception& ex) {
                if (robustDirectoryMode) {
                    std::cerr << "[WARN] skipping dataset " << csvPath << ": " << ex.what() << std::endl;
                    continue;
                }
                throw;
            }
        }

        if (allRuns.empty()) {
            throw std::runtime_error("No successful dataset runs were produced");
        }

        if (!runFeatureAblation) {
            return 0;
        }

        if (ablationReportPath.empty()) {
            ablationReportPath = "build/results/multi_ticker_feature_ablation.csv";
        }

        writeAblationReportCsv(ablationReportPath, allRuns);
        std::cout << "[ABLATION] wrote comparison report to " << ablationReportPath << std::endl;

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
