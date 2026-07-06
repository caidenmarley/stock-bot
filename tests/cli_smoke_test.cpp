#include <cstdlib>
#include <algorithm>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iostream>
#include <sstream>
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

std::filesystem::path findRepoRoot() {
    std::filesystem::path cur = std::filesystem::current_path();

    for (int up = 0; up < 6; ++up) {
        const bool hasCMake = std::filesystem::exists(cur / "CMakeLists.txt");
        const bool hasData = std::filesystem::exists(cur / "data" / "AAAU.csv");
        if (hasCMake && hasData) {
            return cur;
        }
        if (!cur.has_parent_path()) {
            break;
        }
        cur = cur.parent_path();
    }

    throw std::runtime_error("could not locate repository root containing CMakeLists.txt and data/AAAU.csv");
}

std::filesystem::path findStockBotExecutable(const std::filesystem::path& repoRoot) {
    const std::vector<std::filesystem::path> candidates = {
        repoRoot / "build" / "stock_bot",
        repoRoot / "stock_bot"
    };

    for (const auto& c : candidates) {
        std::error_code ec;
        if (std::filesystem::exists(c, ec) && !ec) {
            return c;
        }
    }

    throw std::runtime_error("could not locate stock_bot executable in expected paths");
}

int runCommand(const std::string& command) {
    return std::system(command.c_str());
}

void writeSyntheticCsv(const std::filesystem::path& filePath, int rows, double baseClose) {
    if (filePath.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(filePath.parent_path(), ec);
    }

    std::ofstream out(filePath);
    if (!out) {
        throw std::runtime_error("failed to create synthetic csv: " + filePath.string());
    }

    out << "Date,Open,High,Low,Close,Adj Close,Volume\n";
    for (int i = 0; i < rows; ++i) {
        const double close = baseClose + static_cast<double>(i) * 0.5;
        const double open = close - 0.2;
        const double high = close + 0.4;
        const double low = close - 0.5;
        const double adjClose = close;
        const std::uint64_t volume = static_cast<std::uint64_t>(100000 + i * 100);
        out << "2020-01-" << (i + 1) << ","
            << open << ","
            << high << ","
            << low << ","
            << close << ","
            << adjClose << ","
            << volume << "\n";
    }
}

std::vector<std::string> splitCsv(const std::string& line) {
    std::vector<std::string> cols;
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, ',')) {
        cols.push_back(item);
    }
    return cols;
}

void test_cli_no_results_and_safe_results_file_paths() {
    const std::filesystem::path repoRoot = findRepoRoot();
    const std::filesystem::path stockBotPath = findStockBotExecutable(repoRoot);

    const std::filesystem::path testsResultsPath = repoRoot / "tests" / "results.csv";
    const std::filesystem::path dataResultsPath = repoRoot / "data" / "results.csv";
    const FileSnapshot beforeTestsResults = snapshotFile(testsResultsPath);
    const FileSnapshot beforeDataResults = snapshotFile(dataResultsPath);

    const std::filesystem::path safeResultsPath = repoRoot / "build" / "test_outputs" / "cli_smoke_results.csv";
    std::error_code ec;
    std::filesystem::remove(safeResultsPath, ec);

    const std::filesystem::path fixtureRoot = repoRoot / "build" / "test_outputs" / "cli_fixture_data";
    std::filesystem::remove_all(fixtureRoot, ec);
    std::filesystem::create_directories(fixtureRoot, ec);

    const std::filesystem::path singleCsvPath = fixtureRoot / "single.csv";
    writeSyntheticCsv(singleCsvPath, /*rows=*/60, /*baseClose=*/100.0);

    const std::filesystem::path multiDirPath = fixtureRoot / "multi";
    std::filesystem::create_directories(multiDirPath, ec);
    writeSyntheticCsv(multiDirPath / "aaa.csv", /*rows=*/60, /*baseClose=*/120.0);
    writeSyntheticCsv(multiDirPath / "zzz.csv", /*rows=*/60, /*baseClose=*/160.0);
    writeSyntheticCsv(multiDirPath / "too_short.csv", /*rows=*/20, /*baseClose=*/200.0);
    {
        std::ofstream malformed(multiDirPath / "malformed.csv");
        malformed << "Date,Open,High,Low,Close,Adj Close,Volume\n";
        malformed << "bad,row,that,should,fail,parsing,text\n";
    }

    const std::string runNoResults =
        "cd \"" + repoRoot.string() + "\" && \"" + stockBotPath.string() +
        "\" --epochs 1 --seed 0 --early-stop-patience 1 --no-results";

    const int noResultsExit = runCommand(runNoResults);
    expectTrue(noResultsExit == 0,
               "stock_bot should exit with status 0 for minimal run with --no-results");

    const std::string runDataPathSingle =
        "cd \"" + repoRoot.string() + "\" && \"" + stockBotPath.string() +
        "\" --epochs 1 --seed 0 --early-stop-patience 1 --data-path \"" +
        singleCsvPath.string() + "\" --feature-count 13 --no-results";
    const int dataPathSingleExit = runCommand(runDataPathSingle);
    expectTrue(dataPathSingleExit == 0,
               "stock_bot should accept --data-path for a single synthetic CSV");

    const std::string runFeature6 =
        "cd \"" + repoRoot.string() + "\" && \"" + stockBotPath.string() +
        "\" --epochs 1 --seed 0 --early-stop-patience 1 --feature-count 6 --no-results";
    const int feature6Exit = runCommand(runFeature6);
    expectTrue(feature6Exit == 0,
               "stock_bot should accept --feature-count 6 for a short deterministic run");

    const std::string runFeature13 =
        "cd \"" + repoRoot.string() + "\" && \"" + stockBotPath.string() +
        "\" --epochs 1 --seed 0 --early-stop-patience 1 --feature-count 13 --no-results";
    const int feature13Exit = runCommand(runFeature13);
    expectTrue(feature13Exit == 0,
               "stock_bot should accept --feature-count 13 for a short deterministic run");

    const std::string runSafeResults =
        "cd \"" + repoRoot.string() + "\" && \"" + stockBotPath.string() +
        "\" --epochs 1 --seed 0 --early-stop-patience 1 --results-file \"" +
        safeResultsPath.string() + "\"";

    const int safeResultsExit = runCommand(runSafeResults);
    expectTrue(safeResultsExit == 0,
               "stock_bot should exit with status 0 for minimal run with --results-file");

    const std::filesystem::path ablationReportPath = repoRoot / "build" / "test_outputs" / "feature_ablation_smoke.csv";
    std::filesystem::remove(ablationReportPath, ec);

    const std::string runAblationDir =
        "cd \"" + repoRoot.string() + "\" && \"" + stockBotPath.string() +
        "\" --epochs 1 --seed 0 --early-stop-patience 1 --feature-ablation --data-dir \"" +
        multiDirPath.string() + "\" --ablation-report \"" +
        ablationReportPath.string() + "\" --no-results";
    const int ablationDirExit = runCommand(runAblationDir);
    expectTrue(ablationDirExit == 0,
               "stock_bot should run deterministic multi-ticker 6-vs-13 ablation mode");

    expectTrue(std::filesystem::exists(safeResultsPath),
               "stock_bot should create build-local results file when --results-file is used");
    expectTrue(std::filesystem::file_size(safeResultsPath) > 0,
               "build-local CLI smoke result file should be non-empty");

    expectTrue(std::filesystem::exists(ablationReportPath),
               "ablation report path should be created");
    expectTrue(std::filesystem::file_size(ablationReportPath) > 0,
               "ablation report should be non-empty");

    std::ifstream ablationIn(ablationReportPath);
    expectTrue(static_cast<bool>(ablationIn), "ablation report should be readable");
    std::string ablationHeader;
    expectTrue(static_cast<bool>(std::getline(ablationIn, ablationHeader)),
               "ablation report should include a header");
    expectTrue(
        ablationHeader ==
            "ticker,csv_path,feature_count,fold,best_val_loss,epoch_of_best_val_loss,total_epochs,model_sharpe_net,avg_turnover,benchmark_strategy,benchmark_sharpe_net,benchmark_avg_turnover,benchmark_cumulative_net_return,benchmark_num_observations",
        "ablation report header should match stable schema");

    std::string firstDataLine;
    expectTrue(static_cast<bool>(std::getline(ablationIn, firstDataLine)),
               "ablation report should include at least one data line");
    std::vector<std::string> firstCols = splitCsv(firstDataLine);
    expectTrue(firstCols.size() == 14,
               "ablation report data lines should contain 14 CSV columns");
    expectTrue(firstCols[0] == "aaa",
               "data-dir CSV discovery should be deterministic and sorted by filename");
    expectTrue(firstCols[1].find("aaa.csv") != std::string::npos,
               "first ablation row should reference the sorted first csv path");

    const FileSnapshot afterTestsResults = snapshotFile(testsResultsPath);
    const FileSnapshot afterDataResults = snapshotFile(dataResultsPath);

    expectTrue(sameSnapshot(beforeTestsResults, afterTestsResults),
               "CLI smoke runs should not modify tests/results.csv");
    expectTrue(sameSnapshot(beforeDataResults, afterDataResults),
               "CLI smoke runs should not modify data/results.csv");
}

} // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"stock_bot minimal CLI runs with safe output behavior", test_cli_no_results_and_safe_results_file_paths},
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

    std::cout << "\nCLI smoke tests: " << passed << "/" << tests.size() << " passed\n";
    return passed == tests.size() ? 0 : 1;
}
