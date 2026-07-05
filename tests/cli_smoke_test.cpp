#include <cstdlib>
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

    const std::string runNoResults =
        "cd \"" + repoRoot.string() + "\" && \"" + stockBotPath.string() +
        "\" --epochs 1 --seed 0 --early-stop-patience 1 --no-results";

    const int noResultsExit = runCommand(runNoResults);
    expectTrue(noResultsExit == 0,
               "stock_bot should exit with status 0 for minimal run with --no-results");

    const std::string runSafeResults =
        "cd \"" + repoRoot.string() + "\" && \"" + stockBotPath.string() +
        "\" --epochs 1 --seed 0 --early-stop-patience 1 --results-file \"" +
        safeResultsPath.string() + "\"";

    const int safeResultsExit = runCommand(runSafeResults);
    expectTrue(safeResultsExit == 0,
               "stock_bot should exit with status 0 for minimal run with --results-file");

    expectTrue(std::filesystem::exists(safeResultsPath),
               "stock_bot should create build-local results file when --results-file is used");
    expectTrue(std::filesystem::file_size(safeResultsPath) > 0,
               "build-local CLI smoke result file should be non-empty");

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
