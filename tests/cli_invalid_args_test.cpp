#include <cstdlib>
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

void test_invalid_cli_cases_fail_nonzero_and_keep_source_tree_clean() {
    const std::filesystem::path repoRoot = findRepoRoot();
    const std::filesystem::path stockBotPath = findStockBotExecutable(repoRoot);

    const std::filesystem::path testsResultsPath = repoRoot / "tests" / "results.csv";
    const std::filesystem::path dataResultsPath = repoRoot / "data" / "results.csv";
    const FileSnapshot beforeTestsResults = snapshotFile(testsResultsPath);
    const FileSnapshot beforeDataResults = snapshotFile(dataResultsPath);

    const std::string base = "cd \"" + repoRoot.string() + "\" && \"" + stockBotPath.string() + "\"";

    const std::vector<std::pair<std::string, std::string>> invalidCases = {
        {"unknown flag", base + " --definitely-invalid-option"},
        {"epochs missing value", base + " --epochs"},
        {"epochs not-a-number", base + " --epochs not-a-number"},
        {"seed not-a-number", base + " --seed not-a-number"},
        {"results-file missing value", base + " --results-file"},
        {"epochs zero", base + " --epochs 0"},
        {"epochs negative", base + " --epochs -1"},
        {"patience zero", base + " --early-stop-patience 0"},
        {"feature-count missing value", base + " --feature-count"},
        {"feature-count not-a-number", base + " --feature-count nope"},
        {"feature-count zero", base + " --feature-count 0"},
        {"feature-count negative", base + " --feature-count -1"},
        {"feature-count above engineered max", base + " --feature-count 14"},
        {"feature-count unsupported mid value", base + " --feature-count 7"},
        {"ablation report missing value", base + " --ablation-report"},
        {"ablation with explicit feature-count is ambiguous", base + " --feature-ablation --feature-count 6"},
        {"data-path missing value", base + " --data-path"},
        {"data-dir missing value", base + " --data-dir"},
        {"max-files missing value", base + " --max-files"},
        {"file-offset missing value", base + " --file-offset"},
        {"max-files zero", base + " --data-dir data --max-files 0"},
        {"max-files negative", base + " --data-dir data --max-files -1"},
        {"file-offset negative", base + " --data-dir data --file-offset -1"},
        {"max-files without data-dir", base + " --max-files 1"},
        {"file-offset without data-dir", base + " --file-offset 0"},
        {"file-offset beyond discovered files", base + " --data-dir data --file-offset 999999"},
        {"data-path does not exist", base + " --data-path build/test_outputs/no_such_file.csv"},
        {"data-dir does not exist", base + " --data-dir build/test_outputs/no_such_dir"},
        {"data-path and data-dir both provided", base + " --data-path data/AAAU.csv --data-dir data"},
    };

    for (const auto& [name, cmd] : invalidCases) {
        const int exitCode = runCommand(cmd);
        expectTrue(exitCode != 0, "invalid case should exit non-zero: " + name);
    }

    const FileSnapshot afterTestsResults = snapshotFile(testsResultsPath);
    const FileSnapshot afterDataResults = snapshotFile(dataResultsPath);

    expectTrue(sameSnapshot(beforeTestsResults, afterTestsResults),
               "invalid CLI runs should not modify tests/results.csv");
    expectTrue(sameSnapshot(beforeDataResults, afterDataResults),
               "invalid CLI runs should not modify data/results.csv");
}

} // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"invalid CLI cases fail safely", test_invalid_cli_cases_fail_nonzero_and_keep_source_tree_clean},
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

    std::cout << "\nCLI invalid-arg tests: " << passed << "/" << tests.size() << " passed\n";
    return passed == tests.size() ? 0 : 1;
}
