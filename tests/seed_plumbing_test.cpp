#include <cstdlib>
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

std::string readSummaryLine(const std::filesystem::path& outputPath) {
    std::ifstream in(outputPath);
    if (!in) {
        throw std::runtime_error("could not open output file for summary parsing: " + outputPath.string());
    }

    std::string line;
    std::string summary;
    while (std::getline(in, line)) {
        if (line.find("[SUMMARY]") != std::string::npos) {
            summary = line;
        }
    }

    return summary;
}

void test_cli_seed_plumbing_same_seed_repeatable_summary() {
    const std::filesystem::path repoRoot = findRepoRoot();
    const std::filesystem::path stockBotPath = findStockBotExecutable(repoRoot);

    const std::filesystem::path outA = repoRoot / "build" / "test_outputs" / "seed_plumbing_a.txt";
    const std::filesystem::path outB = repoRoot / "build" / "test_outputs" / "seed_plumbing_b.txt";
    const std::filesystem::path outUnseeded = repoRoot / "build" / "test_outputs" / "seed_plumbing_unseeded.txt";

    std::error_code ec;
    std::filesystem::remove(outA, ec);
    std::filesystem::remove(outB, ec);
    std::filesystem::remove(outUnseeded, ec);

    const std::string base = "cd \"" + repoRoot.string() + "\" && \"" + stockBotPath.string() +
        "\" --epochs 1 --early-stop-patience 1 --no-results";

    const int exitA = runCommand(base + " --seed 0 > \"" + outA.string() + "\" 2>&1");
    const int exitB = runCommand(base + " --seed 0 > \"" + outB.string() + "\" 2>&1");
    const int exitUnseeded = runCommand(base + " > \"" + outUnseeded.string() + "\" 2>&1");

    expectTrue(exitA == 0, "seeded run A should exit with status 0");
    expectTrue(exitB == 0, "seeded run B should exit with status 0");
    expectTrue(exitUnseeded == 0, "unseeded run should exit with status 0");

    const std::string summaryA = readSummaryLine(outA);
    const std::string summaryB = readSummaryLine(outB);

    expectTrue(!summaryA.empty(), "seeded run A should produce a [SUMMARY] line");
    expectTrue(!summaryB.empty(), "seeded run B should produce a [SUMMARY] line");
    expectTrue(summaryA == summaryB,
               "same-seed CLI runs should produce identical [SUMMARY] output in covered short run");
}

} // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"main/trainer seed plumbing yields repeatable same-seed summary", test_cli_seed_plumbing_same_seed_repeatable_summary},
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

    std::cout << "\nSeed plumbing tests: " << passed << "/" << tests.size() << " passed\n";
    return passed == tests.size() ? 0 : 1;
}
