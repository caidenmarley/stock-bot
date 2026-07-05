#include "model/metrics.h"
#include <vector>
#include <cassert>
#include <cmath>
#include <numeric>
#include <random>
#include <string>


namespace metrics {

std::vector<double> predictionsToPositions(const std::vector<double>& predictions, double thresholdToEnter) {
    std::vector<double> positions;
    positions.reserve(predictions.size());
    
    // Convert prediction to position
    for(double prediction : predictions) {
        // If prediction > threshold, go long (1.0) otherwise stay flat (0.0)
        positions.push_back(prediction > thresholdToEnter ? 1.0 : 0.0);
    }
    
    return positions;
}

DailyPnLAndTurnover calcDailyPnLAndTurnover(
    const std::vector<double>& positions,
    const std::vector<double>& actualReturns,
    double costToChangePos
) {
    assert(positions.size() == actualReturns.size());
    const size_t n = positions.size();

    DailyPnLAndTurnover result;
    result.grossReturn.reserve(n);
    result.netReturn.reserve(n);
    result.turnover.reserve(n);

    double prevPosition = 0.0; // assume yesterday was flat
    for(size_t i = 0; i < n; ++i) {
        const double gross = positions[i] * actualReturns[i];
        const double turnover = std::abs(positions[i] - prevPosition);
        const double tradingCost = costToChangePos * turnover;
        const double net = gross - tradingCost;

        result.grossReturn.push_back(gross);
        result.netReturn.push_back(net);
        result.turnover.push_back(turnover);

        prevPosition = positions[i];
    }

    return result;
}

// TODO could use Kahan summation to improve accuracy
double mean(const std::vector<double>& vec){
    if(vec.empty()) return 0.0;
    return std::accumulate(vec.begin(), vec.end(), 0.0) / vec.size();
}

double stddev(const std::vector<double>& vec){
    if(vec.size() < 2) return 0.0;
    const double m = mean(vec);
    double sum = 0.0;
    for(double v : vec){
        sum += (v - m)*(v - m);
    }
    // Bessel's correction (-1) for sample standard deviation
    return std::sqrt(sum / (vec.size() - 1));
}

double sharpe(const std::vector<double>& dailyNetReturns, int periodsPerYear){
    if(dailyNetReturns.size() < 2) return 0.0;
    const double m = mean(dailyNetReturns);
    const double dev = stddev(dailyNetReturns);
    if(dev == 0.0) return 0.0;
    return m / dev * std::sqrt(static_cast<double>(periodsPerYear));
}

SharpeAndTurnover calcSharpeAndTurnover(
    const std::vector<double>& predictions,
    const std::vector<double>& actualReturns,
    const ProfitAndLossParams& params
){
    assert(predictions.size() == actualReturns.size());
    const std::vector<double> positions = predictionsToPositions(predictions, params.thresholdToEnter);

    const DailyPnLAndTurnover daily = calcDailyPnLAndTurnover(positions, actualReturns, params.costToChangePos);

    SharpeAndTurnover out;
    out.sharpeNet = sharpe(daily.netReturn, params.periodsPerYear);
    out.avgTurnover = mean(daily.turnover);
    return out;
}

std::vector<double> cashBaselinePositions(std::size_t length) {
    return std::vector<double>(length, 0.0);
}

std::vector<double> buyAndHoldBaselinePositions(std::size_t length) {
    return std::vector<double>(length, 1.0);
}

std::vector<double> randomNoSkillBaselinePositions(std::size_t length, uint32_t seed) {
    std::vector<double> positions;
    positions.reserve(length);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 1);
    for (std::size_t i = 0; i < length; ++i) {
        positions.push_back(static_cast<double>(dist(rng)));
    }

    return positions;
}

std::vector<double> previousReturnMomentumPositions(const std::vector<double>& actualReturns) {
    const std::size_t n = actualReturns.size();
    std::vector<double> positions(n, 0.0);
    if (n == 0) {
        return positions;
    }

    positions[0] = 0.0;
    for (std::size_t t = 1; t < n; ++t) {
        positions[t] = actualReturns[t - 1] > 0.0 ? 1.0 : 0.0;
    }
    return positions;
}

static BenchmarkSummary evaluateBenchmarkFromPositions(
    const std::string& name,
    const std::vector<double>& positions,
    const std::vector<double>& actualReturns,
    const ProfitAndLossParams& params
) {
    assert(positions.size() == actualReturns.size());

    const DailyPnLAndTurnover daily = calcDailyPnLAndTurnover(
        positions,
        actualReturns,
        params.costToChangePos
    );

    BenchmarkSummary out;
    out.name = name;
    out.sharpeNet = sharpe(daily.netReturn, params.periodsPerYear);
    out.avgTurnover = mean(daily.turnover);
    out.cumulativeNetReturn = std::accumulate(daily.netReturn.begin(), daily.netReturn.end(), 0.0);
    out.numObservations = actualReturns.size();
    return out;
}

std::vector<BenchmarkSummary> evaluateStandardBenchmarks(
    const std::vector<double>& actualReturns,
    const ProfitAndLossParams& params,
    uint32_t randomSeed
) {
    const std::size_t n = actualReturns.size();

    std::vector<BenchmarkSummary> out;
    out.reserve(4);

    out.push_back(evaluateBenchmarkFromPositions(
        "cash",
        cashBaselinePositions(n),
        actualReturns,
        params
    ));

    out.push_back(evaluateBenchmarkFromPositions(
        "buy_and_hold",
        buyAndHoldBaselinePositions(n),
        actualReturns,
        params
    ));

    out.push_back(evaluateBenchmarkFromPositions(
        "random_noskill",
        randomNoSkillBaselinePositions(n, randomSeed),
        actualReturns,
        params
    ));

    out.push_back(evaluateBenchmarkFromPositions(
        "prev_return_momentum",
        previousReturnMomentumPositions(actualReturns),
        actualReturns,
        params
    ));

    return out;
}

std::vector<BenchmarkSummary> evaluateModelAndBenchmarks(
    const std::vector<double>& modelPredictions,
    const std::vector<double>& actualReturns,
    const ProfitAndLossParams& params,
    uint32_t randomSeed
) {
    assert(modelPredictions.size() == actualReturns.size());

    std::vector<BenchmarkSummary> out;
    out.reserve(5);

    const std::vector<double> modelPositions = predictionsToPositions(
        modelPredictions,
        params.thresholdToEnter
    );
    out.push_back(evaluateBenchmarkFromPositions(
        "model",
        modelPositions,
        actualReturns,
        params
    ));

    const std::vector<BenchmarkSummary> benchmarks = evaluateStandardBenchmarks(
        actualReturns,
        params,
        randomSeed
    );
    out.insert(out.end(), benchmarks.begin(), benchmarks.end());

    return out;
}

}