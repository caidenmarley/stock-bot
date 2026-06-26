#pragma once

#include <vector>

namespace metrics{
struct ProfitAndLossParams{ // PnL
    double thresholdToEnter; // if = 0.001 then only enter trade when model predicts > 0.1% (change position to 1)
    double costToChangePos; // cost to change your position by 1
    int periodsPerYear; // 252 trading days per year
};

/**
 * Function to convert the models predictions to positions
 * 1.0 (go long, meaning you own it and ride the move increase) if predictions > thresholdToEnter
 * 0.0 (stay flat, you hold nothing) otherwise
 *
 * @param predictions vector of models predictions per day (expected next day returns in decimal 0.002 = +0.2%)
 * @param thresholdToEnter cutoff value (in decimal 0.002 = +0.2%)
 * @return vector of positions
 */
std::vector<double> predictionsToPositions(const std::vector<double>& predictions, double thresholdToEnter);

/**
 * Function to convert the models predictions to scaled positions
 * 1.0 (go long, meaning you own it and ride the move increase) if predictions > thresholdToEnter
 * 0.0 (stay flat, you hold nothing) otherwise
 *
 * @param predictions vector of models predictions per day (expected next day returns in decimal 0.002 = +0.2%)
 * @param scaleFactor scale factor to apply to the predictions
 * @return vector of scaled positions
 */
std::vector<double> predictionsToScaledPositions(const std::vector<double>& predictions, double scaleFactor);

struct DailyPnLAndTurnover{
    std::vector<double> grossReturn; // return BEFORE costs for each day
    std::vector<double> netReturn; // return AFTER costs for each day
    std::vector<double> turnover; // how much the position changed compared to the day before
};

/**
 * Calculates daily PnL & turnover from positions and actual returns
 *
 * All these are daily:
 * - grossReturn[t] = positions[t] * actualReturns[t]
 *      if position = 1.0 (you own it), you get the full return
 *      if position = 0.0 (you hold nothing), you get 0
 *
 * - turnover[t] = abs(positions[t] - positions[t-1])  // with positions[-1]=0
 *      measures how much position changed today compared to yesterday
 *
 * - tradingCost[t] = costToChangePos * turnover[t]
 *      simple cost model to determine how much you pay when your position changes
 *
 * - netReturn[t] = grossReturn[t] - tradingCost[t]
 *      actual return after costs
 *
 * @param positions vector of positions per day
 * @param actualReturns vector of actual returns per day
 * @param costToChangePos incurred cost upon position change
 * @return DailyPnLAndTurnover struct
 */
DailyPnLAndTurnover calcDailyPnLAndTurnover(
    const std::vector<double>& positions,
    const std::vector<double>& actualReturns,
    double costToChangePos
);

/**
 * Mean of values, returns 0.0 for empty input
 */
double mean(const std::vector<double>& vec);

/**
 * Sample standard deviation
 * returns 0.0 if size < 2
 */
double stddev(const std::vector<double>& vec);

/**
 * Annualized Sharpe ratio for a dailyNetReturns vector
 * Sharpe = (mean(daily) / stddev(daily)) * sqrt(periodsPerYear)
 * Returns 0.0 if stddev is 0 or input empty
 *
 * @param dailyNetReturns vector of daily net returns (after costs)
 * @param periodsPerYear 252 for daily data
 * @return sharpe ratio
 */
double sharpe(const std::vector<double>& dailyNetReturns, int periodsPerYear);

struct SharpeAndTurnover{
    double sharpeNet; // sharpe of net daily returns after trading costs
    double avgTurnover;
};

/**
 * Runs the full metrics pipeline
 *
 * @param predictions vector of models predictions per day (expected next day returns in decimal 0.002 = +0.2%)
 * @param actualReturns vector of actual returns per day
 * @param params thresholdToEnter, costToChangePos &periodsPerYear
 * @return sharpe and avgTurnover
 */
SharpeAndTurnover calcSharpeAndTurnover(
    const std::vector<double>& predictions,
    const std::vector<double>& actualReturns,
    const ProfitAndLossParams& params
);
}