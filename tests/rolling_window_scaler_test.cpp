#include "inputs/parser.h"
#include "inputs/rolling_window_scaler.h"

#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kTol = 1e-9;

void expectNear(double actual, double expected, const std::string& msg) {
    if (std::abs(actual - expected) > kTol) {
        throw std::runtime_error(msg + " expected=" + std::to_string(expected) + " actual=" + std::to_string(actual));
    }
}

void expectEq(std::size_t actual, std::size_t expected, const std::string& msg) {
    if (actual != expected) {
        throw std::runtime_error(msg + " expected=" + std::to_string(expected) + " actual=" + std::to_string(actual));
    }
}

PriceData makePrice(double open, double high, double low, double close, double adjClose, uint64_t volume) {
    PriceData p{};
    p.open = open;
    p.high = high;
    p.low = low;
    p.close = close;
    p.adjClose = adjClose;
    p.volume = volume;
    p.date = "";
    return p;
}

void test_first_value_after_reset_zero() {
    RollingWindowScaler scaler(3, 6);
    scaler.add(makePrice(10.0, 10.0, 10.0, 10.0, 10.0, 10));
    auto scaled = scaler.scaledValuesPerDay();
    expectEq(scaled.size(), 6, "scaled vector size");
    for (double v : scaled) {
        expectNear(v, 0.0, "first value should scale to 0");
    }

    scaler.reset();
    scaler.add(makePrice(42.0, 42.0, 42.0, 42.0, 42.0, 42));
    scaled = scaler.scaledValuesPerDay();
    for (double v : scaled) {
        expectNear(v, 0.0, "first value after reset should scale to 0");
    }
}

void test_constant_values_scale_to_zero() {
    RollingWindowScaler scaler(3, 6);

    scaler.add(makePrice(5.0, 5.0, 5.0, 5.0, 5.0, 5));
    for (double v : scaler.scaledValuesPerDay()) {
        expectNear(v, 0.0, "constant day 1 should scale to 0");
    }

    scaler.add(makePrice(5.0, 5.0, 5.0, 5.0, 5.0, 5));
    for (double v : scaler.scaledValuesPerDay()) {
        expectNear(v, 0.0, "constant day 2 should scale to 0");
    }

    scaler.add(makePrice(5.0, 5.0, 5.0, 5.0, 5.0, 5));
    for (double v : scaler.scaledValuesPerDay()) {
        expectNear(v, 0.0, "constant day 3 should scale to 0");
    }
}

void test_mean_std_for_growing_window() {
    RollingWindowScaler scaler(3, 6);

    scaler.add(makePrice(1.0, 1.0, 1.0, 1.0, 1.0, 1));
    expectNear(scaler.scaledValuesPerDay()[0], 0.0, "day 1 open scaled");

    scaler.add(makePrice(2.0, 2.0, 2.0, 2.0, 2.0, 2));
    expectNear(scaler.scaledValuesPerDay()[0], 1.0, "day 2 open scaled");

    scaler.add(makePrice(3.0, 3.0, 3.0, 3.0, 3.0, 3));
    expectNear(scaler.scaledValuesPerDay()[0], 1.224744871391589, "day 3 open scaled");
}

void test_window_dropoff_behavior() {
    RollingWindowScaler scaler(3, 6);

    scaler.add(makePrice(1.0, 1.0, 1.0, 1.0, 1.0, 1));
    scaler.add(makePrice(2.0, 2.0, 2.0, 2.0, 2.0, 2));
    scaler.add(makePrice(3.0, 3.0, 3.0, 3.0, 3.0, 3));

    scaler.add(makePrice(4.0, 4.0, 4.0, 4.0, 4.0, 4));
    // Window for day 4 should be [2, 3, 4]
    expectNear(scaler.scaledValuesPerDay()[0], 1.224744871391589, "day 4 open scaled with drop-off window [2,3,4]");
}

void test_features_scaled_independently() {
    RollingWindowScaler scaler(3, 6);

    scaler.add(makePrice(1.0, 10.0, 100.0, 5.0, 50.0, 1000));
    scaler.add(makePrice(2.0, 10.0, 200.0, 5.0, 60.0, 2000));
    auto scaled = scaler.scaledValuesPerDay();

    expectNear(scaled[0], 1.0, "open should scale independently");
    expectNear(scaled[1], 0.0, "high constant feature should scale to 0");
    expectNear(scaled[2], 1.0, "low should scale independently");
    expectNear(scaled[3], 0.0, "close constant feature should scale to 0");
    expectNear(scaled[4], 1.0, "adjClose should scale independently");
}

void test_volume_scaled_correctly() {
    RollingWindowScaler scaler(3, 6);

    scaler.add(makePrice(0.0, 0.0, 0.0, 0.0, 0.0, 100));
    scaler.add(makePrice(0.0, 0.0, 0.0, 0.0, 0.0, 300));
    auto scaled = scaler.scaledValuesPerDay();

    // Volume is index 5 and should be scaled with the same rolling-stat logic.
    expectNear(scaled[5], 1.0, "volume feature should be included and scaled");
}

void test_reset_clears_state() {
    RollingWindowScaler scaler(3, 6);

    scaler.add(makePrice(1.0, 2.0, 3.0, 4.0, 5.0, 6));
    scaler.add(makePrice(2.0, 3.0, 4.0, 5.0, 6.0, 7));

    scaler.reset();
    scaler.add(makePrice(99.0, 99.0, 99.0, 99.0, 99.0, 99));
    auto scaled = scaler.scaledValuesPerDay();

    for (double v : scaled) {
        expectNear(v, 0.0, "reset should clear internal rolling state");
    }
}

void test_current_day_uses_available_window_not_future() {
    RollingWindowScaler scaler(3, 6);

    scaler.add(makePrice(1.0, 1.0, 1.0, 1.0, 1.0, 1));
    double day1 = scaler.scaledValuesPerDay()[0];

    scaler.add(makePrice(2.0, 2.0, 2.0, 2.0, 2.0, 2));
    double day2 = scaler.scaledValuesPerDay()[0];

    scaler.add(makePrice(3.0, 3.0, 3.0, 3.0, 3.0, 3));
    double day3 = scaler.scaledValuesPerDay()[0];

    scaler.add(makePrice(4.0, 4.0, 4.0, 4.0, 4.0, 4));
    double day4 = scaler.scaledValuesPerDay()[0];

    expectNear(day1, 0.0, "day 1 should use window [1]");
    expectNear(day2, 1.0, "day 2 should use window [1,2]");
    expectNear(day3, 1.224744871391589, "day 3 should use window [1,2,3]");
    expectNear(day4, 1.224744871391589, "day 4 should use window [2,3,4] without future values");
}

} // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"first value after reset is zero", test_first_value_after_reset_zero},
        {"constant values scale to zero", test_constant_values_scale_to_zero},
        {"mean/std for growing window", test_mean_std_for_growing_window},
        {"window drop-off behavior", test_window_dropoff_behavior},
        {"features scaled independently", test_features_scaled_independently},
        {"volume feature scaling", test_volume_scaled_correctly},
        {"reset clears all state", test_reset_clears_state},
        {"current day uses available window only", test_current_day_uses_available_window_not_future},
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

    std::cout << "\nRollingWindowScaler tests: " << passed << "/" << tests.size() << " passed\n";
    return passed == tests.size() ? 0 : 1;
}
