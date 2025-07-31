#pragma once

#include <map>
#include <string>
#include <vector>
#include "model/trainer.h"

enum class ParamType {INTEGER, DOUBLE};

struct HyperParam{
    std::string name;   // e.g. learning rate
    ParamType type; // INTEGER or DOUBLE, type to cast value to
    std::vector<double> values; // values to sweep over
};

// use double and just cast to int if need
using mapping = std::map<std::string, double>;

