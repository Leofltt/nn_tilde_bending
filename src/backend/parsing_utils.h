#pragma once
#include <algorithm>
#include <cctype>
#include <iomanip>
#include <sstream>
#include <string>

auto to_bool(std::string str) -> bool;
auto to_int(std::string str) -> int;
auto to_float(std::string str) -> float;