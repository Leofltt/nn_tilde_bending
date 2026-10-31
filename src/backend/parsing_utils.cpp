#include "parsing_utils.h"

auto to_bool(std::string str) -> bool {
  if ((str == "0") || (str == "false")) {
    return false;
  } else {
    return true;
  }
}

auto to_int(std::string str) -> int { return stoi(str); }

auto to_float(std::string str) -> float { return stof(str); }