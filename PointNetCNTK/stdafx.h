// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#include "targetver.h"

#include <algorithm>
#include <numeric>
#include <stdio.h>
#include <tchar.h>
#include <Windows.h>
#include <iostream>
#include <chrono>
#include <random>
#include <functional>
#include <fstream>
#include <memory>
#include <unordered_map>
#include <sstream>
#include <filesystem>

#include <CNTKLibrary.h>

#include <json.hpp>


using namespace std::placeholders;

namespace std
{
	namespace filesystem = experimental::filesystem;
}