#pragma once

#ifdef __INTELLISENSE__
#include <CNTKLibrary.h>
#endif

class ShapeNetCoreDataset
{
private:
	size_t num_points;
	std::unordered_map<std::string, std::string> classNameDict;
	std::unordered_map<std::string, size_t> classIndexDict;
	std::vector<std::string> allClassIds;
	std::vector<std::vector<std::array<float, 3>>> trainingPointsets;
	std::vector<std::vector<std::array<float, 3>>> validationPointsets;
	std::vector<std::vector<std::array<float, 3>>> testingPointsets;
	std::vector<size_t> trainingClasses;
	std::vector<size_t> validationClasses;
	std::vector<size_t> testingClasses;
	std::vector<std::vector<size_t>> trainingParts;
	std::vector<std::vector<size_t>> validationParts;
	std::vector<std::vector<size_t>> testingParts;

	std::vector<std::pair<std::string, std::string>> trainingSplit;
	std::vector<std::pair<std::string, std::string>> validationSplit;
	std::vector<std::pair<std::string, std::string>> testingSplit;

public:
	bool Load(const std::string& rootDir, size_t num_points = 2500);
	size_t GetNumClasses() const;

private:
	bool LoadClassNameDict(std::filesystem::path rootDirPath);
	bool LoadSplits(std::filesystem::path rootDirPath);
	bool LoadPointFile(std::filesystem::path filePath, std::vector<std::array<float, 3>>& result);
	bool LoadSegFile(std::filesystem::path filePath, std::vector<size_t>& result);
};