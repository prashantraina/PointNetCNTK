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

	CNTK::NDShape pointsetShape, pointsetSeqShape;
	std::mt19937 randEngine;

public:
	ShapeNetCoreDataset(size_t num_points = 2500);
	bool Load(const std::string& rootDir);
	size_t GetNumClasses() const;
	size_t GetNumTrainingPointsets() const;
	CNTK::NDArrayViewPtr GetTrainingPointset(size_t index);
	CNTK::NDArrayViewPtr GetTestingPointset(size_t index);
	size_t GetTrainingClass(size_t index) const;
	size_t GetTestingClass(size_t index) const;
	std::string GetModelClassName(size_t classIndex);

private:
	bool LoadClassNameDict(std::filesystem::path rootDirPath);
	bool LoadSplits(std::filesystem::path rootDirPath);
	bool LoadPointFile(std::filesystem::path filePath, std::vector<std::array<float, 3>>& result);
	bool LoadSegFile(std::filesystem::path filePath, std::vector<size_t>& result);
};