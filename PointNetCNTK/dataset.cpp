#include "stdafx.h"
#include "dataset.h"

bool ShapeNetCoreDataset::Load(const std::string& rootDir, size_t num_points)
{
	std::cout << "Loading dataset..." << std::endl;

	std::filesystem::path rootDirPath(rootDir);

	if (!std::filesystem::is_directory(rootDirPath))
	{
		std::cerr << "Cannot find directory: " << rootDir << std::endl;
		return false;
	}

	if (!LoadClassNameDict(rootDirPath))
		return false;

	if (!LoadSplits(rootDirPath))
		return false;

	std::vector<std::tuple<
		std::vector<std::pair<std::string, std::string>>*, 
		std::vector<std::vector<std::array<float, 3>>>*, 
		std::vector<std::vector<size_t>>*
		>>
		splitTuples =
	{
		{ &trainingSplit, &trainingPointsets, &trainingParts },
		{ &validationSplit, &validationPointsets, &validationParts },
		{ &testingSplit, &testingPointsets, &testingParts },
	};

	volatile bool failed = false;

#pragma omp parallel for num_threads(3)
	for (int i = 0; i < splitTuples.size(); i++)
	{
		auto& tuple = splitTuples[i];
		const auto& splitList = *std::get<0>(tuple);
		auto& pointsetList = *std::get<1>(tuple);
		auto& partsList = *std::get<2>(tuple);

		for (const auto& pair : splitList)
		{
			if (failed)
				continue;

			const std::string& classId = pair.first;
			const std::string& fileId = pair.second;

			std::filesystem::path ptsFilePath = (rootDirPath / classId / "points" / fileId).concat(".pts");
			std::filesystem::path segFilePath = (rootDirPath / classId / "points_label" / fileId).concat(".seg");

			std::vector<std::array<float, 3>> pointset;
			std::vector<size_t> pointParts;

			if (LoadPointFile(ptsFilePath, pointset) && LoadSegFile(segFilePath, pointParts))
			{
				pointsetList.push_back(pointset);
				partsList.push_back(pointParts);
			}
			else
			{
				failed = true;
			}
		}
	}

	if (failed)
		return false;

	this->num_points = num_points;

	std::cout << "Finished loading dataset!" << std::endl;

	return true;
}

size_t ShapeNetCoreDataset::GetNumClasses() const
{
	return allClassIds.size();
}

bool ShapeNetCoreDataset::LoadClassNameDict(std::filesystem::path rootDirPath)
{
	std::filesystem::path dictFilePath = rootDirPath / "synsetoffset2category.txt";

	std::ifstream fin(dictFilePath);

	if (!fin.is_open())
	{
		std::cerr << "Unable to open file: " << dictFilePath << std::endl;
		return false;
	}

	std::string line;

	while (std::getline(fin, line))
	{
		std::istringstream strin(line);

		std::string className;
		std::string classId;

		strin >> className >> classId;

		if (strin.fail())
			break;

		classIndexDict[classId] = allClassIds.size();
		allClassIds.push_back(className);
		classNameDict[classId] = className;
	}

	return true;
}

bool ShapeNetCoreDataset::LoadSplits(std::filesystem::path rootDirPath)
{
	std::filesystem::path splitDir = rootDirPath / "train_test_split";
	std::filesystem::path trainingSplitFile = splitDir / "shuffled_train_file_list.json";
	std::filesystem::path validationSplitFile = splitDir / "shuffled_val_file_list.json";
	std::filesystem::path testingSplitFile = splitDir / "shuffled_test_file_list.json";

	std::vector<std::tuple<std::filesystem::path*, std::vector<std::pair<std::string, std::string>>*, std::vector<size_t>*>> 
		splitTuples =
	{ 
		{ &trainingSplitFile, &trainingSplit, &trainingClasses },
		{ &validationSplitFile, &validationSplit, &validationClasses },
		{ &testingSplitFile, &testingSplit, &testingClasses },
	};

#pragma omp parallel for num_threads(3)
	for (int i = 0; i < splitTuples.size(); i++)
	{
		auto& tuple = splitTuples[i];
		const std::filesystem::path& filePath = *std::get<0>(tuple);
		auto& splitList = *std::get<1>(tuple);
		auto& classList = *std::get<2>(tuple);

		std::ifstream fin(filePath);

		if (!fin.is_open())
		{
			std::cerr << "Unable to open file: " << filePath << std::endl;
			return false;
		}

		auto json = nlohmann::json::parse(fin);

		assert(json.is_array());

		for (const auto& elem : json)
		{
			std::string entry = elem.get<std::string>();
			std::istringstream strin(entry);
			std::string shape_data;
			std::string classId;
			std::string fileId;
			std::getline(strin, shape_data, '/');
			std::getline(strin, classId, '/');
			std::getline(strin, fileId, '/');

			splitList.emplace_back(classId, fileId);
			classList.push_back(classIndexDict[classId]);
		}
	}

	return true;
}

bool ShapeNetCoreDataset::LoadPointFile(std::filesystem::path filePath, std::vector<std::array<float, 3>>& result)
{
	std::ifstream fin(filePath);

	if (!fin.is_open())
	{
		std::cerr << "Unable to open file: " << filePath << std::endl;
		return false;
	}

	while (!fin.eof())
	{
		std::array<float, 3> point;
		fin >> point[0] >> point[1] >> point[2];

		if (fin.fail())
			break;

		result.push_back(point);
	}

	return true;
}

bool ShapeNetCoreDataset::LoadSegFile(std::filesystem::path filePath, std::vector<size_t>& result)
{
	std::ifstream fin(filePath);

	if (!fin.is_open())
	{
		std::cerr << "Unable to open file: " << filePath << std::endl;
		return false;
	}

	while (!fin.eof())
	{
		size_t partId;
		fin >> partId;

		if (fin.fail())
			break;

		result.push_back(partId);
	}

	return true;
}