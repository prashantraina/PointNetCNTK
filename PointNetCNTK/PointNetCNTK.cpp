// PointNetCNTK.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "dataset.h"
#include "cntk_layers.hpp"

constexpr size_t num_points = 250;

ShapeNetCoreDataset dataset(num_points);

int main()
{
	CNTK::DeviceDescriptor cpu = CNTK::DeviceDescriptor::CPUDevice();
	CNTK::DeviceDescriptor gpu = CNTK::DeviceDescriptor::GPUDevice(0);
	bool set = CNTK::DeviceDescriptor::TrySetDefaultDevice(gpu);
	CNTK::DeviceDescriptor device = CNTK::DeviceDescriptor::UseDefaultDevice();
	//gpu = cpu;

	std::wcout << gpu.AsString() << std::endl;

	dataset.Load("E:\\shapenetcore_partanno_segmentation_benchmark_v0");

	CNTK::Variable inputVar = CNTK::InputVariable({ num_points, 3 }, CNTK::DataType::Float);
	CNTK::Variable labelVar = CNTK::InputVariable({ dataset.GetNumClasses() }, CNTK::DataType::Float);
	CNTK::FunctionPtr model = Pointnet::PointNetClassifier(inputVar, num_points, dataset.GetNumClasses());
	//CNTK::FunctionPtr model = Pointnet::PointNetClassifier(inputVar, num_points, dataset.GetNumClasses(), L"pointnet");

	size_t totalParam = 0;

	for (auto param : model->Parameters())
	{
		totalParam += param.Shape().TotalSize();
	}

	std::cout << "The model has " << totalParam << " parameters" << std::endl;

	/*auto rand_arr = CNTK::NDArrayView::RandomUniform<float>({ num_points, 3 }, 0.0, 1.0, CNTK::SentinelValueForAutoSelectRandomSeed, gpu);
	auto rand_value = CNTK::MakeSharedObject<CNTK::Value>(rand_arr);

	std::vector<std::vector<float>> input_floats;
	rand_value->CopyVariableValueTo(inputVar, input_floats);

	std::unordered_map<CNTK::Variable, CNTK::ValuePtr> outputs = { { model, nullptr } };
	std::vector<std::vector<float>> output_floats;

	model->Evaluate({ { inputVar, rand_value } }, outputs, gpu);
	outputs[model]->CopyVariableValueTo(model, output_floats);

	return 0;*/

	CNTK::FunctionPtr loss = CNTK::CrossEntropyWithSoftmax(model, labelVar, L"loss");
	CNTK::FunctionPtr metric = CNTK::ClassificationError(model, labelVar, L"metric");

	CNTK::LearningRateSchedule learningRate = CNTK::TrainingParameterPerSampleSchedule(0.0001);
	CNTK::MomentumSchedule momentum = CNTK::TrainingParameterPerSampleSchedule(0.0000);

	CNTK::LearnerPtr learner = CNTK::AdamLearner(model->Parameters(), learningRate, momentum);

	CNTK::TrainerPtr trainer = CNTK::CreateTrainer(model, loss, metric, { learner });

	const size_t num_training_samples = dataset.GetNumTrainingPointsets();
	const size_t num_epochs = 25;
	const size_t minibatch_size = 32;
	const size_t num_minibatches = (num_training_samples + minibatch_size - 1) / minibatch_size;

	std::mt19937 rand_engine(std::random_device{}());

	std::vector<size_t> indices(num_training_samples);
	std::iota(indices.begin(), indices.end(), 0);

	for (size_t epoch_i = 0; epoch_i < num_epochs; epoch_i++)
	{
		std::shuffle(indices.begin(), indices.end(), rand_engine);

		double epoch_loss = 0.0;

		for (size_t batch_i = 0; batch_i < num_minibatches; batch_i++)
		{
			std::vector<CNTK::NDArrayViewPtr> inputArrays;
			std::vector<size_t> labelPrehots;
			inputArrays.reserve(num_minibatches);
			labelPrehots.reserve(num_minibatches);

			for (size_t i = 0; i < minibatch_size && (batch_i * minibatch_size + i) < num_training_samples; i++)
			{
				size_t sampleIndex = indices[batch_i * minibatch_size + i];
				inputArrays.push_back(dataset.GetTrainingPointset(sampleIndex));

				size_t sampleClass = dataset.GetTrainingClass(sampleIndex);

				labelPrehots.push_back(sampleClass);
			}
			
			const CNTK::NDShape inputShape = { num_points, 3 };

			CNTK::ValuePtr input_value = CNTK::Value::Create(inputShape, inputArrays, {}, device, true, true);
			CNTK::ValuePtr label_value = CNTK::Value::CreateBatch<float>(dataset.GetNumClasses(), labelPrehots, device, true);


			const CNTK::NDShape input_value_shape = input_value->Shape();
			const CNTK::NDShape label_value_shape = label_value->Shape();

			trainer->TrainMinibatch({ { inputVar, input_value },{ labelVar, label_value } }, device);

			epoch_loss += trainer->PreviousMinibatchLossAverage();
		}

		epoch_loss /= num_minibatches;

		wprintf_s(L"Epoch %d: avg loss = %lf\n", epoch_i, epoch_loss);

	}//end of epoch loop

	if (IsDebuggerPresent())
		system("pause");

    return 0;
}

