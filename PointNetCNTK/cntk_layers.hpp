#pragma once

#ifdef __INTELLISENSE__
#include <CNTKLibrary.h>
#endif

namespace Layers
{
	using namespace CNTK;

	inline long long CurrentTimeCount()
	{
		return std::chrono::high_resolution_clock::now().time_since_epoch().count();
	}

	inline FunctionPtr Linear(Variable input, size_t outputDim, const std::wstring& layerName = L"")
	{
		const DeviceDescriptor device = DeviceDescriptor::UseDefaultDevice();
		const DataType floatType = input.GetDataType();

		assert(input.Shape().Rank() == 1);
		size_t inputDim = input.Shape()[0];

		Parameter weightsParam({ outputDim, inputDim }, floatType,
			GlorotUniformInitializer(1, SentinelValueForInferParamInitRank, SentinelValueForInferParamInitRank,
			//NormalInitializer(0.05, SentinelValueForInferParamInitRank, SentinelValueForInferParamInitRank,
				CurrentTimeCount()), device, layerName + L"/weights");

		Parameter biasesParam({ outputDim }, floatType, 0.0, device, layerName + L"/biases");

		return Plus(biasesParam, Times(weightsParam, input, layerName + L"times_op"), layerName + L"plus_op");
	}

	inline FunctionPtr Linear(Variable input, size_t outputDim,
		const std::function<CNTK::FunctionPtr(const CNTK::FunctionPtr&, std::wstring)>& nonLinearity,
		const std::wstring& layerName = L"")
	{
		return nonLinearity(Linear(input, outputDim, layerName), layerName + L"/activation");
	}

	//assumes W x H x C

	/*inline FunctionPtr Conv2D(Variable input,  size_t kernelWidth, size_t kernelHeight,size_t outFeatureMapCount, bool autoPadding = true,
		const std::wstring& layerName = L"", bool bias = true)
	{
		const DeviceDescriptor device = DeviceDescriptor::CPUDevice();
		const DataType floatType = input.GetDataType();

		NDShape inputShape = input.Shape();
		size_t numInputChannels = inputShape[inputShape.Rank() - 1];
		NDShape strides = { 1, 1, numInputChannels };

		Parameter convParams({ kernelWidth, kernelHeight, numInputChannels, outFeatureMapCount }, floatType,
			GlorotUniformInitializer(1.0, -1, 2, CurrentTimeCount()), device, 
			layerName + L"/kernels");
		Parameter biasesParam({ 1, numInputChannels, outFeatureMapCount }, floatType, 0.0, device, layerName + L"/biases");

		FunctionPtr func = Convolution(convParams, input, strides, { true }, { autoPadding });
		func->SetName(layerName + L"/conv2d_op");

		if (bias)
			func = Plus(func, biasesParam, layerName + L"/plus_op");

		return func;
	}

	inline FunctionPtr MaxPooling2D(Variable input, size_t poolWidth, size_t poolHeight, const std::wstring& layerName = L"")
	{
		FunctionPtr func = Pooling(input, PoolingType::Max, { poolWidth, poolHeight, 1 });
		func->SetName(layerName + L"maxpool2d_op");

		return func;
	}*/


	inline FunctionPtr Conv1D(Variable input, size_t kernelWidth, size_t outFeatureMapCount, bool autoPadding = true,
		const std::wstring& layerName = L"", bool bias = true)
	{
		const DeviceDescriptor device = DeviceDescriptor::UseDefaultDevice();
		const DataType floatType = input.GetDataType();

		NDShape inputShape = input.Shape();
		size_t numInputChannels = inputShape[inputShape.Rank() - 1];
		NDShape strides = { 1, numInputChannels };

		Parameter convParams({ kernelWidth, numInputChannels, outFeatureMapCount }, floatType,
			GlorotUniformInitializer(1.0, SentinelValueForInferParamInitRank, SentinelValueForInferParamInitRank, CurrentTimeCount()), 
			device, layerName + L"/kernels");
		/*Parameter convParams({ kernelWidth, numInputChannels, outFeatureMapCount }, floatType,
			0.0,
			device, layerName + L"/kernels");*/
		Parameter biasesParam({ 1, outFeatureMapCount }, floatType, 0.0, device, layerName + L"/biases");

		FunctionPtr func = Convolution(convParams, input, strides, { true }, { autoPadding });
		func->SetName(layerName + L"/conv1d_op");

		if (bias)
			func = Plus(func, biasesParam, layerName + L"/plus_op");

		return func;
	}

	inline FunctionPtr MaxPooling1D(Variable input, size_t poolWidth, const std::wstring& layerName = L"")
	{
		FunctionPtr func = Pooling(input, PoolingType::Max, { poolWidth, 1 }, { 1, 1 });
		func->SetName(layerName + L"maxpool1d_op");

		return func;
	}

	inline FunctionPtr BatchNorm(Variable input, bool inputIsConvolution = true, const std::wstring& layerName = L"")
	{
		const DeviceDescriptor device = DeviceDescriptor::UseDefaultDevice();
		const DataType floatType = input.GetDataType();

		Parameter biasParams({ NDShape::InferredDimension }, floatType, 0.0, device, layerName + L"/params/bias");
		Parameter scaleParams({ NDShape::InferredDimension }, floatType, 1.0, device, layerName + L"/params/scale");
		Parameter runningMean({ NDShape::InferredDimension }, floatType, 0.0, device, layerName + L"/params/running_mean");
		Constant runningInvStd({ NDShape::InferredDimension }, floatType, 0.0, device, layerName + L"/params/running_inv_std");
		Constant runningCount({ }, floatType, 0.0, device, layerName + L"/params/running_count");

		return BatchNormalization(input, scaleParams, biasParams, runningMean, runningInvStd, runningCount, inputIsConvolution, 5000.0, 0.0, 1e-5 /* epsilon */,
			true, false, layerName + L"/batchnorm_op");
	}
};

namespace Pointnet
{
	using namespace CNTK;

	static float identityData[] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };

	inline Variable IdentityMatrix(const std::wstring& parentName = L"")
	{
		NDShape rotationMatShape = { 3, 3 };
		//NDShape rotationMatSeqShape = rotationMatShape.AppendShape({ 1 });
		auto ndarr = MakeSharedObject<NDArrayView>(rotationMatShape, identityData, rotationMatShape.TotalSize(), DeviceDescriptor::CPUDevice());

		return Constant(ndarr, parentName + L"/identity_matrix");
	}

	inline FunctionPtr ConvBNRelu(Variable input,size_t outFeatureMapCount, const std::wstring& parentName = L"")
	{
		FunctionPtr model = Layers::Conv1D(input, 1, outFeatureMapCount, true, parentName + L"/conv", true);
		model = Layers::BatchNorm(model, true, parentName + L"/bnorm");
		model = ReLU(model, parentName + L"/activation");

		return model;
	}

	inline FunctionPtr ConvBN(Variable input, size_t outFeatureMapCount, const std::wstring& parentName = L"")
	{
		FunctionPtr model = Layers::Conv1D(input, 1, outFeatureMapCount, false, parentName + L"/conv", true);
		model = Layers::BatchNorm(model, true, parentName + L"/bnorm");

		return model;
	}

	inline FunctionPtr LinearBNRelu(Variable input, size_t hiddenUnitCount, const std::wstring& parentName = L"")
	{
		FunctionPtr model = Layers::Linear(input, hiddenUnitCount, parentName + L"/linear");
		model = Layers::BatchNorm(model, false, parentName + L"/bnorm");
		model = ReLU(model, parentName + L"/activation");

		return model;
	}

	inline FunctionPtr STN3d(Variable input, size_t num_points, const std::wstring& parentName = L"")
	{
		assert(input.Shape()[input.Shape().Rank() - 2] == num_points);

		FunctionPtr model = input;
		model = ConvBNRelu(model,  64, parentName + L"/layer1");
		model = ConvBNRelu(model,  128, parentName + L"/layer2");
		model = ConvBNRelu(model, 1024, parentName + L"/layer3");
		model = Layers::MaxPooling1D(model, num_points, parentName + L"/maxpool");
		model = Reshape(model, { 1024 }, parentName + L"/reshape_as_1024");
		model = LinearBNRelu(model, 512, parentName + L"/layer4");
		model = LinearBNRelu(model, 256, parentName + L"/layer5");
		model = Layers::Linear(model, 9, parentName + L"/layer6/linear");
		model = Reshape(model, { 3, 3 }, parentName + L"/reshape_as_3x3");
		//model = Sigmoid(model);
		Variable identityMat = IdentityMatrix(parentName);
		//NDShape identityShape = identityMat.Shape();
		//NDShape beforePlusShape = model->Output().Shape();
		model = Plus(model, identityMat, parentName + L"/add_identity_mat");
		//model = Times(identityMat, Transpose(input));
		//NDShape afterPlusShape = model->Output().Shape();

		model = Alias(model, parentName + L"/output");
		//NDShape afterAliasShape = model->Output().Shape();
		return model;
	}

	inline FunctionPtr PointNetFeatures(Variable input, size_t num_points, const std::wstring& parentName = L"/pointnet")
	{
		//NDShape beforeTransShape = input.Shape();
		FunctionPtr trans = STN3d(input, num_points, parentName + L"/trans");
		//NDShape transShape = trans->Output().Shape();
		FunctionPtr model = Times(input, trans);
		//NDShape afterTransShape = model->Output().Shape();
		//model = Reshape(input, { num_points, 3, 1 }, parentName + L"/reshape_as_nx3x1");
//		FunctionPtr model = Reshape(input, { num_points * 3 }, parentName + L"/reshape_as_n*3");
		model = ConvBNRelu(model, 64, parentName + L"/layer1");
		//model = LinearBNRelu(input, 64 * 3, parentName + L"/layer1");
		FunctionPtr pointfeat = Alias(model, parentName + L"/pointfeat");
		model = ConvBNRelu(model, 128, parentName + L"/layer2");
		//model = LinearBNRelu(model, 128 * 3, parentName + L"/layer2");
		model = ConvBN(model, 1024, parentName + L"/layer3");
		//model = Layers::Linear(model, 1024 * 3, parentName + L"/layer3");
		model = Layers::MaxPooling1D(model, num_points, parentName + L"/maxpool");
		//NDShape afterPoolShape = model->Output().Shape();
		model = Reshape(model, { 1024 }, parentName + L"/reshape_as_3x3");

		//model = Alias(model, parentName + L"/pointnetfeat_output");
		return model;
	}


	inline FunctionPtr PointNetClassifier(Variable input, size_t num_points, size_t numClasses = 2, const std::wstring& parentName = L"/pointnet")
	{
		FunctionPtr model = PointNetFeatures(input, num_points, parentName);
		model = LinearBNRelu(model, 512, parentName + L"/layer4");
		model = LinearBNRelu(model, 256, parentName + L"/layer5");
		model = Layers::Linear(model, numClasses, parentName + L"/layer6");
		//model = LogSoftmax(model, parentName + L"/output");

		return model;
	}
};