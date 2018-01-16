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
		const DeviceDescriptor device = DeviceDescriptor::GPUDevice(0);
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

	//assumes H x W x C

	inline FunctionPtr Conv2D(Variable input, size_t kernelHeight, size_t kernelWidth, size_t outFeatureMapCount, bool autoPadding = true,
		const std::wstring& layerName = L"", bool bias = true)
	{
		const DeviceDescriptor device = DeviceDescriptor::GPUDevice(0);
		const DataType floatType = input.GetDataType();

		size_t numInputChannels = input.Shape()[input.Shape().Rank() - 1];
		NDShape strides = { 1, 1, numInputChannels };

		Parameter convParams({ kernelHeight, kernelWidth, numInputChannels, outFeatureMapCount }, floatType,
			CNTK::GlorotUniformInitializer(1.0, -1, 2, CurrentTimeCount()), device, 
			layerName + L"/kernels");
		Parameter biasesParam({ outFeatureMapCount }, floatType, 0.0, device, layerName + L"/biases");

		FunctionPtr func = Convolution(convParams, input, strides, { true }, { autoPadding });
		func->SetName(layerName + L"/conv2d_op");

		if (bias)
			func = Plus(func, biasesParam, layerName + L"/plus_op");

		return func;
	}

	inline FunctionPtr MaxPooling(Variable input, size_t poolHeight, size_t poolWidth, const std::wstring& layerName = L"")
	{
		FunctionPtr func = Pooling(input, PoolingType::Max, { poolHeight, poolWidth, 1 });
		func->SetName(layerName + L"maxpool_op");

		return func;
	}

	inline FunctionPtr BatchNorm(Variable input, bool inputIsConvolution = true, const std::wstring& layerName = L"")
	{
		const DeviceDescriptor device = DeviceDescriptor::GPUDevice(0);
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