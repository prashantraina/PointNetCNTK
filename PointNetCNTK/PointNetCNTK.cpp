// PointNetCNTK.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "dataset.h"
#include "cntk_layers.hpp"

ShapeNetCoreDataset dataset;

int main()
{
	CNTK::DeviceDescriptor gpu = CNTK::DeviceDescriptor::GPUDevice(0);

	std::wcout << gpu.AsString() << std::endl;

	dataset.Load("E:\\shapenetcore_partanno_segmentation_benchmark_v0");

	if (IsDebuggerPresent())
		system("pause");

    return 0;
}

