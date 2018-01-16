// PointNetCNTK.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "cntk_layers.hpp"

int main()
{
	CNTK::DeviceDescriptor gpu = CNTK::DeviceDescriptor::GPUDevice(0);

	std::wcout << gpu.AsString() << std::endl;

	if (IsDebuggerPresent())
		system("pause");

    return 0;
}

