#define CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>

#include "tests_common.h"

TEST(CUDATest, CudaDeviceTest)
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	ASSERT_TRUE(deviceCount > 0);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	ASSERT_TRUE(deviceProp.major >= 1);
	printf("CUDA device found: %s \n",deviceProp.name);
}

TEST(CNNetTest, ReadFromHDF)
{
	CNNetCudaF cnnet;
	cnnet.LoadFromFile("test_mnist_net.h5");
}



int main(int argc, char **argv) 
{
  	::testing::InitGoogleTest(&argc, argv);
	RUN_ALL_TESTS();
	
    //_CrtDumpMemoryLeaks();
	return 0;
	
}
