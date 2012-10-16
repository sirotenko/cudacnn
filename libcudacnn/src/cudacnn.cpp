

#include "precomp.hpp"

//using namespace H5;
using namespace std;

namespace cudacnn
{

void CudaInfo()
{
	cudaError_t err = cudaSuccess;
	int            deviceCount;
	cudaDeviceProp devProp;

	err = cudaGetDeviceCount ( &deviceCount );
	if(!deviceCount) 
	{
		printf("Error: No CUDA devices found!");
		return;
	}

	printf ( "Found %d devices\n", deviceCount );

	for ( int device = 0; device < deviceCount; device++ )
	{
		cudaGetDeviceProperties ( &devProp, device );

		printf ( "Device %d\n", device );
		printf ( "Compute capability     : %d.%d\n", devProp.major, devProp.minor );
		printf ( "Name                   : %s\n", devProp.name );
		printf ( "Total Global Memory    : %d\n", devProp.totalGlobalMem );
		printf ( "Shared memory per block: %d\n", devProp.sharedMemPerBlock );
		printf ( "Registers per block    : %d\n", devProp.regsPerBlock );
		printf ( "Warp size              : %d\n", devProp.warpSize );
		printf ( "Max threads per block  : %d\n", devProp.maxThreadsPerBlock );
		printf ( "Total constant memory  : %d\n", devProp.totalConstMem );
	}

}


}

