#ifndef _TESTS_COMMON_H
#define _TESTS_COMMON_H

#define GPU_FLOAT_TOLLERANCE 0.00001
using namespace cudacnn;

template<class T>
void RandomWeightInit(Tensor<T>& weight)
{
	for (UINT i = 0; i < weight.num_elements(); ++i)	{
		weight[i] = -0.5f + (double(std::rand()) / RAND_MAX);
	}
}

template <class T>
void ReadCSV(Tensor<T>* buff, std::string filename)
{
	unsigned i = 0;
	std::ifstream  data(filename.c_str());
	std::string line;
	while(std::getline(data,line))
	{
		if (i > buff->num_elements()) {
			throw std::runtime_error("Buffer size is less than number of elements in CSV-file");
		}
		std::stringstream  line_stream(line);
		std::string        cell;
		while(std::getline(line_stream,cell,','))
		{
			std::stringstream  cell_stream(cell);
			T val;
			cell_stream>>val;
			buff->data()[i] = val;
			i++;
		}
	}
}



template <class T>
void AssertNearTensors(const Tensor<T>& host, const TensorGPU<T>& gpu, std::string message)
{
	//Check if both null
	if(host.data() == NULL && gpu.data() == NULL ) return;
	Tensor<T> gpu_host = gpu;
	ASSERT_EQ(host.num_elements(), gpu_host.num_elements())<<"Tensors have different number of elements";
	for (UINT i = 0; i < host.num_elements(); ++i) {
		ASSERT_NEAR(host[i],gpu_host[i], GPU_FLOAT_TOLLERANCE)<<message<<"Host and CUDA tensors have different values at i = "<<i;
	}
}


class CUDATest : public ::testing::Test {
protected:	
	cudaDeviceProp deviceProp;
	virtual void SetUp()
	{
		int deviceCount;
		cudaGetDeviceCount(&deviceCount);
		ASSERT_TRUE(deviceCount > 0) <<"No CUDA graphics card \n";
		cudaGetDeviceProperties(&deviceProp, 0);
		ASSERT_TRUE(deviceProp.major >= 1)<<"Not enough CUDA compute capability version \n";
	}
	virtual void TearDown()
	{
	}
};

// Predicates

//Predicates
class Predicates
{
public:
    static bool SameOrderAndSign(double val1, double val2)
    {
        //Check sign
        if(val1*val2 < 0) return false;
        //Check order
        if(val1/val2 > 10 || val1/val2 < 0.1) return false;
        return true;
    }

};

#endif