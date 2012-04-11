#include "tests_common.h"

TEST(FLayerTest, FLayerPropagateRandom)
{
	//Zero inputs, weights, biases, connections
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	ASSERT_TRUE(deviceCount > 0) <<"No CUDA graphics card \n";
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	ASSERT_TRUE(deviceProp.major >= 1)<<"Not enough CUDA compute capability version \n";
	TensorGPUFloat inputs(120, 1, 1);
	TensorGPUFloat weights(84,120,1);
	TensorGPUFloat biases(84,1,1);
	TensorFloat inputsHost = inputs;
	TensorFloat weightsHost = weights;
	TensorFloat biasesHost = biases;

	ReadCSV(&inputsHost, "flayer_inputs.csv");
	ReadCSV(&weightsHost, "flayer_weights.csv");
	ReadCSV(&biasesHost, "flayer_biases.csv");

	weights = weightsHost;
	biases = biasesHost;
	inputs = inputsHost;
	FLayerCudaFTS flayer(weights, biases);
	flayer.Propagate(inputs);
	//outsHost.Init(flayer.out().w(), flayer.out().h(), flayer.out().m());
	TensorFloat outsHost = flayer.out();
	TensorFloat true_out(outsHost);
	ReadCSV(&true_out, "flayer_outputs.csv");
	for (unsigned i = 0; i < outsHost.num_elements(); i++) {
		//EXPECT_EQ(outHost[i], true_out);
		TansigMod<float> tf;
		//Purelin<float> tf;
		//Square<float> tf;
		//Tansig<float> tf;
		//ASSERT_FLOAT_EQ(tf(true_out[i]), outsHost[i])<< "outHost is wrong at index " << i;

		ASSERT_NEAR(tf(true_out[i]), outsHost[i],0.000001)<< "outHost is wrong at index " << i;
	}

}
