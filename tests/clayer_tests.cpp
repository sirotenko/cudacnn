#include "tests_common.h"


class CLayerPropagateTest : public ::testing::Test {
protected:
	//CLayerCudaFTS clayer;
	TensorGPUFloat weights, biases, inputs;
	TensorGPUInt conn_map;
	virtual void SetUp()
	{
		int deviceCount;
		cudaGetDeviceCount(&deviceCount);
		ASSERT_TRUE(deviceCount > 0);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, 0);
		ASSERT_TRUE(deviceProp.major >= 1);
		std::vector<UINT> wdims;
		wdims.push_back(7); wdims.push_back(7);
		wdims.push_back(3); wdims.push_back(4);

		weights = TensorGPUFloat(wdims);
		biases = TensorGPUFloat(1, 1, 4);
		inputs = TensorGPUFloat(20+(7-1), 20+(7-1), 3);
		std::vector<UINT> dims(2);
		dims[0] = inputs.d();
		dims[1] = weights.d2();
		conn_map = TensorGPUInt(dims);
	}
	virtual void TearDown()
	{
	}
};

TEST_F(CLayerPropagateTest, PropagateZeros)
{
	//Zero inputs, weights, biases, connections
	TensorFloat inputsHost = inputs;
	for (unsigned i = 0; i < inputsHost.num_elements(); i++) {
		EXPECT_EQ(inputsHost[i], 0);
	}
	Tensor<int> connHost(conn_map.dims());
	for (unsigned i = 0; i < connHost.num_elements(); i++) {
		connHost[i] = 1;
	}
	conn_map = connHost;
	CLayerCudaFTS clayer(inputs.w(), inputs.h(), weights, biases, conn_map );

	clayer.Propagate(inputs);
	TensorFloat outHost;
	outHost = clayer.out();
	for (unsigned i = 0; i < outHost.num_elements(); i++) {
		ASSERT_EQ(outHost[i], 0);
	}
}

TEST_F(CLayerPropagateTest, BackPropZeros)
{
	TensorFloat inputsHost;
	inputsHost = inputs;
	for (unsigned i = 0; i < inputsHost.num_elements(); i++) {
		EXPECT_EQ(inputsHost[i], 0);
	}
	Tensor<int> connHost(conn_map.dims());
	for (unsigned i = 0; i < connHost.num_elements(); i++) {
		connHost[i] = 1;
	}
	conn_map = connHost;
	CLayerCudaFTS clayer(inputs.w(), inputs.h(), weights, biases, conn_map );
    TensorGPUFloat de_dx_prev_gpu(inputs);
	clayer.PrepareForTraining();
	clayer.BackPropagate(inputs, de_dx_prev_gpu);
    TensorFloat de_dx_prev = de_dx_prev_gpu;
	for(unsigned i = 0; i < de_dx_prev.num_elements(); ++i){
		ASSERT_NEAR(de_dx_prev[i], 0.f, std::numeric_limits<float>::epsilon());
	}
	TensorFloat de_dw = clayer.de_dw();
	for(unsigned i = 0; i < de_dw.num_elements(); ++i){
		ASSERT_NEAR(de_dw[i], 0.f, std::numeric_limits<float>::epsilon());
	}
	TensorFloat de_db = clayer.de_db();
	for(unsigned i = 0; i < de_db.num_elements(); ++i){
		ASSERT_NEAR(de_db[i], 0.f, std::numeric_limits<float>::epsilon());
	}

}



TEST_F(CLayerPropagateTest, PropagateOnes)
{
	//Zero inputs, weights, biases, connections
	TensorFloat inputsHost(inputs.dims());
	TensorFloat weightsHost(weights.dims());
	Tensor<int> connHost(conn_map.dims());

	for (unsigned i = 0; i < inputsHost.num_elements(); i++) {
		inputsHost[i] = 1.f;
	}
	for (unsigned i = 0; i < weightsHost.num_elements(); i++) {
		weightsHost[i] = 1.f;
	}
	for (unsigned i = 0; i < connHost.num_elements(); i++) {
		connHost[i] = 1;
	}

	inputs = inputsHost;
	weights = weightsHost;
	conn_map = connHost;
	CLayerCudaFTS clayer(inputs.w(), inputs.h(), weights, biases, conn_map );

	clayer.Propagate(inputs);
	TensorFloat out_cuda_host;
	out_cuda_host = clayer.out();
	TansigMod<float> transfer_func;
	float true_out = transfer_func(float(weightsHost.w()*weightsHost.h()*inputs.d()));
	for (unsigned i = 0; i < out_cuda_host.num_elements(); i++) {
		//EXPECT_EQ(outHost[i], true_out);
		ASSERT_EQ(out_cuda_host[i], true_out)<< "outHost is wrong at index " << i;
	}

}

class CLayerPropagateRandomTest : public CLayerPropagateTest {
protected:
	virtual void SetUp()
	{
		int deviceCount;
		cudaGetDeviceCount(&deviceCount);
		ASSERT_TRUE(deviceCount > 0);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, 0);
		ASSERT_TRUE(deviceProp.major >= 1);

		TensorFloat inputsHost(32, 32, 2);
		//ReadCSV(&inputsHost, "clayer_rand_inp.csv");
		inputsHost = TensorFloat::Rand(inputsHost.dims(), 1.f);

		std::vector<UINT> wdims;
		wdims.push_back(5); wdims.push_back(5);
		wdims.push_back(inputsHost.d()); wdims.push_back(3);

		TensorFloat weightsHost(wdims);
		weightsHost = TensorFloat::Rand(weightsHost.dims(), 1.f);
		//ReadCSV(&weightsHost,"clayer_weights.csv");

		TensorFloat biasesHost(1, 1, weightsHost.d2());
		biasesHost = TensorFloat::Rand(biasesHost.dims(), 1.f);
		//ReadCSV(&biasesHost,"clayer_biases.csv");
		
		std::vector<unsigned> dims(2);
		dims[0] = weightsHost.d();
		dims[1] = weightsHost.d2();
		Tensor<int> connHost = (dims);
		connHost = Tensor<int>::Ones(connHost.dims());
		//ReadCSV(&connHost,"clayer_con_map.csv");

		inputs = inputsHost;
		weights = weightsHost;
		conn_map = connHost;
		biases = biasesHost;
	
	}
	virtual void TearDown()
	{
	}

};

TEST_F(CLayerPropagateRandomTest, PropagateRandom)
{
	CLayerCudaFTS clayer(32, 32, weights, biases, conn_map );
	clayer.Propagate(inputs);
	TensorFloat outHost = clayer.out();
	CLayerFTS clayer_host(32, 32, weights, biases, conn_map ); 
	
	clayer_host.Propagate(inputs);
	TensorFloat true_out = clayer_host.out();
	///ReadCSV(&true_out, "clayer_outs.csv");
	//TansigMod<float> transfer_func;
	for (unsigned i = 0; i < outHost.num_elements(); i++) {
		//EXPECT_EQ(outHost[i], true_out);
		//ASSERT_FLOAT_EQ(outHost[i], true_out[i])<< "outHost is wrong at index " << i;
		ASSERT_NEAR(outHost[i], true_out[i], 10*FLT_EPSILON)<< "outHost is wrong at index " << i;
	}
}

