#include "precomp.h"

class PLayerPropagateTest : public CUDATest {
protected:
	TensorGPUFloat inputs, weights, biases;
	virtual void SetUp()
	{
		CUDATest::SetUp();
	}
	virtual void TearDown()
	{

	}
};

TEST_F(PLayerPropagateTest, PropagateOnes)
{
	//Zero inputs, weights, biases, connections
	inputs = TensorFloat(40, 40, 5);
	weights = TensorFloat(1,1,5);
	biases = TensorFloat(1,1,5);
	TensorFloat inputsHost(inputs.w(),inputs.h(), inputs.d());
	for (unsigned i = 0; i < inputsHost.num_elements(); ++i)	{
		inputsHost[i] = 1;
	}
	TensorFloat weightsHost(weights.w(),weights.h(), weights.d());
	for (unsigned i = 0; i < weightsHost.num_elements(); ++i)	{
		weightsHost[i] = 1;
	}

	TensorFloat biasesHost(biases.w(),biases.h(), biases.d());
	for (unsigned i = 0; i < biasesHost.num_elements(); ++i)	{
		biasesHost[i] = 0;
	}
	inputs = inputsHost;
	weights = weightsHost;
	biases = biasesHost;
	PoolingLayerT<TensorGPU, float>::Params params_gpu;
	params_gpu.inp_width = inputs.w();
	params_gpu.inp_height = inputs.h();
	params_gpu.ninputs = 5;
	params_gpu.sx = 2;
	params_gpu.sy = 2;
    params_gpu.pooling_type = PoolingLayerT<TensorGPU, float>::eAverage;
	PoolingLayer<TensorGPU, float> player_gpu(params_gpu);	
	player_gpu.Propagate(inputs);

	PoolingLayerT<Tensor, float>::Params params_host;
	params_host.inp_width = inputs.w();
	params_host.inp_height = inputs.h();
	params_host.ninputs = 5;
	params_host.sx = 2;
	params_host.sy = 2;
    params_host.pooling_type = PoolingLayerT<Tensor, float>::eAverage;

	PoolingLayer<Tensor, float> player_host(params_host);
	player_host.Propagate(inputsHost);
	AssertNearTensors(player_host.out(), player_gpu.out(), "Error in slayer random propagation");

}

TEST_F(PLayerPropagateTest, PropagateRandomSimple)
{
	//Zero inputs, weights, biases, connections
	inputs = TensorFloat(40, 40, 5);
	weights = TensorFloat(1,1,5);
	biases = TensorFloat(1,1,5);
	TensorFloat inputsHost(inputs.w(),inputs.h(), inputs.d());
	for (unsigned i = 0; i < inputsHost.num_elements(); ++i)	{
		inputsHost[i] = float(rand())/RAND_MAX;
	}
	TensorFloat weightsHost(weights.w(),weights.h(), weights.d());
	for (unsigned i = 0; i < weightsHost.num_elements(); ++i)	{
		weightsHost[i] = float(rand())/RAND_MAX;
	}

	TensorFloat biasesHost(biases.w(),biases.h(), biases.d());
	for (unsigned i = 0; i < biasesHost.num_elements(); ++i)	{
		biasesHost[i] = float(rand())/RAND_MAX;
	}

	inputs = inputsHost;
	weights = weightsHost;
	biases = biasesHost;
	PoolingLayerT<TensorGPU, float>::Params params_gpu;
	params_gpu.inp_width = inputs.w();
	params_gpu.inp_height = inputs.h();
	params_gpu.ninputs = 5;
	params_gpu.sx = 2;
	params_gpu.sy = 2;
    params_gpu.pooling_type = PoolingLayerT<TensorGPU, float>::eAverage;

	PoolingLayer<TensorGPU, float> player_gpu(params_gpu);	
	player_gpu.Propagate(inputs);

	PoolingLayerT<Tensor, float>::Params params_host;
	params_host.inp_width = inputs.w();
	params_host.inp_height = inputs.h();
	params_host.ninputs = 5;
	params_host.sx = 2;
	params_host.sy = 2;
    params_host.pooling_type = PoolingLayerT<Tensor, float>::eAverage;

	PoolingLayer<Tensor, float> player_host(params_host);
	player_host.Propagate(inputsHost);
	AssertNearTensors(player_host.out(), player_gpu.out(), "Error in slayer random propagation");

}


TEST_F(PLayerPropagateTest, PropagateRandomIrregularAverage)
{
	//Zero inputs, weights, biases, connections
	inputs = TensorFloat(40, 12, 5);
	TensorFloat inputsHost(inputs.w(),inputs.h(), inputs.d());
	for (unsigned i = 0; i < inputsHost.num_elements(); ++i)	{
		inputsHost[i] = float(rand())/RAND_MAX;
	}
	TensorFloat weightsHost(weights.w(),weights.h(), weights.d());
	for (unsigned i = 0; i < weightsHost.num_elements(); ++i)	{
		weightsHost[i] = float(rand())/RAND_MAX;
	}

	TensorFloat biasesHost(biases.w(),biases.h(), biases.d());
	for (unsigned i = 0; i < biasesHost.num_elements(); ++i)	{
		biasesHost[i] = float(rand())/RAND_MAX;
	}
	inputs = inputsHost;
	weights = weightsHost;
	biases = biasesHost;
	PoolingLayerT<TensorGPU, float>::Params params_gpu;
	params_gpu.inp_width = inputs.w();
	params_gpu.inp_height = inputs.h();
	params_gpu.ninputs = 5;
	params_gpu.sx = 3;
	params_gpu.sy = 1;
    params_gpu.pooling_type = PoolingLayerT<TensorGPU, float>::eAverage;


	PoolingLayer<TensorGPU, float> player_gpu(params_gpu);	
	player_gpu.Propagate(inputs);

	PoolingLayerT<Tensor, float>::Params params_host;
	params_host.inp_width = inputs.w();
	params_host.inp_height = inputs.h();
	params_host.ninputs = 5;
	params_host.sx = 3;
	params_host.sy = 1;
    params_host.pooling_type = PoolingLayerT<Tensor, float>::eAverage;

	PoolingLayer<Tensor, float> player_host(params_host);
	player_host.Propagate(inputsHost);
	AssertNearTensors(player_host.out(), player_gpu.out(), "Error in slayer random propagation");
}

TEST_F(PLayerPropagateTest, PropagateRandomIrregularMax)
{
    //Zero inputs, weights, biases, connections
    inputs = TensorFloat(40, 12, 5);
    TensorFloat inputsHost(inputs.w(),inputs.h(), inputs.d());
    for (unsigned i = 0; i < inputsHost.num_elements(); ++i)	{
        inputsHost[i] = float(rand())/RAND_MAX;
    }
    TensorFloat weightsHost(weights.w(),weights.h(), weights.d());
    for (unsigned i = 0; i < weightsHost.num_elements(); ++i)	{
        weightsHost[i] = float(rand())/RAND_MAX;
    }

    TensorFloat biasesHost(biases.w(),biases.h(), biases.d());
    for (unsigned i = 0; i < biasesHost.num_elements(); ++i)	{
        biasesHost[i] = float(rand())/RAND_MAX;
    }
    inputs = inputsHost;
    weights = weightsHost;
    biases = biasesHost;
    PoolingLayerT<TensorGPU, float>::Params params_gpu;
    params_gpu.inp_width = inputs.w();
    params_gpu.inp_height = inputs.h();
    params_gpu.ninputs = 5;
    params_gpu.sx = 3;
    params_gpu.sy = 2;
    params_gpu.pooling_type = PoolingLayerT<TensorGPU, float>::eMax;

    PoolingLayer<TensorGPU, float> player_gpu(params_gpu);	
    player_gpu.Propagate(inputs);

    PoolingLayerT<Tensor, float>::Params params_host;
    params_host.inp_width = inputs.w();
    params_host.inp_height = inputs.h();
    params_host.ninputs = 5;
    params_host.sx = 3;
    params_host.sy = 2;
    params_host.pooling_type = PoolingLayerT<Tensor, float>::eMax;

    PoolingLayer<Tensor, float> player_host(params_host);
    player_host.Propagate(inputsHost);
    AssertNearTensors(player_host.out(), player_gpu.out(), "Error in slayer random propagation");
}

