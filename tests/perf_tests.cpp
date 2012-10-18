//Copyright (c) 2012, Mikhail Sirotenko <mihail.sirotenko@gmail.com>
//All rights reserved.
//
//Redistribution and use in source and binary forms, with or without
//modification, are permitted provided that the following conditions are met:
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//
//THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
//ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
//DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
//ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/**
 * @file	perf_tests.cpp
 * @brief	This files contains set of performance tests 
 *
 * @brief	Pefrormance tests are used to measure speed of a certain function 
 * @version	$Revision$
 * @author	Mikhail Sirotenko
 */

#include "precomp.h"
#ifdef HAVE_CUDA
#ifdef HAVE_HDF5
using namespace cudacnn;
class CLayerPerfTest : public CUDATest {
protected:
    CNNet<TensorGPU, float>* convnet;
	TensorGPUFloat weights, biases, inputs, outputs;
	TensorGPUInt conn_map;
	TensorFloat inputsHost, weightsHost, biasesHost, outputsHost;
	TensorInt connHost;

    virtual void InitRandom()
    {
        for (unsigned i = 0; i < inputsHost.num_elements(); i++) {
			inputsHost[i] = float(std::rand()) / RAND_MAX;
		}

		for (unsigned i = 0; i < weightsHost.num_elements(); i++) {
			weightsHost[i] = float(std::rand()) / RAND_MAX;
		}
		for (unsigned i = 0; i < biasesHost.num_elements(); i++) {
			biasesHost[i] = float(std::rand()) / RAND_MAX;
		}
		for (unsigned i = 0; i < outputsHost.num_elements(); i++) {
			outputsHost[i] = 0.f;
		}
		for (unsigned i = 0; i < connHost.num_elements(); i++) {
			connHost[i] = 1;
		}

		inputs = inputsHost;
		weights = weightsHost;
		biases = biasesHost;
		outputs = outputsHost;
        conn_map = connHost;
    }

	virtual void SetUp()
	{
        const int inp_width = 32;
        const int inp_height = 32;
        const bool is_trainable = false;

		CUDATest::SetUp();
        inputs = TensorGPUFloat(32, 32, 3);		
        inputsHost = inputs;		
        convnet = new CNNet<TensorGPU, float>();
        convnet->LoadFromFile("cnet32.h5");
        outputs = convnet->out();
	}
	virtual void TearDown()
	{
        delete convnet;
	}
};

TEST_F(CLayerPerfTest, DISABLED_TestFprop)
{
    InitRandom();
    //Simply loop for n iterations
    const int its = 4000;
    for(int i = 0; i < its; ++i){
        //Emulate loading data into sliding window
        inputs = inputsHost;		
        convnet->Sim(inputs);        
        outputsHost = convnet->out();
    }
    
}


#endif //HAVE_HDF5
#endif //HAVE_CUDA