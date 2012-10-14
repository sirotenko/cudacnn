#include "precomp.h"


//TODO: Make these test independent from hdf5
#ifdef HAVE_HDF5
class CNNTrainTest : public ::testing::Test {
protected:
	CNNetCudaF cnnet_gpu;
	CNNetF cnnet_host;	
	TensorFloat e_host, e_gpu; 
	TensorFloat input_host; TensorGPUFloat input_gpu;
	TensorFloat target_out;
	MSEFunction<float> performance_function;
	virtual void SetUp()
	{
		std::string test_file = "test_mnist_net.h5";
		cnnet_gpu.LoadFromFile(test_file);
		cnnet_host.LoadFromFile(test_file);
		input_host = TensorFloat(32,32,1);
		input_gpu = TensorFloat(32,32,1);
		ReadCSV(&input_host, "digit_image_preproc.csv");
		input_gpu = input_host;

		target_out = TensorFloat(10,1,1);
		float out[] = {-1.f, -1.f, -1.f, -1.f, -1.f, 1.f, -1.f, -1.f, -1.f, -1.f};
		memcpy(target_out.data(), out, sizeof(float)*10);
		cnnet_host.Sim(input_host);
		cnnet_gpu.Sim(input_gpu);
		e_host = cnnet_host.out() - target_out;
		TensorFloat out_gpu(cnnet_gpu.out());
		for(UINT i = 0; i < out_gpu.num_elements(); ++i){
			ASSERT_NEAR(out_gpu[i],cnnet_host.out()[i],GPU_FLOAT_TOLLERANCE)<<"GPU and CPU network outputs for same input is different at out "<<i<<std::endl;
		}
		e_gpu = out_gpu - target_out;
		cnnet_gpu.PrepareForTraining();
		cnnet_host.PrepareForTraining();
	}
	virtual void TearDown()
	{
	}
};
//
TEST_F(CNNTrainTest, TestGradientDescent)
{
	//Step by step backpropagation
	//Performance function
	TensorFloat dedx_host(performance_function.dydx(e_host));
	TensorFloat dedx_gpu(performance_function.dydx(e_gpu));
	
	size_t last_layer = cnnet_gpu.nlayers() - 1;
	cnnet_gpu[last_layer]->set_de_dx(dedx_gpu);
	cnnet_host[last_layer]->set_de_dx(dedx_host);

	std::stringstream ss;	
	for(size_t i = cnnet_gpu.nlayers() - 1; i > 0; --i ) {
		cnnet_host[i]->BackPropagate(cnnet_host[i-1]->out(), cnnet_host[i-1]->de_dx());
		cnnet_gpu[i]->BackPropagate(cnnet_gpu[i-1]->out(), cnnet_gpu[i-1]->de_dx());
		ss.str("");ss<<"de_dx check failed."<<std::endl<<"Layer N "<<i<<std::endl;
		AssertNearTensors(cnnet_host[i]->de_dx(),cnnet_gpu[i]->de_dx(), ss.str());
		ss.str("");ss<<"de_dw check failed."<<std::endl<<"Layer N "<<i<<std::endl;
		AssertNearTensors(cnnet_host[i]->de_dw(),cnnet_gpu[i]->de_dw(), ss.str());
		ss.str("");ss<<"de_db check failed."<<std::endl<<"Layer N "<<i<<std::endl;
		AssertNearTensors(cnnet_host[i]->de_db(),cnnet_gpu[i]->de_db(), ss.str());
	}
	cnnet_host[0]->ComputeGradient(input_host);
	cnnet_gpu[0]->ComputeGradient(input_gpu);
	AssertNearTensors(cnnet_host[0]->de_dw(),cnnet_gpu[0]->de_dw(), "de_dw check failed. ");
	AssertNearTensors(cnnet_host[0]->de_db(),cnnet_gpu[0]->de_db(), "de_db check failed. ");
	AssertNearTensors(cnnet_host[0]->de_dx(),cnnet_gpu[0]->de_dx(), "de_dx check failed. ");

}

TEST_F(CNNTrainTest, TestBackpropHighLevel)
{
	//Step by step backpropagation
	//Performance function
	cnnet_host.BackpropGradients(e_host, input_host);
	cnnet_gpu.BackpropGradients(e_gpu, input_gpu);
	cnnet_host.ResetHessian(); 
	cnnet_gpu.ResetHessian();
	cnnet_host.AccumulateHessian(e_host, input_host);
	cnnet_gpu.AccumulateHessian(e_gpu, input_gpu);
    CNNetCudaF::const_iterator it_gpu;
    CNNetF::const_iterator it_host;
    ASSERT_EQ(cnnet_gpu.nlayers(),cnnet_host.nlayers());
	//for(size_t i = cnnet_gpu.nlayers() - 1; i >= 0; --i ) {
    for(it_gpu = cnnet_gpu.begin(), it_host = cnnet_host.begin(); 
        it_gpu != cnnet_gpu.end(); ++it_gpu, ++it_host){

		//First derriv check
        AssertNearTensors((*it_host)->de_dx(),(*it_gpu)->de_dx(), "de_dx check failed. ");
        AssertNearTensors((*it_host)->de_dw(),(*it_gpu)->de_dw(), "de_dw check failed. ");
        AssertNearTensors((*it_host)->de_db(),(*it_gpu)->de_db(), "de_db check failed. ");

		//2nd derriv check
        AssertNearTensors((*it_host)->d2e_dx2(),(*it_gpu)->d2e_dx2(), "d2e_dx2 check failed. ");
        AssertNearTensors((*it_host)->d2e_dw2(),(*it_gpu)->d2e_dw2(), "d2e_dw2 check failed. ");
        AssertNearTensors((*it_host)->d2e_db2(),(*it_gpu)->d2e_db2(), "d2e_db2 check failed. ");

	}
}

TEST_F(CNNTrainTest, DISABLED_TestHessianAveraging)
{
	cnnet_host.ResetHessian(); 
	cnnet_gpu.ResetHessian();
	cnnet_gpu.AccumulateHessian(e_gpu, input_gpu);
	for(int i = 0; i < 7; ++i){
		cnnet_host.AccumulateHessian(e_host, input_host);
	}
	cnnet_host.AverageHessian();

	for(size_t i = cnnet_gpu.nlayers() - 1; i >= 0; --i ) {
		//Should be equal between each other
		std::stringstream ss;
		ss<<"d2e_dw2 check failed."<<std::endl<<"Layer N "<<i<<std::endl;
		AssertNearTensors(cnnet_host[i]->d2e_dw2(),cnnet_gpu[i]->d2e_dw2(), ss.str());
		ss<<"d2e_db2 check failed."<<std::endl<<"Layer N "<<i<<std::endl;
		AssertNearTensors(cnnet_host[i]->d2e_db2(),cnnet_gpu[i]->d2e_db2(), ss.str());
	}

	cnnet_host.ResetHessian(); 
	cnnet_gpu.ResetHessian();
	cnnet_host.AccumulateHessian(e_host, input_host);
	for(int i = 0; i < 8; ++i){
		cnnet_gpu.AccumulateHessian(e_gpu, input_gpu);
	}
	cnnet_gpu.AverageHessian();

	for(int i = 0; i < cnnet_gpu.nlayers(); ++i ) {
		//Should be equal between each other
		std::stringstream ss;
		ss<<"d2e_dw2 check failed."<<std::endl<<"Layer N "<<i<<std::endl;
		AssertNearTensors(cnnet_host[i]->d2e_dw2(),cnnet_gpu[i]->d2e_dw2(), ss.str());
		ss<<"d2e_db2 check failed."<<std::endl<<"Layer N "<<i<<std::endl;
		AssertNearTensors(cnnet_host[i]->d2e_db2(),cnnet_gpu[i]->d2e_db2(), ss.str());
	}

}
//
class CNNTrainTestNumeric : public ::testing::Test {
protected:
	CNNetD cnnet_host;	
	TensorDouble e_host, e_gpu; 
	TensorDouble input_host; 
	TensorDouble target_out;
	MSEFunction<double> performance_function;
	virtual void SetUp()
	{
		cnnet_host.LoadFromFile("test_mnist_net.h5");
		input_host = TensorDouble(32,32,1);
		ReadCSV(&input_host, "digit_image_preproc.csv");
		target_out = TensorDouble(10,1,1);
		double out[] = {-1.f, -1.f, -1.f, -1.f, -1.f, 1.f, -1.f, -1.f, -1.f, -1.f};
		memcpy(target_out.data(), out, sizeof(double)*10);
		cnnet_host.Sim(input_host);
		e_host = cnnet_host.out() - target_out;
		cnnet_host.PrepareForTraining();

	}
	virtual void TearDown()
	{
	}
};
//

TEST_F(CNNTrainTestNumeric, TestGradientsNumeric)
{
	double eps = 0.00001;
	//Compute gradients
	cnnet_host.InitWeights(&RandomWeightInit);
	cnnet_host.Sim(input_host);
	e_host = performance_function.dydx(cnnet_host.out() - target_out);
	cnnet_host.BackpropGradients(e_host, input_host);

    CNNetD::const_reverse_iterator it;
    size_t i = 0;
	//for(size_t i = cnnet_host.nlayers() - 1; i >=0; --i){
    for(it = cnnet_host.rbegin(), i = cnnet_host.nlayers()-1; it != cnnet_host.rend(); ++it, --i){
		for(int t = 0; t < 5; ++t ) {
			if((*it)->de_dw().num_elements() < 1 ) continue;
			//Pick up some random weight
			int nweights = (*it)->weights().num_elements();
			int weight_idx = int((nweights - 1)*(float(rand())/RAND_MAX));

			double grad_comp = (*it)->de_dw()[weight_idx];
			//Now change weight by eps and compute cost=
			//(*it)->weights()[weight_idx] -= eps;
			TensorDouble* weights_mutable = const_cast<TensorDouble*>(&((*it)->weights()));
			(*weights_mutable)[weight_idx] -= eps;
			cnnet_host.Sim(input_host);
			e_host = cnnet_host.out() - target_out;
			double loss_1 = performance_function(e_host);
			(*weights_mutable)[weight_idx] += 2*eps;
			cnnet_host.Sim(input_host);
			e_host = cnnet_host.out() - target_out;
			double loss_2 = performance_function(e_host);

			double numeric_grad = (loss_2 - loss_1)/(2*eps);
			//Get back weight 
			(*weights_mutable)[weight_idx] -= eps;
			//EXPECT_NEAR(numeric_grad, grad_comp, abs(0.1*numeric_grad))<<"Weight gradient for "<<i<<" layer don't match to numeric"<<std::endl;
			EXPECT_NEAR(numeric_grad, grad_comp, 10*eps)<<"Weight gradient for "<<i<<" layer don't match to numeric"<<std::endl;

			//========= Check bias =============
			if((*it)->de_db().num_elements() < 1 ) continue;
			int nbiases = (*it)->biases().num_elements();
			int bias_idx = int((nbiases - 1)*(float(rand())/RAND_MAX));
			grad_comp = (*it)->de_db()[bias_idx];
			//Now change weight by eps and compute cost=
			weights_mutable = const_cast<TensorDouble*>(&((*it)->biases()));
			(*weights_mutable)[bias_idx] -= eps;
			cnnet_host.Sim(input_host);
			loss_1 = performance_function(cnnet_host.out() - target_out);
			(*weights_mutable)[bias_idx] += 2*eps;
			cnnet_host.Sim(input_host);
			loss_2 = performance_function(cnnet_host.out() - target_out);

			numeric_grad = (loss_2 - loss_1)/(2*eps);
			//Get back bias
			(*weights_mutable)[bias_idx] -= eps;
			//EXPECT_NEAR(numeric_grad, grad_comp, abs(0.1*numeric_grad))<<"Bias gradient for "<<i<<" layer don't match to numeric"<<std::endl;
			EXPECT_NEAR(numeric_grad, grad_comp, 10*eps)<<"Bias gradient for "<<i<<" layer don't match to numeric"<<std::endl;
		}
	}
}

#endif  //HAVE_HDF5