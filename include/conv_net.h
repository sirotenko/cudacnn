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

#ifndef _CONV_NET_H
#define _CONV_NET_H

namespace cudacnn
{

template< template<class> class T, class ET>
class CNNet
{

public:
#ifdef HAVE_BOOST    
    typedef boost::shared_ptr<Layer<T,ET> > LayerPtr;
#else
    typedef std::shared_ptr<Layer<T,ET> > LayerPtr;
#endif //HAVE_BOOST    

    typedef typename std::vector<LayerPtr>::iterator iterator;
    typedef typename std::vector<LayerPtr>::const_iterator const_iterator;
    typedef typename std::vector<LayerPtr>::reverse_iterator reverse_iterator;
    typedef typename std::vector<LayerPtr>::const_reverse_iterator const_reverse_iterator;

    CNNet():layers_(0), ninputs_(0), input_width_(0), input_height_(0)	{	}
    int Init(std::vector<LayerPtr>& layers, int ninputs, int inp_width, int inp_height);
    
//=========================== Network utility methods
//If hdf5 library unavailable file read and save operations will not work
//This can be the case if Matlab is used for saving and loading nnet
#ifdef HAVE_HDF5
	int LoadFromFile(const std::string& filename);
	int SaveToFile(const std::string& filename);
#endif
//=========================== Simulation
	//Perform propagation of inputs from the input layer to the output layer
	//Network must be initialized and inputs loaded
	void Sim(const T<ET>& inputs);
//=========================== Weights initialization

	void InitWeights(void(*weight_init_func)(T<ET>&) );	
//=========================== Train methods
	void PrepareForTraining();
	//Backpropagate error and update weights
	void BackpropGradients(const T<ET>& error, const T<ET>& input);
	//Apply calculated gradients to weigts with specified teta
	//Network must be initialized and gradients calculated
	void AdaptWeights(ET tau, bool use_hessian, ET mu);
	void ResetHessian();

	// Calculate and accumulate second derrivatives of errors by weights
	void AccumulateHessian(const T<ET>& error, const T<ET>& input);
	void AverageHessian();
	//Calculate second derrivative of performance function
	void ComputePerfSecondDerriv(void);
//=========================== Getters  
    const T<ET>& out() const { return layers_.back()->out() ;}
	size_t nlayers() const { return layers_.size(); };
	UINT ninputs() const { return ninputs_; };
	UINT input_width() const { return input_width_; };
	UINT input_height() const { return input_height_; };
    
    LayerPtr operator[](size_t idx)
    {
        assert(idx < nlayers() && idx >= 0);
        return layers_[idx];
    }
    iterator       begin() {return layers_.begin();}
    const_iterator begin() const {return layers_.begin();}

    iterator       end() {return layers_.end();}
    const_iterator end() const {return layers_.end();}

    reverse_iterator       rbegin() {return layers_.rbegin();}
    const_reverse_iterator rbegin() const {return layers_.rbegin();}

    reverse_iterator       rend() {return layers_.rend();}
    const_reverse_iterator rend() const {return layers_.rend();}

protected:
    std::vector<LayerPtr> layers_;
	BYTE ninputs_;
	UINT input_width_;
	UINT input_height_;

};

template< template<class> class T, class ET>
int CNNet<T,ET>::Init(std::vector<LayerPtr>& layers, int ninputs, int inp_width, int inp_height)
{
	//nlayers_ = nlayers; 
	ninputs_ = ninputs;
	input_width_ = inp_width;
	input_height_ = inp_height;
	layers_ = layers;
	return 0;

}


//=========================== Network utility methods
#ifdef HAVE_HDF5

template< template<class> class T, class ET>
int CNNet<T, ET>::LoadFromFile(const std::string& filename)
{
	try
	{
		H5File file( filename, H5F_ACC_RDONLY );
		
		Group rootGroup = file.openGroup(ROOT_LAYERS_GROUP_NAME);
		double ver = hdf5Helper::ReadScalar<double>(rootGroup,"Version");
		if(ver != __CNN_FILE_VERSION)
			throw std::runtime_error("Unsupported version of CNN file");

		size_t nlayers_ = hdf5Helper::ReadIntAttribute(rootGroup, "nlayers");
		ninputs_ = hdf5Helper::ReadIntAttribute(rootGroup, "nInputs");
		input_width_ = hdf5Helper::ReadIntAttribute(rootGroup, "inputWidth");
		input_height_ = hdf5Helper::ReadIntAttribute(rootGroup, "inputHeight");

		//Impossible number of layers
		assert(nlayers_ < INT_MAX);

        layers_.clear();

		for (int i = 0; i < nlayers_; i++)
		{
			std::stringstream group_name;
			group_name<<"Layer"<<i+1;
			Group layerGroup = rootGroup.openGroup(group_name.str());
			Layer<T, ET>::eLayerType lt = (Layer<T, ET>::eLayerType)hdf5Helper::ReadIntAttribute(layerGroup, "LayerType");
			switch(lt){
				case Layer<T, ET>::ePLayer:
					{
						int num_fmaps = hdf5Helper::ReadIntAttribute(layerGroup, "NumFMaps");
						int inp_width = hdf5Helper::ReadIntAttribute(layerGroup, "InpWidth");
						int inp_height = hdf5Helper::ReadIntAttribute(layerGroup, "InpHeight");

						int sx = hdf5Helper::ReadIntAttribute(layerGroup, "SXRate");
						int sy = hdf5Helper::ReadIntAttribute(layerGroup, "SYRate");

                        PoolingLayerT<T, ET>::Params params;
						params.inp_width = inp_width;
						params.inp_height = inp_height;
						params.ninputs = num_fmaps;
						params.sx = sx;
						params.sy = sy;
                        params.pooling_type = static_cast<PoolingLayerT<T, ET>::PoolingType>(hdf5Helper::ReadIntAttribute(layerGroup, "PoolingType"));

                        layers_.push_back(LayerPtr(new PoolingLayer<T,ET>(params)));
						break;
					}
				case Layer<T, ET>::eCLayer:
					{
						bool is_trainable = hdf5Helper::ReadIntAttribute(layerGroup, "Trainable") == 1 ? true : false;
						int inp_width = hdf5Helper::ReadIntAttribute(layerGroup, "InpWidth");
						int inp_height = hdf5Helper::ReadIntAttribute(layerGroup, "InpHeight");
						eTransfFunc tf = (eTransfFunc)hdf5Helper::ReadIntAttribute(layerGroup, "TransferFunc");
						Tensor<ET> weights, biases;
						TensorInt conn_map;
						hdf5Helper::ReadArray<ET>(layerGroup, "Weights", weights);
						hdf5Helper::ReadArray<ET>(layerGroup, "Biases", biases);
						hdf5Helper::ReadArray<int>(layerGroup, "ConnMap", conn_map);
						switch(tf){
							case eTansig_mod:
                                layers_.push_back(LayerPtr(new CLayer<T,ET,TansigMod<ET> >(inp_width, inp_height, weights, biases, conn_map)));
								break;
							case eTansig:
                                layers_.push_back(LayerPtr(new CLayer<T,ET,Tansig<ET> >(inp_width, inp_height, weights, biases, conn_map)));
								break;
							case ePurelin:
                                layers_.push_back(LayerPtr(new CLayer<T,ET,Purelin<ET> >(inp_width, inp_height, weights, biases, conn_map)));
								break;
							default:
								throw std::runtime_error("Unknown transfer function");
						}
						break;
					}
				case Layer<T, ET>::eFLayer:
					{
						bool is_trainable = hdf5Helper::ReadIntAttribute(layerGroup, "Trainable") == 1 ? true : false;
						eTransfFunc tf = (eTransfFunc)hdf5Helper::ReadIntAttribute(layerGroup, "TransferFunc");
						Tensor<ET> weights, biases;

						hdf5Helper::ReadArray<ET>(layerGroup, "Weights", weights);
						hdf5Helper::ReadArray<ET>(layerGroup, "Biases", biases);
						switch(tf){
							case eTansig_mod:
                                layers_.push_back(LayerPtr(new FLayer<T,ET,TansigMod<ET> >(weights, biases)));
								break;
							case eTansig:
                                layers_.push_back(LayerPtr(new FLayer<T,ET,Tansig<ET> >(weights, biases)));
								break;
							case ePurelin:
                                layers_.push_back(LayerPtr(new FLayer<T,ET,Purelin<ET> >(weights, biases)));
								break;
							default:
								throw std::runtime_error("Unknown transfer function");
						}
						break;
					}
				default:
					throw std::runtime_error("Unknown layer type");
			}
		}
	}
	// catch failure caused by the H5File operations
	catch( FileIException error )
	{
		std::cout<<"Unable to load CNN from file"<<std::endl;
		error.printError();
		return -1;
	}

	// catch failure caused by the DataSet operations
	catch( DataSetIException error )
	{
		std::cout<<"Unable to load CNN from file"<<std::endl;
		error.printError();
		return -1;
	}

	// catch failure caused by the DataSpace operations
	catch( DataSpaceIException error )
	{
		std::cout<<"Unable to load CNN from file"<<std::endl;
		error.printError();
		return -1;
	}

	// catch failure caused by the Attribute operations
	catch( AttributeIException error )
	{
		std::cout<<"Unable to load CNN from file"<<std::endl;
		error.printError();
		return -1;
	}
	return 1;
}

//TODO:This function is not working. Fix it
template< template<class> class T, class ET>
int CNNet<T,ET>::SaveToFile(const std::string& filename)
{
    /*
	try
	{
		Exception::dontPrint();
		H5File file( filename, H5F_ACC_TRUNC );
		Group rootGroup( file.createGroup( ROOT_LAYERS_GROUP_NAME ));

		hdf5Helper::WriteIntAttribute(rootGroup, "nLayers",nlayers_);
		hdf5Helper::WriteIntAttribute(rootGroup, "nInputs",ninputs_);
		hdf5Helper::WriteIntAttribute(rootGroup, "inputWidth",input_width_);
		hdf5Helper::WriteIntAttribute(rootGroup, "inputHeight",input_height_);
		hdf5Helper::WriteIntAttribute(rootGroup, "nOutputs",noutputs_);
		hdf5Helper::WriteIntAttribute(rootGroup, "perfFunc", (int) perf_func_);

		for(int i=0;i<nlayers_;i++) //Skip input and output layers
		{
			Group layerGroup( rootGroup.createGroup(string(ROOT_LAYERS_GROUP_NAME) + string(LAYER_GROUP_NAME) + to_string(i)) ); 
		}

	}
	// catch failure caused by the H5File operations
	catch( FileIException error )
	{
		error.printError();
		return -1;
	}

	// catch failure caused by the DataSet operations
	catch( DataSetIException error )
	{
		error.printError();
		return -1;
	}

	// catch failure caused by the DataSpace operations
	catch( DataSpaceIException error )
	{
		error.printError();
		return -1;
	}

	// catch failure caused by the Attribute operations
	catch( AttributeIException error )
	{
		error.printError();
		return -1;
	}
	return 1;
    */
}

#endif

//=========================== Simulation
//Perform propagation of inputs from the input layer to the output layer
template< template<class> class T, class ET>
void CNNet<T, ET>::Sim(const T<ET>& net_inputs)
{
	typename std::vector<LayerPtr>::iterator it = layers_.begin();
	const T<ET>* layer_input = &net_inputs;
    for ( it = layers_.begin(); it != layers_.end(); ++it)   {
        (*it)->Propagate(*layer_input);
        layer_input = &(*it)->out();
	}
}
//=========================== Initialization
//Perform propagation of inputs from the input layer to the output layer
//Network must be initialized and inputs loaded
template< template<class> class T, class ET>
void CNNet<T, ET>::InitWeights(void(*weight_init_func)(T<ET>&) )
{
    typename std::vector<LayerPtr>::iterator it;
    for(it = layers_.begin(); it != layers_.end(); ++it){
        (*it)->InitWeights(weight_init_func);
    }
}

//=========================== Train methods
template< template<class> class T, class ET>
void CNNet<T,ET>::PrepareForTraining()
{
	typename std::vector<LayerPtr>::iterator it;
    for(it = layers_.begin(); it != layers_.end(); ++it){
        (*it)->PrepareForTraining();
    }
}

//Backpropagate error and update weights
template< template<class> class T, class ET>
void CNNet<T,ET>::BackpropGradients(const T<ET>& net_dedx, const T<ET>& net_input)
{
	assert(net_dedx.num_elements() == this->out().num_elements());	

    layers_.back()->set_de_dx(net_dedx);
    typename std::vector<LayerPtr>::reverse_iterator rit = layers_.rbegin();
    LayerPtr next_layer = *rit;
	++rit;
    for(; rit != layers_.rend(); ++rit){
        next_layer->BackPropagate((*rit)->out(), (*rit)->de_dx());
        next_layer = *rit;
    }
	layers_.front()->ComputeGradient(net_input);
}

template< template<class> class T, class ET>
void CNNet<T,ET>::AccumulateHessian(const T<ET>& net_d2edx2, const T<ET>& net_input)
{
	assert(net_d2edx2.num_elements() == this->out().num_elements());

    layers_.back()->set_d2e_dx2(net_d2edx2);
    typename std::vector<LayerPtr>::reverse_iterator rit = layers_.rbegin();
    LayerPtr next_layer = *rit;
    //Skip last layer
    ++rit;	
    for(; rit != layers_.rend(); ++rit)
	{
		next_layer->BackPropagateHessian((*rit)->out(), (*rit)->d2e_dx2());
        next_layer = *rit;
    }
    layers_.front()->ComputeHessian(net_input);

}

template< template<class> class T, class ET>
void CNNet<T,ET>::ResetHessian()
{
    typename std::vector<LayerPtr>::iterator it;
    for(it = layers_.begin(); it != layers_.end(); ++it){
        (*it)->ResetHessian();
    }
}

template< template<class> class T, class ET>
void CNNet<T,ET>::AverageHessian()
{
	typename std::vector<LayerPtr>::iterator it;
    for(it = layers_.begin(); it != layers_.end(); ++it){
        (*it)->AverageHessian();
    }
}



//Apply calculated gradients to weigts with specified teta
//Network must be initialized and gradients calculated
template< template<class> class T, class ET>
void CNNet<T,ET>::AdaptWeights(ET tau, bool use_hessian, ET mu)
{
	typename std::vector<LayerPtr>::iterator it;
    for(it = layers_.begin(); it != layers_.end(); ++it)
	{
		if ((*it)->is_trainable()) 
		{
			(*it)->AdaptWeights(tau, use_hessian, mu);
		}
    }
}


typedef CNNet< Tensor, float > CNNetF;
typedef CNNet< Tensor, double > CNNetD;
typedef CNNet< TensorGPU, float > CNNetCudaF;
typedef CNNet< TensorGPU, double > CNNetCudaD;

}

#endif