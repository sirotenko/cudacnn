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

#ifndef _MATLAB_IMPORTER_H
#define _MATLAB_IMPORTER_H


template<template<class> class TT, class T>
class MatlabSaveLoad : public cudacnn::Layer<TT,T>::ILoadSaveObject
{
public:
	MatlabSaveLoad(mxArray* struct_arr): struct_arr_(struct_arr){};
	virtual void AddScalar(T scal, const std::string name); 
	virtual void AddScalar(int scal, const std::string name); 
	virtual void AddScalar(UINT scal, const std::string name); 
	virtual void AddArray(const TT<T>& arr, const std::string name); 
	virtual void AddArray(const TT<int>& arr, const std::string name); 
	virtual void AddString(const std::string str, const std::string name); 

	virtual void GetScalar(T& scal, const std::string name); 
	virtual void GetScalar(int& scal, const std::string name); 
	virtual void GetScalar(UINT& scal, const std::string name); 
	virtual void GetArray(TT<T>& arr, const std::string name); 
	virtual void GetArray(TT<int>& arr, const std::string name); 
	virtual void GetString(std::string& str, const std::string name); 
	virtual ~MatlabSaveLoad(){};
private:
	mxArray* struct_arr_;
};

template<template<class> class TT, class T>
void MatlabSaveLoad<TT,T>::AddScalar(T scal, const std::string name)
{
	MatlabTools::AddScalar<T>(scal,name.c_str(),struct_arr_);
}
template<template<class> class TT, class T>
void MatlabSaveLoad<TT,T>::AddScalar(UINT scal, const std::string name)
{
	MatlabTools::AddScalar<UINT>(scal,name.c_str(),struct_arr_);
}
template<template<class> class TT, class T>
void MatlabSaveLoad<TT,T>::AddScalar(int scal, const std::string name)
{
	MatlabTools::AddScalar<int>(scal,name.c_str(),struct_arr_);
}
template<template<class> class TT, class T>
void MatlabSaveLoad<TT,T>::AddArray(const TT<T>& arr, const std::string name)
{
	MatlabTools::AddArray<T>(arr,name.c_str(),struct_arr_);
}
template<template<class> class TT, class T>
void MatlabSaveLoad<TT,T>::AddArray(const TT<int>& arr, const std::string name)
{
	MatlabTools::AddArray<int>(arr,name.c_str(),struct_arr_);
}
template<template<class> class TT, class T>
void MatlabSaveLoad<TT,T>::AddString(const std::string str, const std::string name)
{
	mxArray* str_arr = mxCreateString(str.c_str());
	mxAddField(struct_arr_, name.c_str());
	mxSetField(struct_arr_, 0, name.c_str(), str_arr);
}
//================ Getters ============================
template<template<class> class TT, class T>
void MatlabSaveLoad<TT,T>::GetScalar(T& scal, const std::string name)
{
	scal = MatlabTools::GetScalar<T>(struct_arr_, name.c_str());
}
template<template<class> class TT, class T>
void MatlabSaveLoad<TT,T>::GetScalar(UINT& scal, const std::string name)
{
	scal = MatlabTools::GetScalar<UINT>(struct_arr_, name.c_str());
}
template<template<class> class TT, class T>
void MatlabSaveLoad<TT,T>::GetScalar(int& scal, const std::string name)
{
	scal = MatlabTools::GetScalar<int>(struct_arr_, name.c_str());
}
template<template<class> class TT, class T>
void MatlabSaveLoad<TT,T>::GetArray(TT<T>& arr, const std::string name)
{
	MatlabTools::GetArray<T>(struct_arr_, name.c_str(),arr);
}
template<template<class> class TT, class T>
void MatlabSaveLoad<TT,T>::GetArray(TT<int>& arr, const std::string name)
{
	MatlabTools::GetArray<int>(struct_arr_, name.c_str(),arr);
}
template<template<class> class TT, class T>
void MatlabSaveLoad<TT,T>::GetString(std::string& str, const std::string name)
{
	str = MatlabTools::GetSVal(struct_arr_,name.c_str());
}



using namespace MatlabTools;
cudacnn::eTransfFunc GetTransferFunc(const mxArray* pCell);

//TODO: create some universal function for CNNet import and export either to/from matlab or hdf5
template<template<class> class TT, class T>
int CNNetLoadFromMatlab(const mxArray* cnnAr, cudacnn::CNNet<TT,T>& cnet)
{

	try{
		unsigned char nlayers = GetScalar<unsigned char>(cnnAr,"nlayers");
		unsigned char ninputs = GetScalar<unsigned char>(cnnAr,"nInputs");
		UINT input_width = GetScalar<UINT>(cnnAr,"inputWidth");
		UINT input_height = GetScalar<UINT>(cnnAr,"inputHeight");

        std::vector<CNNet<TT,T>::LayerPtr> layers;

		//Load all layers
		mxArray* players_ar = mxGetField(cnnAr,0,"layers");
		if(players_ar == NULL) {
			std::stringstream ss;
			ss<<"layers field not found in cnnet object.";
			throw std::runtime_error(ss.str());
		}

		for(int i = 0; i < nlayers; i++) 
		{
			mxArray* pCell = mxGetCell(players_ar,i);
			Layer<TT, T>::eLayerType lt = Layer<TT, T>::eLayerUnknown;			
			if(GetSVal(pCell,"LayerType").compare("clayer")==0)	lt = Layer<TT, T>::eCLayer;
			if(GetSVal(pCell,"LayerType").compare("flayer")==0)	lt = Layer<TT, T>::eFLayer;
            if(GetSVal(pCell,"LayerType").compare("pooling")==0)	lt = Layer<TT, T>::ePLayer;
			switch(lt){
		case Layer<TT, T>::ePLayer:
			{
				PoolingLayerT<TT,T>::Params params;
				params.ninputs = GetScalar<UINT>(pCell,"NumFMaps");
				params.inp_width = GetScalar<UINT>(pCell,"InpWidth");
				params.inp_height = GetScalar<UINT>(pCell,"InpHeight");
				params.sx = GetScalar<UINT>(pCell,"SXRate");
				params.sy = GetScalar<UINT>(pCell,"SYRate");
                params.pooling_type = PoolingLayerT<TT, T>::eUnknown;
                if(GetSVal(pCell,"PoolingType").compare("average")==0)	
                    params.pooling_type = PoolingLayerT<TT, T>::eAverage;
                if(GetSVal(pCell,"PoolingType").compare("max")==0)	
                    params.pooling_type = PoolingLayerT<TT, T>::eMax;

                layers.push_back(CNNet<TT,T>::LayerPtr(new PoolingLayer<TT,T>(params)));
				break;
			}

		case Layer<TT, T>::eCLayer:
			{
				bool is_trainable = GetScalar<UINT>(pCell,"Trainable") == 1 ? true : false;
				UINT inp_width = GetScalar<UINT>(pCell,"InpWidth");
				UINT inp_height = GetScalar<UINT>(pCell,"InpHeight");
				Tensor<T> weights, biases;
				TensorInt conn_map;
				GetArray<T>(pCell,"Weights", weights);
				GetArray<T>(pCell,"Biases", biases);
				GetArray<int>(pCell,"conn_map", conn_map);
				eTransfFunc tf = GetTransferFunc(pCell);
				switch(tf)	{
                    case eTansig_mod:
                        {
                            layers.push_back(CNNet<TT,T>::LayerPtr(new CLayer<TT,T,TansigMod<T> >(inp_width, inp_height, is_trainable, weights, biases, conn_map)));
                        }
                        break;
                    case eTansig:
                        {
                            layers.push_back(CNNet<TT,T>::LayerPtr(new CLayer<TT,T,Tansig<T> >(inp_width, inp_height, is_trainable, weights, biases, conn_map)));
                        }
                        break;
                    case ePurelin:
                        {
                            layers.push_back(CNNet<TT,T>::LayerPtr(new CLayer<TT,T,Purelin<T> >(inp_width, inp_height, is_trainable, weights, biases, conn_map)));
                        }
                        break;
					default:
						throw std::runtime_error("Unknown transfer function");

				}
				break;
			}
			case Layer<TT, T>::eFLayer:
				{
					bool is_trainable = GetScalar<UINT>(pCell,"Trainable") == 1 ? true : false;
					eTransfFunc tf = GetTransferFunc(pCell);
					Tensor<T> weights, biases;
					GetArray<T>(pCell,"Weights", weights);
					GetArray<T>(pCell,"Biases", biases);
			
					switch(tf){
                        case eTansig_mod:
                            {
                                layers.push_back(CNNet<TT,T>::LayerPtr(new FLayer<TT,T,TansigMod<T> >(weights, biases, is_trainable)));
                            }
                            break;
                        case eTansig:
                            {
                                layers.push_back(CNNet<TT,T>::LayerPtr(new FLayer<TT,T,Tansig<T> >(weights, biases, is_trainable)));
                            }
                            break;
                        case ePurelin:
                            {
                                layers.push_back(CNNet<TT,T>::LayerPtr(new FLayer<TT,T,Purelin<T> >(weights, biases, is_trainable)));
                            }
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
		cnet.Init(layers, ninputs, input_width, input_height);
	}
	catch(std::runtime_error& ex){
		std::stringstream ss;
		ss<<"NNet loading failed "<<ex.what()<<std::endl;
		mexErrMsgTxt(ss.str().c_str());
	}

	return 0;
}



template<template<class> class TT, class T>
mxArray* CNNetSaveToMatlab(cudacnn::CNNet<TT,T>& cnet, bool debug_save = false)
{
	try{
		mxArray* struct_arr = mxCreateStructMatrix(1,1,0,NULL);
		
		AddScalar(cnet.nlayers(),"nlayers", struct_arr);
		AddScalar(cnet.ninputs(),"nInputs", struct_arr);
		AddScalar(cnet.input_width(), "inputWidth", struct_arr);
		AddScalar(cnet.input_height(), "inputHeight", struct_arr);	

		mxArray* layers_cell_arr = mxCreateCellMatrix(cnet.nlayers(),1);
		for(int i = 0; i < cnet.nlayers(); ++i){
			mxArray* layer_arr = mxCreateStructMatrix(1,1,0,NULL);
			MatlabSaveLoad<TT,T> layer_save_obj(layer_arr);
			cnet[i]->Save(layer_save_obj, debug_save);
			mxSetCell(layers_cell_arr,i,layer_arr);
		}
		mxAddField(struct_arr,"layers");
		mxSetField(struct_arr,0,"layers",layers_cell_arr);

		return struct_arr;
	}
	//TODO: free all allocated memory
	catch(std::runtime_error& ex){
		mexErrMsgTxt(ex.what());
	}
	return NULL;
}


#endif

