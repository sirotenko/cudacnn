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

#ifndef _CUDACNN_EXCEPTIONS_H_
#define _CUDACNN_EXCEPTIONS_H_
#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <string>

#ifndef NDEBUG
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 200
#define cuassert(condition) \
	if(!(condition)) {printf("CUDA assert failed. File: %s, line: %d \n",__FILE__, __LINE__); }
#else
#define cuassert(condition) \
	if(!(condition)) 
#endif
#else
#define cuassert(condition) \
	if(!(condition)) 
#endif
#else
#define cuassert(condition)
#endif

namespace cudacnn
{

class CudaException 
{
private:
	std::string m_error_str_;
	cudaError m_cuda_err_;
	cublasStatus_t m_cublas_err_;
public:
	CudaException(cudaError err, const char * msg, const char *inpfile, int inpline);
	CudaException(cublasStatus_t err, const char * msg, const char *inpfile, int inpline);
	const char* what() const;
};

inline void cutilSafeCall(cudaError err, const char* msg = "Error. ")
{
	if( cudaSuccess != err) 
	{
		throw CudaException(err, msg, __FILE__, __LINE__);
	}
}

inline void cutilCheckMsg(const char *errorMessage)
{
	cudaError_t err = cudaGetLastError();	
	if( cudaSuccess != err) 
	{
		throw CudaException(err, errorMessage, __FILE__, __LINE__);
	}
}

inline void cublasCheckMsg(cublasStatus_t status, const char* msg)
{
	if(status != CUBLAS_STATUS_SUCCESS){
		throw CudaException(status, msg, __FILE__, __LINE__);
	}
}

}
#endif /* HAVE_CUDA */
#endif /* _CUDACNN_EXCEPTIONS_H_ */
