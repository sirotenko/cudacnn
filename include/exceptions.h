/*
 * exceptions.h
 *
 *  Created on: Apr 13, 2010
 *      Author: sirotenko
 */
#pragma once
#ifndef EXCEPTIONS_H_
#define EXCEPTIONS_H_
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

class CudaException //: public std::exception
{
private:
	std::string m_error_str_;
	cudaError m_cuda_err_;
	cublasStatus_t m_cublas_err_;
public:
	CudaException(cudaError err, const char * msg, char *inpfile, int inpline);
	CudaException(cublasStatus_t err, const char * msg, char *inpfile, int inpline);
	const char* CudaException::what() const;
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

#endif /* EXCEPTIONS_H_ */
