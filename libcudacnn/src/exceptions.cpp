//#include <iostream>
//#include "../include/common.h"
#include <string>
#include "exceptions.h"

//CudaException::CudaException(const std::string& _Message, cudaError err)
CudaException::CudaException(cudaError err, const char * msg, char *inpfile, int inpline):m_cuda_err_(cudaSuccess),m_cublas_err_(CUBLAS_STATUS_SUCCESS)
{	
	m_error_str_.append("Cuda exception. ");
	m_error_str_.append(msg);
	m_error_str_.append(cudaGetErrorString( err));
	m_error_str_.append("File: ");
	m_error_str_.append(inpfile);
	m_cuda_err_ = err;
}

CudaException::CudaException(cublasStatus_t err, const char * msg, char *inpfile, int inpline):m_cuda_err_(cudaSuccess),m_cublas_err_(CUBLAS_STATUS_SUCCESS)
{	
	m_error_str_.append("CuBLAS exception. ");
	m_error_str_.append(msg);
	//m_error_str_.append(cudaGetErrorString( err));
	m_error_str_.append("File: ");
	m_error_str_.append(inpfile);

	m_cublas_err_ = err;
}

const char* CudaException::what() const
{
	return cudaGetErrorString(m_cuda_err_);
}


