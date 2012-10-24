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

#include "precomp.hpp"

namespace cudacnn
{
#ifdef HAVE_CUDA

    CudaException::CudaException(cudaError err, const char * msg, const char *inpfile, int) :
    m_cuda_err_(cudaSuccess), m_cublas_err_(CUBLAS_STATUS_SUCCESS) 
    {
        m_error_str_.append("Cuda exception. ");
        m_error_str_.append(msg);
        m_error_str_.append(cudaGetErrorString(err));
        m_error_str_.append("File: ");
        m_error_str_.append(inpfile);
        m_error_str_.append("; Line: ");
        //m_error_str_ .append(inpline);
        m_cuda_err_ = err;
    }

    CudaException::CudaException(cublasStatus_t err, const char * msg, const char *inpfile, int) :
    m_cuda_err_(cudaSuccess), m_cublas_err_(CUBLAS_STATUS_SUCCESS) 
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
	return m_error_str_.c_str();
}

#endif
}
