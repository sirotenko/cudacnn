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

using namespace MatlabTools;


/*!
 * Function loads neural network structure and parameters from matlab mat-file
 * @param filename is a name of mat-file
 * @param varname is a name of variable, containing CNNetF structure
 * @return 1 is successfull, 0 otherwise
 */
//int MatlabImportExport::CNNetLoadFromMatlab(const char* filename, const char* varname, CNNetF& cnet)
//{
//	MATFile *pmat;
//
//	pmat  = matOpen(filename,"r");
//	if(pmat==NULL)
//		return -1;
//	mxArray *cnnMat = matGetVariable(pmat, varname);
//	if(cnnMat==NULL)
//		return -1;
//	//Parse and load CNNetF
//	CNNetLoadFromMatlab(cnnMat, cnet);
//	return 1;
//}

cudacnn::eTransfFunc GetTransferFunc(const mxArray* pCell)
{
	cudacnn::eTransfFunc tf = cudacnn::eTransferUnknown;
	if(GetSVal(pCell,"TransferFunc").compare("tansig")==0)	tf = cudacnn::eTansig;
	if(GetSVal(pCell,"TransferFunc").compare("tansig_mod")==0)	tf = cudacnn::eTansig_mod;
	if(GetSVal(pCell,"TransferFunc").compare("purelin")==0)	tf = cudacnn::ePurelin;

	return tf;
}


/*!
 * This function pased to matlab to allow automatic freeing of memory in case of error or matlab exit
 */