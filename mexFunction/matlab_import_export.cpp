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

eTransfFunc GetTransferFunc(const mxArray* pCell)
{
	eTransfFunc tf = eTransferUnknown;
	if(strcmp(GetSVal(pCell,"TransferFunc"),"tansig")==0)	tf = eTansig;
	if(strcmp(GetSVal(pCell,"TransferFunc"),"tansig_mod")==0)	tf = eTansig_mod;
	if(strcmp(GetSVal(pCell,"TransferFunc"),"purelin")==0)	tf = ePurelin;
	return tf;
}


/*!
 * This function pased to matlab to allow automatic freeing of memory in case of error or matlab exit
 */