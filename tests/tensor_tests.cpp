#include "tests_common.h"


TEST(TensorTest, TestShallowCreate)
{
	const int ndims = 3;
	std::vector<UINT> dims(ndims);
	for(int i = 0; i < ndims; ++i) dims[i] = i;
	Tensor<float> out_tens;
	float* pdata = new float[6];
	out_tens = Tensor<float>(dims, pdata);
}