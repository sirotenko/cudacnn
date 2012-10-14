//#define CRTDBG_MAP_ALLOC
#include "precomp.h"

#ifdef HAVE_HDF5
//FIXME test_mnist_net.h5 does not support tf_params
TEST(CNNetTest, ReadFromHDF)
{
	CNNetCudaF cnnet;
	cnnet.LoadFromFile("H5_TEST_H5_v2.h5");
}
#endif //HAVE_HDF5

int main(int argc, char **argv) 
{
	std::srand(static_cast<unsigned int>(time(0)));

  	::testing::InitGoogleTest(&argc, argv);
	RUN_ALL_TESTS();
    std::cin.get();
	
    //_CrtDumpMemoryLeaks();
	return 0;
	
}
