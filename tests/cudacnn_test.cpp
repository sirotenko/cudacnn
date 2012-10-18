//#define CRTDBG_MAP_ALLOC
#include "precomp.h"

#ifdef HAVE_HDF5
TEST(CNNetTest, ReadFromHDF)
{
	CNNetCudaF cnnet;
	ASSERT_NO_THROW(cnnet.LoadFromFile("test_mnist_net.h5"));
}
#endif //HAVE_HDF5

int main(int argc, char **argv) {
    std::srand(static_cast<unsigned int> (time(0)));

    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    std::cin.get();

    //_CrtDumpMemoryLeaks();
    return ret;

}
