#include "precomp.h"

class TensorTest : public ::testing::Test {
protected:
	//TensorFloat tensor;
	virtual void SetUp()
	{
	}

	virtual void TearDown()
	{
	}
};

//Only work in debug mode and in VS
#ifdef _MSC_VER
#ifdef _DEBUG
TEST_F(TensorTest, TestMemoryLeak)
{
    _CrtMemState s1, s2, s3;
    _CrtMemCheckpoint( &s1 );         
    for (int i = 0; i < 10; ++i)  {
        TensorFloat tens(100,100,3);
    }
    _CrtMemCheckpoint( &s2 );
    _CrtMemDifference( &s3, &s1, &s2);
    ASSERT_EQ(s3.lCounts[0] | s3.lCounts[1] | s3.lCounts[2] | s3.lCounts[3] |
              s3.lSizes[0]  | s3.lSizes[1]  | s3.lSizes[2]  | s3.lSizes[3] , 0)<<"Memory leaks in Tensor"<<std::endl;
}
#endif
#endif

