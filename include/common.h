#pragma once

//TODO: make this dependent from compute capability
#define MAX_THREADS 512
//Optimal number of threads for the best occupancy.
//This is kind of heuristic. Good setting for many modern nVidia GeForce cards
//and for not too sophisticated kernels.
#define MAX_OCCUP_THREADS 192

#define NULL    0

typedef unsigned char BYTE;
typedef unsigned int UINT;

#define ROOT_LAYERS_GROUP_NAME "/Layers"
#define LAYER_GROUP_NAME "/Layer"

#define Sqr(a) (a)*(a)

enum ePerfType
{
    eMSE,
    eSSE
};


enum eTransfFunc
{
	eTransferUnknown,
	ePurelin,
	eTansig_mod,
	eTansig,
	eSquare
};

enum ePoolingType
{
	eSubsampling,
	eMaxPooling
};

struct sSRate
{
	BYTE x; 
	BYTE y;
};

inline int iRoundUpPow2(int v){
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v++;
	return v;
}

//template<class T, class B> struct Derived_from {
//	static void constraints(T* p) { B* pb = p; }
//	Derived_from() { void(*p)(T*) = constraints; }
//};
