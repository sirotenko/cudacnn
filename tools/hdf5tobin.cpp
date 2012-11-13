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

#include <string>

#include "cudacnn.h"

int HelpPrintout(int exitCode) 
{
    printf(" USAGE : hdf5tobin input output \n");
    printf("\t input .. input convolutional neural network in HDF5 format\n");
    printf("\t output .. output ConvNet in custom binary format\n");
    return exitCode;	
}

inline void CutTrailing(std::string & s) {
	while(s.length() > 0 && (s[s.length()-1] == '"' || s[s.length()-1] == '\xA' || s[s.length()-1] == ' ')) {
		s = s.substr(0,s.length()-1);
	}
}

inline void AddSlash(std::string & path) 
{
	if (path.length() > 0 && path[path.length()-1] != '/') {
		path += '/';
	}
}


int main(int argc, char* argv[]) 
{
    if(argc != 2)
    {
        return HelpPrintout(EXIT_FAILURE);
    }
    int i = 1;
    std::vector<std::string> params;	
    
    while (i < argc)
    {
        std::string name = argv[i++];        

        if (name[0] != '-') 
        {	
            printf("param : %s\n", name.c_str());
            params.push_back(name);
            continue;
        }
    }    
    if (params.size() != 2)
        return HelpPrintout(EXIT_FAILURE);
    CutTrailing(params[1]);
    std::string output_filename = params[1];
    std::string input_filename = params[0];
#ifdef HAVE_CUDA
    cudacnn::CNNetCudaF cnn;
#else
    cudacnn::CNNetF cnn;
#endif
    cnn.LoadFromFile(input_filename);
    cnn.SaveToFileSimple(output_filename);
  
    return EXIT_SUCCESS;
}