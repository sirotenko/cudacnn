function [out, cnet_out] = sim(cnet,inp)
%SIM simulate convolutional neural network 
%
%  Syntax
%  
%    [out, sinet] = sim(cnet,inp)
%    
%  Description
%   Input:
%    cnet - Convolutional neural network struct
%    inp - input image matrix or 3D array of images
%   Output:
%    cnet_out - Convolutional neural network with unchanged weignts and biases
%    but with saved layers outputs 
%    out - simulated neural network output
%
%(c) Sirotenko Mikhail, 2011

out = cudacnnMex(cnet,'sim',single(inp));
%cnet_out = cudacnnMex(cnet, 'save');
cnet_out = cnet;

