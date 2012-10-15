function out = mse(input)
%MSE compute mean squared error
%
%  Syntax
%  
%    [out] = mse(input)
%    
%  Description
%   Input:
%    input - vector or matrix of elements
%   Output:
%    out - Mean squared error. In case of vector input output is scalar,
%    vector otherwise
%
%(c) Sirotenko Mikhail, 2012
out = sum(input.^2)/length(input);