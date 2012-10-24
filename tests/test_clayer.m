function out = test_clayer(inp, weights, biases, con_map)
numKernels = length(weights);
YC = num2cell(zeros(numKernels,1));
for l=1:numKernels %For all convolutional kernels
    for m=find(con_map(l,:)) %For all feature maps of previous layer and according to connection map
        %Convolute and accumulate
        YC{l} = YC{l}+fastFilter2(weights{l},inp{m},'valid')+biases{l};
        %YC{l} = YC{l}+conv2(inp{m},weights{l},'valid')+biases{l};
    end
end
out = YC;