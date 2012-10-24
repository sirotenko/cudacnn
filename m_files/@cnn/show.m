function cnnet = show(cnnet)
cnnet = cudacnnMex(cnnet, 'debug_save');
weights_img = cell(length(cnnet.layers));
outs_img = cell(length(cnnet.layers));
for i = 1:length(cnnet.layers)
    if(strcmp(cnnet.layers{i}.LayerType, 'clayer'))
        wsz = size(cnnet.layers{i}.Weights);
        for c = 1:wsz(4)
            for r = 1:wsz(3)
                weights_img{i} = [weights_img{i}; cnnet.layers{i}.Weights(:,:,r,c)];                
            end
            outs_img{i} = [outs_img{i}; cnnet.layers{i}.Output(:,:,c)];
        end
    end
    if(strcmp(cnnet.layers{i}.LayerType, 'pooling'))
        for c = 1:cnnet.layers{i}.NumFMaps
            outs_img{i} = [outs_img{i}; cnnet.layers{i}.Output(:,:,c)];
        end
    end
    
   subplot(1,2*length(cnnet.layers),i*2 - 1) ;
   imshow(weights_img{i},[min(weights_img{i}(:)) max(weights_img{i}(:))]);
   subplot(1,2*length(cnnet.layers),i*2) ;
   imshow(outs_img{i},[min(outs_img{i}(:)) max(outs_img{i}(:))]);   
end
