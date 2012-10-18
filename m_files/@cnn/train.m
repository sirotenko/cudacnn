function [cnet] = train(cnet, trainer, train_datareader, test_datareader)

%TRAIN train convolutional neural network
%
%  Syntax
%  
%    [cnet] = train(cnet, trainer_struct, inp_train, targ_train, inp_test, targ_test)
%    
%  Description
%   Input:
%    cnet - Convolutional neural network class object
%    trainer_struct - Structure containing training settings 
%    inp_train - cell array, containing preprocessed inputs for training
%    targ_train - cell array of targets, corresponding to train images
%    inp_test - cell array, containing preprocessed inputs for testing
%    targ_test - cell array of targets, corresponding to test inputs
%   Output:
%    cnet - trained convolutional neural network
%
%   Hessian running estimate not supported yet
%(c) Sirotenko Mikhail, 2009

%Initialize GUI
h_gui = cnn_gui();
%Progress bars
h_HessPatch = findobj(h_gui,'Tag','HessPatch');
h_HessEdit = findobj(h_gui,'Tag','HessPrEdit');
h_TrainPatch = findobj(h_gui,'Tag','TrainPatch');
h_TrainEdit = findobj(h_gui,'Tag','TrainPrEdit');
%Axes
h_MCRaxes = findobj(h_gui,'Tag','MCRaxes');
h_RMSEaxes = findobj(h_gui,'Tag','RMSEaxes');
%Info textboxes
h_EpEdit = findobj(h_gui,'Tag','EpEdit');
h_ItEdit = findobj(h_gui,'Tag','ItEdit');
h_RMSEedit = findobj(h_gui,'Tag','RMSEedit');
h_MCRedit = findobj(h_gui,'Tag','MCRedit');
h_ThetaEdit = findobj(h_gui,'Tag','TetaEdit');
%Buttons
h_AbortButton = findobj(h_gui,'Tag','AbortButton');

tic;    %Fix the start time
perf_plot = []; %Array for storing performance data
%Coefficient, determining the running estimation of diagonal 


trainer.MCRSubsetSize = min(trainer.MCRSubsetSize, test_datareader.num_samples);

num_samples = train_datareader.num_samples;
mcr = [];
%Initial MCR calculation
[mcr(1) test_datareader]=calcMCR(cnet,test_datareader, 1:trainer.MCRSubsetSize);
plot(h_MCRaxes,mcr);
SetText(h_MCRedit,mcr(end));
%Initialize trainer
%trainer = validate_trainer(trainer,true);
%cudacnnMex(cnet,'init_trainer',trainer);

if(trainer.HcalcMode == 1)
    cudacnnMex(cnet,'reset_hessian');
    for i=1:trainer.HrecalcSamplesNum
        %Simulating
        [inp, targ, train_datareader] = train_datareader.read(train_datareader, i);
        [out, cnet] = sim(cnet,single(inp));    
        %Calculate the error
        e_derriv = 2*single(ones(size(out)));
        cudacnnMex(cnet,'accum_hessian',e_derriv,single(inp));
        SetHessianProgress(h_HessPatch,h_HessEdit,i/trainer.HrecalcSamplesNum);
    end
    %Averaging
    cudacnnMex(cnet,'average_hessian');
end
%For all epochs
for t=1:trainer.epochs
    SetText(h_EpEdit,t);
    SetTextHP(h_ThetaEdit,trainer.theta(t));
    %For all patterns
    for n=1:num_samples
        
        iteration = (t-1)*num_samples + n;
        
         if(trainer.HcalcMode == 1)            
            if(mod(t * num_samples + n,trainer.Hrecalc) == 0) %If it is time to recalculate Hessian
				cudacnnMex(cnet,'reset_hessian');
                if(n+trainer.HrecalcSamplesNum > num_samples)
                    stInd = num_samples - trainer.HrecalcSamplesNum;
                else
                    stInd = n;
                end
                for i=stInd:stInd+trainer.HrecalcSamplesNum
                    %Simulating
                    [inp, targ, train_datareader] = train_datareader.read(train_datareader, i);
                    [out, cnet] = sim(cnet,single(inp));    
                    e_derriv2 = 2*single(ones(size(out)));
                    cudacnnMex(cnet,'accum_hessian',e_derriv2,single(inp));                
                    SetHessianProgress(h_HessPatch,h_HessEdit,(i-stInd)/trainer.HrecalcSamplesNum);
                end
                %Averaging
                cudacnnMex(cnet,'average_hessian');
            end
        end
        %Simulating
        [inp, targ, train_datareader] = train_datareader.read(train_datareader,n);
                
        [out, cnet] = sim(cnet,single(inp));         
      
        %Calculate the error
        e = single(out-targ');
        %TODO: this is only true for MSE. Fix it
        e_derriv = 2*e/length(e);        
        cudacnnMex(cnet, 'compute_gradient', e_derriv, single(inp));


        perf(n) = mse(e); %Store the error
        
        %Adapt
        cudacnnMex(cnet, 'adapt', trainer.theta(t), uint32(trainer.HcalcMode>1), trainer.mu);
        
        %Plot mean of performance for every N patterns
        if(n>1)
            if(~mod(n-1,trainer.RMSEUpdatePeriod))
                perf_plot = [perf_plot,mean(sqrt(perf(n-10:n)))];
                plot(h_RMSEaxes,perf_plot);
                SetText(h_RMSEedit,perf_plot(end));
            end
            if(~mod(n-1,trainer.MCRUpdatePeriod))
                [new_mcr test_datareader] = calcMCR(cnet,test_datareader, 1:trainer.MCRSubsetSize);
                mcr = [mcr new_mcr];
                plot(h_MCRaxes,mcr);
                SetText(h_MCRedit,mcr(end));
            end
        end
        
        SetTrainingProgress(h_TrainPatch,h_TrainEdit,(n+(t-1)*num_samples)/(num_samples*trainer.epochs));
        SetText(h_ItEdit,n);
        drawnow;
        if(~isempty(get(h_AbortButton,'UserData')))
            fprintf('Training aborted \n');
            cnet = cudacnnMex(cnet, 'debug_save');
            evalc('cnet = cnn(cnet,false);'); %Suppress "CNN successfully initialized." message
            return;
        end
    end

end
cnet = cudacnnMex(cnet, 'save');
toc

%calcMCR Calculate missclassification rate
function [mcr dreader] = calcMCR(cnet,dreader, idxs)
correct=0;
for i=idxs
    [inp, targ, dreader] = dreader.read(dreader, i);
    out = cudacnnMex(cnet,'sim',single(inp));   
    if(find(out == max(out))==(find(targ == max(targ))))
        correct=correct+1;
    end
end
mcr = 1-correct/length(idxs);

%Sets Hessian progress
%hp - handle of patch
%hs - handle of editbox
%pr - value from 0 to 1
function SetHessianProgress(hp,hs,pr)
xpatch = [0 pr*100 pr*100 0];
set(hp,'XData',xpatch);
set(hs,'String',[num2str(pr*100,'%5.2f'),'%']);
drawnow;


%Sets Training progress
%hp - handle of patch
%hs - handle of editbox
%pr - value from 0 to 1
function SetTrainingProgress(hp,hs,pr)
xpatch = [0 pr*100 pr*100 0];
set(hp,'XData',xpatch);
set(hs,'String',[num2str(pr*100,'%5.2f'),'%']);

%Set numeric text in the specified edit box
%hs - handle of textbox
%num - number to convert and set
function SetText(hs,num)
set(hs,'String',num2str(num,'%5.2f'));

%Set numeric text in the specified edit box with high preceition
%hs - handle of textbox
%num - number to convert and set
function SetTextHP(hs,num)
set(hs,'String',num2str(num,'%5.3e'));