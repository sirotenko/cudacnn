%Test the performance of CNN on NORB test set

norb_test_reader.num_samples = 50000;
norb_test_reader.current = 1;
norb_test_reader.data_files = {'..\data\NORB\norb-5x01235x9x18x6x2x108x108-testing-01-dat.mat',...
                                 '..\data\NORB\norb-5x01235x9x18x6x2x108x108-testing-02-dat.mat',...
                                 };
norb_test_reader.label_files = {'..\data\NORB\norb-5x01235x9x18x6x2x108x108-testing-01-cat.mat',...
                                 '..\data\NORB\norb-5x01235x9x18x6x2x108x108-testing-02-cat.mat',...
                                };
norb_test_reader.buffer_size = 1000;
norb_test_reader.stereo = false;
norb_test_reader.read = @norb_datareader;

correct=0;
for i = 1:norb_test_reader.num_samples
    [inp, targ, norb_test_reader] = norb_test_reader.read(norb_test_reader);
    out = cudacnnMex(cnnet,'sim',single(inp)); 
    if(find(out == max(out))==(find(targ == max(targ))))
        correct=correct+1;
    end    
end
mcr = 1-correct/norb_test_reader.num_samples;