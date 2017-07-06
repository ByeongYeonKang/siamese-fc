%% COMPILE_MATCONVNET
%
% Compile MatConvNet
%
% ByeongYeon Kang, 2017 
%

run matconvnet-1.0-beta20/matlab/vl_setupnn ;
cd matconvnet-1.0-beta20;

vl_compilenn('enableGpu', true, ...
               'cudaRoot', '/usr/local/cuda-8.0', ...
               'cudaMethod', 'nvcc', ...
               'enableCudnn', true, ...
                'cudnnRoot', '/usr/local/cuda-8.0', ...
                'verbose', 1);

cd ..;
