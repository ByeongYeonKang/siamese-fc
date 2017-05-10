function run_tracker(video, visualization)
% RUN_TRACKER  is the external function of the tracker - does initialization and calls tracker.m
    startup;
    %% Parameters that should have no effect on the result.
    params.video = video;
    params.visualization = visualization;
    params.gpus = 2;
    params.gt = true;
    
    params.t_rgb = true;
    params.t_flow = false;
    params.net_rgb ='2016-08-17_rgb050.net.mat';
    params.net_flow='net-epoch-94.mat';
    params.numScale = 3;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
    %% Parameters that should be recorded.
    % params.foo = 'blah';
    %% Call the main tracking function
    tracker(params);    
 
end