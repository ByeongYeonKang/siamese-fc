function run_tracker(video, visualization, gpus)
% RUN_TRACKER  is the external function of the tracker - does initialization and calls tracker.m
    startup;
    %% Parameters that should have no effect on the result.
    params.video = video;
    params.visualization = visualization;
    params.gpus = gpus;
    
    %% Prameters for debug
    params.bbox_output = true;
    params.video_output = true;
    % params.gt = true
    %% Call the main tracking function
    tracker(params);    
end