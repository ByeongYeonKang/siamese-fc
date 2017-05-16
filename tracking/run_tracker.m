function result = run_tracker(video, visualization, gpus)
% RUN_TRACKER  is the external function of the tracker - does initialization and calls tracker.m
    startup;
    %% Parameters that should have no effect on the result.
    params.video = video;
    params.visualization = visualization;
    params.gpus = gpus;
    
    %% Prameters for debug
    params.saveVideos = true;
    params.getRect = false;
    % params.bbox_output = true;
    % params.gt = true
    %% Call the main tracking function
    result = tracker(params);    
end