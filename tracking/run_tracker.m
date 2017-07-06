function run_tracker(path, dataset, video, visualization, gpus)
% RUN_TRACKER  is the external function of the tracker - does initialization and calls tracker.m
    startup;
    %% Parameters that should have no effect on the result.  
    switch dataset
        case 'OTB'
            RGB_path = '../dataset/OTB/';
            flow_path = '../dataset/OTB_OF/';
            save_path = '../results/OTB/';
            save_name = 'RGB_base';
        case 'VOT'
            RGB_path = '../dataset/VOT/';
            flow_path = '../dataset/VOT_OF/';
            save_path = '../results/VOT/';
            save_name = 'RGB_base';
    end
    
    path = struct();
    path.RGB_path = RGB_path;
    path.flow_path = flow_path;
    path.save_path = save_path;
    path.save_name = save_name;    
    params.path = path;
    params.dataset = dataset;
    params.video = video;
    params.visualization = visualization;
    params.gpus = gpus;

    %% Prameters for debug
    params.bbox_output = true;
    params.video_output = true;
    % params.gt = true
    %% Call the main tracking function
    tracker(params);    
    % two_stream_tracker(params);
end