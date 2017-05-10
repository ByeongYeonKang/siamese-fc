% -------------------------------------------------------------------------------------------------
function bboxes = tracker(varargin)
%TRACKER
%   is the main function that performs the tracking loop
%   Default parameters are overwritten by VARARGIN
%
%   Luca Bertinetto, Jack Valmadre, Joao F. Henriques, 2016
% -------------------------------------------------------------------------------------------------
    
    % These are the default hyper-params for SiamFC-3S
    % The ones for SiamFC (5 scales) are in params-5s.txt
    
    %% execution, visualization, benchmark
    p.video = ' ';
    p.visualization = false;
    p.gpus = 2;
    p.bbox_output = false;
    p.fout = -1;
    %% Params from the network architecture, have to be consistent with the training
    p.exemplarSize = 127;  % input z size
    p.instanceSize = 255;  % input x size (search region)
    p.scoreSize = 17;
    p.totalStride = 8;
    p.contextAmount = 0.5; % context amount for the exemplar
    p.subMean = false;
    %% SiamFC prefix and ids
    p.prefix_z = 'a_'; % used to identify the layers of the exemplar
    p.prefix_x = 'b_'; % used to identify the layers of the instance
    p.prefix_join = 'xcorr';
    p.prefix_adj = 'adjust';
    p.id_feat_z = 'a_feat';
    p.id_score = 'score';
    p.gt = true;
    % Overwrite default parameters with varargin
    p = vl_argparse(p, varargin);
% -------------------------------------------------------------------------------------------------

    % Get environment-specific default paths.
    p = env_paths_tracking(p);
    p = params_3s(p);
    
    % Load ImageNet Video statistics
    if exist(p.stats_path,'file')
        stats = load(p.stats_path);
    else
        warning('No stats found at %s', p.stats_path);
        stats = [];
    end
    % Load two copies of the pre-trained network
%     net_z_rgb = load_pretrained([p.net_base_path p.net_rgb], p.gpus);
    net_z_rgb = load_pretrained([p.net_base_path p.net_rgb], []);
    net_x_rgb = load_pretrained([p.net_base_path p.net_rgb], []);
    
%     net_z_flow = load_pretrained([p.net_base_path p.net_flow], p.gpus);
    net_z_flow = load_pretrained([p.net_base_path p.net_flow], []);
    net_x_flow = load_pretrained([p.net_base_path p.net_flow], []);
    
    % [imgFiles, targetPosition, targetSize] = load_video_info(p.seq_base_path, p.video);
    % flow model
    % [flowFiles, imgFiles, targetPosition, targetSize, video_path] = load_video_info2(p.seq_base_path, p.gt);
    % rgb model
    [flowFiles, imgFiles, targetPosition, targetSize, video_path] = load_video_info2(p.seq_base_path, p.gt);
    [upperPath, deepestFolder] = fileparts(video_path);
    [upperPath, deepestFolder] = fileparts(upperPath);
    p.video = deepestFolder;
    nImgs = numel(flowFiles);
    startFrame = 1;
    % Divide the net in 2
    % exemplar branch (used only once per video) computes features for the target
    remove_layers_from_prefix(net_z_rgb, p.prefix_x);
    remove_layers_from_prefix(net_z_rgb, p.prefix_join);
    remove_layers_from_prefix(net_z_rgb, p.prefix_adj);
    % instance branch computes features for search region x and cross-correlates with z features
    remove_layers_from_prefix(net_x_rgb, p.prefix_z);
    zFeatId = net_z_rgb.getVarIndex(p.id_feat_z);
    scoreId = net_x_rgb.getVarIndex(p.id_score);
    
    % Divide the net in 2
    % exemplar branch (used only once per video) computes features for the target
    remove_layers_from_prefix(net_z_flow, p.prefix_x);
    remove_layers_from_prefix(net_z_flow, p.prefix_join);
    remove_layers_from_prefix(net_z_flow, p.prefix_adj);
    % instance branch computes features for search region x and cross-correlates with z features
    remove_layers_from_prefix(net_x_flow, p.prefix_z);
    
    % get the first frame of the video
    flow = gpuArray(single(flowFiles{startFrame}));
    im = gpuArray(single(imgFiles{startFrame}));
    
    % if grayscale repeat one channel to match filters size
	if(size(im, 3)==1)
        im = repmat(im, [1 1 3]);
    end
    % Init visualization
    videoPlayer = [];
    if p.visualization && isToolboxAvailable('Computer Vision System Toolbox')
        videoPlayer = vision.VideoPlayer('Position', [100 100 [size(im,2), size(im,1)]+30]);
    end
    % get avg for padding
    avgChans = gather([mean(mean(im(:,:,1))) mean(mean(im(:,:,2))) mean(mean(im(:,:,3)))]);

    if ~isdir([p.save_path 'two-stream/' p.video])
        mkdir([p.save_path 'two-stream/' p.video])
%         mkdir([p.save_path p.net '/' p.video '/track'])
%         mkdir([p.save_path p.net '/' p.video '/score'])
    end
    
    v_img = VideoWriter([p.save_path 'two-stream/' p.video '/rgb']);
    open(v_img);
    v_flow = VideoWriter([p.save_path 'two-stream/' p.video '/flow']);
    open(v_flow);
%     v_score = VideoWriter([p.save_path 'two-stream/' p.video '/full_score']);
%     open(v_score);
    v_x_score = VideoWriter([p.save_path 'two-stream/' p.video '/crop_score']);
    open(v_x_score);
    v_x_crop = VideoWriter([p.save_path 'two-stream/' p.video '/crop']);
    open(v_x_crop);

    wc_z = targetSize(2) + p.contextAmount*sum(targetSize);
    hc_z = targetSize(1) + p.contextAmount*sum(targetSize);
    s_z = sqrt(wc_z*hc_z);
    scale_z = p.exemplarSize / s_z;
    % initialize the exemplar
    [z_crop, ~] = get_subwindow_tracking(im, targetPosition, [p.exemplarSize p.exemplarSize], [round(s_z) round(s_z)], avgChans);
    if p.subMean
        z_crop = bsxfun(@minus, z_crop, reshape(stats.z.rgbMean, [1 1 3]));
    end
    d_search = (p.instanceSize - p.exemplarSize)/2;
    pad = d_search/scale_z;
    s_x = s_z + 2*pad;
    % arbitrary scale saturation
    min_s_x = 0.2*s_x;
    max_s_x = 5*s_x;

    switch p.windowing
        case 'cosine'
            window = single(hann(p.scoreSize*p.responseUp) * hann(p.scoreSize*p.responseUp)');
        case 'uniform'
            window = single(ones(p.scoreSize*p.responseUp, p.scoreSize*p.responseUp));
    end
    % make the window sum 1
    window = window / sum(window(:));
    scales = (p.scaleStep .^ ((ceil(p.numScale/2)-p.numScale) : floor(p.numScale/2)));
    % evaluate the offline-trained network for exemplar z features
    net_z_rgb.eval({'exemplar', z_crop});
    z_features_rgb = net_z_rgb.vars(zFeatId).value;
    z_features_rgb = repmat(z_features_rgb, [1 1 1 p.numScale]);

    bboxes = zeros(nImgs, 4);
   
    % start tracking
    tic;
    for i = startFrame:nImgs
        if i>startFrame
            % load new frame on GPU
            flow = gpuArray(single(flowFiles{i}));   
            % load new frame on GPU
            im = gpuArray(single(imgFiles{i}));
            
            if(startFrame==1)
                net_z_flow.eval({'exemplar', z_crop});
                z_features_flow = net_z_flow.vars(zFeatId).value;
                z_features_flow = repmat(z_features_flow, [1 1 1 p.numScale]);
            end
            
   			% if grayscale repeat one channel to match filters size
    		if(size(flow, 3)==1)
        		flow = repmat(flow, [1 1 3]);
    		end
            scaledInstance = s_x .* scales;
            scaledTarget = [targetSize(1) .* scales; targetSize(2) .* scales];
            
            % extract scaled crops for search region x at previous target position
            x_crops_rgb = make_scale_pyramid(im, targetPosition, scaledInstance, p.instanceSize, avgChans, stats, p);
            x_crops_flow = make_scale_pyramid(flow, targetPosition, scaledInstance, p.instanceSize, avgChans, stats, p);
            
            % imwrite(gather(x_crops(:,:,:,1))/255,[p.save_path p.net '/' p.video '/track/' num2str(i) '.jpg']);
            % evaluate the offline-trained network for exemplar x features
            %[newTargetPosition, newScale, scoreMap] = tracker_eval(net_x_rgb, round(s_x), scoreId, z_features_rgb, x_crops_rgb, targetPosition, window, p);
            [newTargetPosition, newScale, scoreMap] = two_tracker_eval(net_x_rgb, net_x_flow, round(s_x), scoreId, z_features_rgb, z_features_flow, x_crops_rgb, x_crops_flow, targetPosition, window, p);
            targetPosition = gather(newTargetPosition);
            % scale damping and saturation
            s_x = max(min_s_x, min(max_s_x, (1-p.scaleLR)*s_x + p.scaleLR*scaledInstance(newScale)));
            targetSize = (1-p.scaleLR)*targetSize + p.scaleLR*[scaledTarget(1,newScale) scaledTarget(2,newScale)];
            vis_crop = gather(x_crops_rgb(:,:,:,newScale));
            writeVideo(v_x_crop, mat2gray(vis_crop));
            vis_score_map(scoreMap, targetPosition, targetSize, v_x_score);
        else
            % at the first frame output position and size passed as input (ground truth)
        end

        rectPosition = [targetPosition([2,1]) - targetSize([2,1])/2, targetSize([2,1])];
        % output bbox in the original frame coordinates
        oTargetPosition = targetPosition; % .* frameSize ./ newFrameSize;
        oTargetSize = targetSize; % .* frameSize ./ newFrameSize;
        bboxes(i, :) = [oTargetPosition([2,1]) - oTargetSize([2,1])/2, oTargetSize([2,1])];

        if p.visualization
            if isempty(videoPlayer)
                figure(1), imshow(flow/255);
                figure(1), rectangle('Position', rectPosition, 'LineWidth', 4, 'EdgeColor', 'y');
                drawnow
                fprintf('Frame %d\n', startFrame+i);
            else
%                 score_vis(net_x_rgb, scoreId, z_features, flow, rectPosition, p, v_score);
                im = gather(im)/255;
                im = insertShape(im, 'Rectangle', rectPosition, 'LineWidth', 4, 'Color', 'yellow');
                % Display the annotated video frame using the video player object.
                step(videoPlayer, im);
                writeVideo(v_img, im);
                
                flow = gather(flow)/255;
                flow = insertShape(flow, 'Rectangle', rectPosition, 'LineWidth', 4, 'Color', 'yellow');
                writeVideo(v_flow, flow);
            end
        end

        if p.bbox_output
            fprintf(p.fout,'%.2f,%.2f,%.2f,%.2f\n', bboxes(i, :));
        end

    end
    
    bboxes = bboxes(startFrame : i, :);
    close(v_img);
    close(v_flow);
%     close(v_score);
    close(v_x_crop);
    close(v_x_score);

end
