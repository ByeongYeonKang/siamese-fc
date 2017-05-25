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
    %% file name
    p.net_rgb= ' ';
    p.net_flow= ' ';
    p.stats_rgb_path= ' ';
    p.stats_flow_path= ' ';
    %% execution, visualization, benchmark
    p.video = 'bag';
    p.visualization = false;
    p.gpus = [];
    %%
    p.subMean = true;
    p.bbox_output = false;
    p.video_output = false;
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
    % Overwrite default parameters with varargin
    p = vl_argparse(p, varargin);
% -------------------------------------------------------------------------------------------------
    % Get environment-specific default paths.
    p = env_paths_tracking(p);
    p = params_3s(p);
    % p = params_5s(p);
    
    %% init u can select option, rgb or flow or two-stream(rgb and flow)
    net = struct();
    useNets = {'RGB', 'flow'};
    net_path = {[p.net_base_path p.net_rgb] ,[p.net_base_path p.net_flow]}; 
    stats_path = {p.stats_rgb_path, p.stats_flow_path};
    data_path = {p.data_rgb_path, p.data_flow_path};
    bbox_path = {p.rgb_bbox, p.flow_bbox};
    for i=1:length(useNets)
        net(i).name = useNets{i};
        net(i).path = net_path{i};
        net(i).stats_path = stats_path{i};
        net(i).data_path = data_path{i};
    end
    
    %% init model setting 
    for i= 1:length(useNets)
        % Load ImageNet Video statistics
        if exist(net(i).stats_path,'file')
            net(i).stats = load(net(i).stats_path);
        else
            warning('No stats found at %s', net(i).stats_path);
            net(i).stats = [];
        end

        % Load two copies of the pre-trained network 
        net_z(i) = load_pretrained(net(i).path, []);
        net_x(i) = load_pretrained(net(i).path, []);
        
        [net(i).imgFiles, net(i).targetPosition(1,:), net(i).targetSize(1,:)] = load_video_info(net(i).data_path, p.video);
        
        % [imgFiles(i), targetPosition(i), targetSize(i), video_path] = load_video_info2(p.seq_base_path, p.gt);
        
        % Divide the net in 2
        % exemplar branch (used only once per video) computes features for the target
        remove_layers_from_prefix(net_z(i), p.prefix_x);
        remove_layers_from_prefix(net_z(i), p.prefix_join);
        remove_layers_from_prefix(net_z(i), p.prefix_adj);
        % instance branch computes features for search region x and cross-correlates with z features
        remove_layers_from_prefix(net_x(i), p.prefix_z);
        zFeatId = net_z(i).getVarIndex(p.id_feat_z);
        scoreId = net_x(i).getVarIndex(p.id_score);

        % get the first frame of the video
        startFrame = 1;
        net(i).nImgs = numel(net(i).imgFiles);
        im = gpuArray(single(net(i).imgFiles{startFrame}));
        
        % if grayscale repeat one channel to match filters size
        if(size(im, 3)==1)
            im = repmat(im, [1 1 3]);
        end
    
        % get avg for padding
        net(i).avgChans = gather([mean(mean(im(:,:,1))) mean(mean(im(:,:,2))) mean(mean(im(:,:,3)))]);

        wc_z = net(i).targetSize(1,2) + p.contextAmount*sum(net(i).targetSize(1,:));
        hc_z = net(i).targetSize(1,1) + p.contextAmount*sum(net(i).targetSize(1,:));
        s_z = sqrt(wc_z*hc_z);
        scale_z = p.exemplarSize / s_z;
        % initialize the exemplar
        [z_crop, ~] = get_subwindow_tracking(im, net(i).targetPosition, [p.exemplarSize p.exemplarSize], [round(s_z) round(s_z)], net(i).avgChans);
        if p.subMean
            z_crop = bsxfun(@minus, z_crop, reshape(net(i).stats.z.rgbMean, [1 1 3]));
        end
        d_search = (p.instanceSize - p.exemplarSize)/2;
        pad = d_search/scale_z;
        net(i).s_x(1) = s_z + 2*pad;
        % arbitrary scale saturation
        min_s_x = 0.2*net(i).s_x(1);
        max_s_x = 5*net(i).s_x(1);

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
        net_z(i).eval({'exemplar', z_crop});
        net(i).z_features = net_z(i).vars(zFeatId).value;
        net(i).z_features = repmat(net(i).z_features, [1 1 1 p.numScale]);
        
        net(i).bboxes = zeros(net(i).nImgs, 4);
        if p.bbox_output 
            p.fout(i) = fopen([p.save_path p.video '/' bbox_path{i}], 'w');
        end
    end
    %% Init visualization for debug
    p.save_path = [p.save_path p.video];
    if ~isdir(p.save_path)
        mkdir(p.save_path)
    end
    videoPlayer = [];
    if p.visualization && isToolboxAvailable('Computer Vision System Toolbox')
        videoPlayer = vision.VideoPlayer('Position', [100 100 [size(im,2), size(im,1)]+30]);
    end
    if p.video_output
        result_video = VideoWriter([p.save_path '/results_videos_of']);
        open(result_video);
    end
    %% start tracking
    tic;
    for i = startFrame:net(2).nImgs          
        if i>startFrame
            % load new frame on GPU
            % targetPosition, s_x, targetSize is pre-valued 
            for ii=1:length(useNets)
%                 if ii==1
%                     im = gpuArray(single(net(ii).imgFiles{i-1}));
%                     [z_crop, ~] = get_subwindow_tracking(im, net(ii).targetPosition(i-1,:), [p.exemplarSize p.exemplarSize], [round(s_z) round(s_z)], net(ii).avgChans);
%                     net_z(ii).eval({'exemplar', z_crop});
%                     net(ii).z_features = net_z(ii).vars(zFeatId).value;
%                     net(ii).z_features = repmat(net(ii).z_features, [1 1 1 p.numScale]);
%                 end
                im = gpuArray(single(net(ii).imgFiles{i}));
                % if grayscale repeat one channel to match filters size
                if(size(im, 3)==1)
                    im = repmat(im, [1 1 3]);
                end
                scaledInstance = net(ii).s_x(i-1) .* scales;
                scaledTarget = [net(ii).targetSize(i-1,1) .* scales; net(ii).targetSize(i-1,2) .* scales];
                % extract scaled crops for search region x at previous target position
                [x_crops, net(ii).image_coord_roi, net(ii).score_coord_roi] = make_scale_pyramid(im, net(ii).targetPosition(i-1,:), scaledInstance, p.instanceSize, net(ii).avgChans, net(ii).stats, p);
                % evaluate the offline-trained network for exemplar x features
                [newTargetPosition, newScale, scoreMap] = tracker_eval(net_x(ii), round(net(ii).s_x(i-1)), scoreId, net(ii).z_features, x_crops, net(ii).targetPosition(i-1,:), window, p);
                net(ii).targetPosition(i,:) = gather(newTargetPosition);
                scoreMap = imresize(scoreMap, net(ii).image_coord_roi(5)/size(scoreMap,1));
                scoreMap = scoreMap - min(scoreMap(:)) ;
                scoreMap = scoreMap / max(scoreMap(:)) ;
                N = 256;
                IN = round(N * (scoreMap-min(scoreMap(:)))/(max(scoreMap(:))-min(scoreMap(:))));
                cmap = jet(N); % see also hot, etc.
                scoreMap = ind2rgb(IN,cmap);
                yy = net(ii).score_coord_roi(2): net(ii).score_coord_roi(4);
                xx = net(ii).score_coord_roi(1): net(ii).score_coord_roi(3);
                net(ii).map = scoreMap(yy,xx,:);
                
                % scale damping and saturation
                net(ii).s_x(i) = max(min_s_x, min(max_s_x, (1-p.scaleLR)*net(ii).s_x(i-1) + p.scaleLR*scaledInstance(newScale)));
                net(ii).targetSize(i,:) = (1-p.scaleLR)*net(ii).targetSize(i-1,:) + p.scaleLR*[scaledTarget(1,newScale) scaledTarget(2,newScale)];
            end
        else
            
        end

        if p.visualization
            if isempty(videoPlayer)
                for ii=1:length(useNets)
%                     figure(1), imshow(flow/255);
%                     figure(1), rectangle('Position', rectPosition, 'LineWidth', 4, 'EdgeColor', 'y');
%                     drawnow
                    fprintf('Frame %d\n', startFrame+i);
                end
            else
                im = gpuArray(single(net(1).imgFiles{i}));
                im = gather(im)/255;
                color = {'red', 'blue'};
                for ii=1:length(useNets)
                    rectPosition = [net(ii).targetPosition(i,[2,1]) - net(ii).targetSize(i,[2,1])/2, net(ii).targetSize(i,[2,1])];
                    im = insertShape(im, 'Rectangle', gather(rectPosition), ...
                        'LineWidth', 6, 'Color', color{ii});
                end
                % Draw Search-region and score map
                if i>startFrame
                    z = 1;
                    rect = net(z).image_coord_roi;
                    alpha = 0.8;
                    s_map = zeros(size(im));
                    s_map(rect(2):rect(4),rect(1):rect(3),:)=net(z).map;
                    im = im * (1-alpha) + s_map*alpha;
                    im = insertShape(im, 'Rectangle', [rect(1), rect(2), rect(3)-rect(1), rect(4)-rect(2)], ...
                        'LineWidth', 5, 'Color', 'green');
                end
                
                % Display the annotated video frame using the video player object.
                step(videoPlayer, im);
                writeVideo(result_video, im);
            end
        end
        if p.bbox_output
            for ii=1:length(useNets)
                rectPosition = [net(ii).targetPosition(i,[2,1]) - net(ii).targetSize(i,[2,1])/2, net(ii).targetSize(i,[2,1])];
                fprintf(p.fout(ii),'%.2f,%.2f,%.2f,%.2f\n', rectPosition);
            end
        end
    end

    if p.visualization        
        close(result_video);
    end
%     bboxes = bboxes(startFrame : i, :);
end
