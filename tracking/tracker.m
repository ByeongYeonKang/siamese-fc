% -------------------------------------------------------------------------------------------------
function result = tracker(varargin)
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
    p.saveVideos = false;
    p.getRect = false;
    % p.bbox_output = false;
    % p.fout = -1;
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
    RGB = 1; 
    flow = 2;
    p.net = {p.net_rgb ,p.net_flow}; 
    p.stats_path = {p.stats_rgb_path, p.stats_flow_path}; 
    % useNets = {RGB, flow};
    useNets = {RGB};
    imgFiles = struct();
    result = struct();
    
    %% init model setting 
    for i= 1:length(useNets)
        % Load ImageNet Video statistics
        if exist(p.stats_path{i},'file')
            stats(i) = load(p.stats_path{i});
        else
            warning('No stats found at %s', p.stats_path{i});
            stats(i) = [];
        end

        % Load two copies of the pre-trained network 
        net_z(i) = load_pretrained([p.net_base_path p.net{i}], []);
        net_x(i) = load_pretrained([p.net_base_path p.net{i}], []);
        
        [imgFiles, targetPosition, targetSize] = load_video_info(p.seq_base_path, p.video);
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
        nImgs = numel(imgFiles);
        im = gpuArray(single(imgFiles{startFrame}));
        
        % if grayscale repeat one channel to match filters size
        if(size(im, 3)==1)
            im = repmat(im, [1 1 3]);
        end
    
        % get avg for padding
        avgChans = gather([mean(mean(im(:,:,1))) mean(mean(im(:,:,2))) mean(mean(im(:,:,3)))]);

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
        net_z(i).eval({'exemplar', z_crop});
        z_features(i) = net_z_rgb.vars(zFeatId).value;
        z_features(i) = repmat(z_features(i), [1 1 1 p.numScale]);
    end
    %%
    gt_path = ['../dataset/OTB/' p.video];
    gt_file = '/groundtruth_rect.txt';
    gt = importdata([gt_path gt_file]);
    
    % Init visualization
    videoPlayer = [];
    if p.visualization && isToolboxAvailable('Computer Vision System Toolbox')
        videoPlayer = vision.VideoPlayer('Position', [100 100 [size(im,2), size(im,1)]+30]);
    end
    
    p.save_path = [p.save_path p.net_flow '/' p.video];
    if ~isdir(p.save_path)
        mkdir(p.save_path)
    end
    
    if p.visualization
        v_tracking = VideoWriter([p.save_path '/tracking_results']);
        open(v_tracking);
    end
    
    bboxes = zeros(nImgs, 4);
   
    targetPosition_rgb = targetPosition; 
    targetSize_rgb = targetSize;
    targetPosition_flow = targetPosition; 
    targetSize_flow = targetSize;
    rectPosition_gt = zeros(nImgs,4);
    
    % start tracking
    tic;
    for i = startFrame:nImgs          
        if i>startFrame
            % load new frame on GPU
            im = gpuArray(single(imgFiles{i}));
   			% if grayscale repeat one channel to match filters size
            if(size(im, 3)==1)
        		im = repmat(im, [1 1 3]);
            end
            scaledInstance = s_x_rgb .* scales;
            scaledTarget = [targetSize(1) .* scales; targetSize(2) .* scales];
            % extract scaled crops for search region x at previous target position
            x_crops = make_scale_pyramid(im, targetPosition, scaledInstance, p.instanceSize, avgChans, stats, p);
            % evaluate the offline-trained network for exemplar x features
            [newTargetPosition, newScale] = tracker_eval(net_x, round(s_x), scoreId, z_features, x_crops, targetPosition, window, p);
            targetPosition = gather(newTargetPosition);
            % imwrite(gather(x_crops(:,:,:,1))/255,[p.save_path p.net '/' p.video '/track/' num2str(i) '.jpg']);
            % evaluate the offline-trained network for exemplar x features
            %[newTargetPosition, newScale, scoreMap] = tracker_eval(net_x_rgb, round(s_x), scoreId, z_features_rgb, x_crops_rgb, targetPosition, window, p);
            % [newTargetPosition_rgb, newScale_rgb] = tracker_eval(net_x_rgb, round(s_x_rgb), scoreId, z_features_rgb, x_crops_rgb, targetPosition_rgb, window, p);
            %[newTargetPosition_flow, newScale_flow] = tracker_eval(net_x_flow, round(s_x_flow), scoreId, z_features_flow, x_crops_flow, targetPosition_flow, window, p);
            %[newTargetPosition, newScale, scoreMap] = two_tracker_eval(net_x_rgb, net_x_flow, round(s_x), scoreId, z_features_rgb, z_features_flow, x_crops_rgb, x_crops_flow, targetPosition, window, p);
            
            % scale damping and saturation
            s_x = max(min_s_x, min(max_s_x, (1-p.scaleLR)*s_x + p.scaleLR*scaledInstance(newScale)));
            targetSize = (1-p.scaleLR)*targetSize + p.scaleLR*[scaledTarget(1,newScale) scaledTarget(2,newScale)];
            % vis_crop = gather(x_crops_rgb(:,:,:,newScale));
            % writeVideo(v_x_crop, mat2gray(vis_crop));
            % vis_score_map(scoreMap, targetPosition, targetSize, v_x_score);
        else
            
        end
        
%         if ~isnan(gt(i,1))
%             rectPosition_gt = gt(i,:);
%             [cx, cy, w, h] = get_axis_aligned_BB(rectPosition_gt);
%             targetPosition_gt = [cy cx]; % centre of the bounding box
%             % targetSize_gt = [h w];    
%         end
        rectPosition = [targetPosition([2,1]) - targetSize([2,1])/2, targetSize([2,1])];
        bboxes(i, :) = rectPosition;

        if p.visualization
            if isempty(videoPlayer)
                figure(1), imshow(flow/255);
                figure(1), rectangle('Position', rectPosition, 'LineWidth', 4, 'EdgeColor', 'y');
                drawnow
                fprintf('Frame %d\n', startFrame+i);
            else
                % score_vis(net_x_rgb, scoreId, z_features, flow, rectPosition, p, v_score);
                im = gather(im)/255;
                im = insertShape(im, 'Rectangle', rectPosition_gt, ...
                    'LineWidth', 4, 'Color', 'red');
                im = insertShape(im, 'Rectangle', rectPosition_rgb, ...
                    'LineWidth', 4, 'Color', 'green');
                im = insertShape(im, 'Rectangle', rectPosition_flow, ...
                    'LineWidth', 4, 'Color','blue');
                
                text_str = cell(2,1);
                overlap_ratio = [bboxOverlapRatio(rectPosition_gt,...
                    rectPosition_rgb, 'Union') ...
                    bboxOverlapRatio(rectPosition_gt,...
                    rectPosition_flow, 'Union') ];
                distance = [sqrt((targetPosition_rgb(1)-targetPosition_gt(1)).^2)+ ...
                    sqrt((targetPosition_rgb(2)-targetPosition_gt(2)).^2), ...
                    sqrt((targetPosition_flow(1)-targetPosition_gt(1)).^2)+ ...
                    sqrt((targetPosition_flow(2)-targetPosition_gt(2)).^2)];
                for ii=1:2
                   text_str{ii} = ['IoU:' num2str(overlap_ratio(ii),'%0.2f') '%', ' Dist:' num2str(distance(ii),'%0.2f')];
                end
                position = [size(im,2)-150 size(im,1)-80; size(im,2)-150 size(im,1)-60];
                box_color = {'green','blue'};
                im = insertText(im, position,text_str,'FontSize',8,'BoxColor',...
                    box_color,'BoxOpacity',0.4,'TextColor','white');
                
                % Display the annotated video frame using the video player object.
                step(videoPlayer, im);
                writeVideo(v_tracking, im);
            end
        end

        if p.bbox_output
            result.GT.bbox(i,:) = rectPosition_gt;
            result.GT.center(i,:) = targetPosition_gt;
            result.TR_RGB.bbox(i,:) = rectPosition_rgb;
            result.TR_RGB.center(i,:) = targetPosition_rgb;
            result.TR_RGB.dist(i) = distance(1);
            result.TR_RGB.IoU(i) = overlap_ratio(1);
            result.TR_flow.bbox(i,:) = rectPosition_flow;
            result.TR_flow.center(i,:) = targetPosition_flow;
            result.TR_flow.dist(i) = distance(2);
            result.TR_flow.IoU(i) = overlap_ratio(2);
            
            % fprintf(p.fout,'GT: %.2f, %.2f TR: %d,%d,%d,%d\n', round(bboxes(i, :)));
            % fprintf(p.fout,'%d,%.2f,%.2f,%.2f\n', bboxes(i, :));
        end

    end
    
    result.TR_RGB.precision(i) = precision_auc(result.TR_RGB.center,...
        result.GT.center, 0.5, nImgs);
    result.TR_flow.precision(i) = precision_auc(result.TR_flow.center,...
        result.GT.center, 0.5, nImgs);
    
    figure;
    plot(1:nImgs,result.TR_RGB.dist,'g',1:nImgs,result.TR_flow.dist,'b');
    title('Distance Plot')
    legend('RGB Dist', 'flow Dist')
    xlabel('frames')
    ylabel('distance')
    saveas(gcf,[p.save_path '_dist.png'])
    
    figure;
    plot(1:nImgs,result.TR_RGB.IoU,'g',1:nImgs,result.TR_flow.IoU,'b');
    title('IoU Plot')
    legend('RGB IoU', 'flow IoU')
    xlabel('frames')
    ylabel('IoU %')
    saveas(gcf,[p.save_path '_IoU.png'])
    
%     DataField = fieldnames(result);
%     dlmwrite('myFile.txt', result, 'delimiter','\t','newline','pc')
%     dlmwrite('FileName.txt', result.(DataField{1}));
    % bboxes = bboxes(startFrame : i, :);
    
    if p.visualization        
        close(v_tracking);
    end

end
