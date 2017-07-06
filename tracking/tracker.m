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
    p.path = struct();
    p.net_rgb= ' ';
    p.net_flow= ' ';
    p.stats_rgb_path= ' ';
    p.stats_flow_path= ' ';
    %% execution, visualization, benchmark
    p.dataset = 'OTB';
    p.video = 'bag';
    p.visualization = false;
    p.gpus = [];
    %%
    p.bbox_output = false;
    p.video_output = false;
    p.fout = -1;
    %% Params from the network architecture, have to be consistent with the training
    p.exemplarSize = 127;  % input z size
    p.instanceSize = 255;  % input x size (search region)
    p.scoreSize = 17;
    p.totalStride = 8;
    p.contextAmount = 0.3; % context amount for the exemplar
    p.subMean = false;
    %% stack size for motion information
    p.Stack = true;
    p.LSize = 3; 
    p.LInterVal = 1;
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
    %p = params_5s(p);
    
    %% init u can select option, rgb or flow or two-stream(rgb and flow)
    net = struct();
    %useNets = {'SiamFC', p.path.save_name};
    useNets = {'SiamFC'};
    net_path = {[p.net_base_path p.net] ,[p.net_base_path p.net_flow]}; 
    %stats_path = {p.stats_rgb_path, p.stats_flow_path};
    %data_path = {p.path.RGB_path, p.path.flow_path};
    stats_path = {p.stats_flow_path, p.stats_flow_path};
    data_path = {p.path.flow_path, p.path.flow_path};
    for i=1:length(useNets)
        net(i).name = useNets{i};
        net(i).path = net_path{i};
        net(i).stats_path = stats_path{i};
        net(i).data_path = data_path{i};
    end
    
    %% init model setting 
    for i= 1:length(useNets)
        % get the first frame of the video
        startFrame = 1;
        switch p.video
            case 'David'
                startFrame = 300; 
        end
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
        
        [net(i).imgFiles, net(i).targetPosition(1,:), net(i).targetSize(1,:), net(i).ground_truth] = load_video_info(net(i).data_path, p.dataset, p.video, startFrame);
        frames = size(net(i).ground_truth,1);
        net(i).endFrame = startFrame+frames-1;
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
        if p.Stack && i==1
            stack_ind = 3-(p.LInterVal*(p.LSize-1)):p.LInterVal:3;
            % stack_ind = 1 + startFrame;
            for s=1:length(stack_ind)
                im = gpuArray(single(net(i).imgFiles{stack_ind(s)}));
                % if grayscale repeat one channel to match filters size
                if(size(im, 3)==1)
                    im = repmat(im, [1 1 3]);
                end
                [z_crop(:,:,1+(s-1)*3:s*3), ~] = get_subwindow_tracking(im, net(i).targetPosition, [p.exemplarSize p.exemplarSize], [round(s_z) round(s_z)], net(i).avgChans);
            end       
            t_alpha= 50>mean(mean(var(z_crop(:,:,:))))
        else 
            [z_crop, ~] = get_subwindow_tracking(im, net(i).targetPosition, [p.exemplarSize p.exemplarSize], [round(s_z) round(s_z)], net(i).avgChans);
        end
               
        if p.subMean && i==1
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
 %       deb_save_name = [p.path.save_path 'bad_env/' p.video '_exemplar.jpg']; 
 %       imwrite(gather(z_crop)/255,deb_save_name);    
        % evaluate the offline-trained network for exemplar z features
        
        figure(1); clf; subplot(1,3,1); h1= imshow(z_crop(:,:,1:3)/255); axis off; axis image; title('flow(t-3,t-2)');
        figure(1); subplot(1,3,2); h2= imshow(z_crop(:,:,4:6)/255); axis off; axis image; title('flow(t-2,t-1)');
        figure(1); subplot(1,3,3); h3= imshow(z_crop(:,:,7:9)/255); axis off; axis image; title('flow(t-1,t)');
         
        net_z(i).eval({'exemplar', z_crop});
        net(i).z_features = net_z(i).vars(zFeatId).value;
        net(i).z_features = repmat(net(i).z_features, [1 1 1 p.numScale]);
        
        net(i).bboxes = zeros(net(i).nImgs, 4);
%         if p.bbox_output 
%             p.fout(i) = fopen([p.save_path p.video '/' bbox_path{i}], 'w');
%         end
    end
    %% Init visualization for debug
    % p.save_path = [p.save_path p.video];
    if ~isdir(p.path.save_path)
        mkdir(p.path.save_path)
    end
    videoPlayer = [];
    if p.visualization && isToolboxAvailable('Computer Vision System Toolbox')
        videoPlayer = vision.VideoPlayer('Position', [100 100 [size(im,2), size(im,1)]+30]);
    end
    if p.video_output
        if ~isdir([p.path.save_path p.path.save_name '/video/'])
            mkdir([p.path.save_path p.path.save_name '/video/'])
        end
        result_video = VideoWriter([p.path.save_path p.path.save_name '/video/' p.video '_' char(useNets(1))]);
        open(result_video);
    end
    
    %% start tracking
    tic;
    deb_save = 1;
    %% nof = endFrame - startFrame;
    for nof = startFrame:net(1).endFrame-10          
        i = (nof-startFrame)+1;
        if nof>startFrame
            % load new frame on GPU
            for ii=1:length(useNets)
                if p.Stack && ii==1 && mod(i,16)==8
                    %stack_ind = nof-(p.LInterVal*(p.LSize-1)):p.LInterVal:nof;
                    stack_ind = nof-(3*(p.LSize-1)):3:nof;
                    % stack_ind = nof-1;
                    for s=1:length(stack_ind)
                        im = gpuArray(single(net(1).imgFiles{stack_ind(s)-1}));
                        % if grayscale repeat one channel to match filters size
                        if(size(im, 3)==1)
                            im = repmat(im, [1 1 3]);
                        end
                        [z_crop(:,:,1+(s-1)*3:s*3), ~] = get_subwindow_tracking(im, net(1).targetPosition((i-1),:), [p.exemplarSize p.exemplarSize], [round(s_z) round(s_z)], net(1).avgChans);
                    end
%                     t_alpha= 50>mean(mean(var(z_crop(:,:,:))))
                    figure(1); clf; subplot(1,3,1); h1= imshow(z_crop(:,:,1:3)/255); axis off; axis image; title('flow(t-3,t-2)');
                    figure(1); subplot(1,3,2); h2= imshow(z_crop(:,:,4:6)/255); axis off; axis image; title('flow(t-2,t-1)');
                    figure(1); subplot(1,3,3); h3= imshow(z_crop(:,:,7:9)/255); axis off; axis image; title('flow(t-1,t)');
                    net_z(1).eval({'exemplar', z_crop});   
                    net(1).z_features = net_z(1).vars(zFeatId).value;
                    net(1).z_features = repmat(net(1).z_features, [1 1 1 p.numScale]);
                end
                scaledInstance = net(ii).s_x(i-1) .* scales;
                scaledTarget = [net(ii).targetSize(i-1,1) .* scales; net(ii).targetSize(i-1,2) .* scales];
                % evaluate the offline-trained network for exemplar x features
                if strcmp(useNets(ii),'SiamFC')
                    % extract scaled crops for search region x at previous target position
                    im = gpuArray(single(net(ii).imgFiles{nof}));
                    % if grayscale repeat one channel to match filters size
                    if(size(im, 3)==1)
                        im = repmat(im, [1 1 3]);
                    end
                    [net(ii).x_crops, net(ii).image_coord_roi, net(ii).score_coord_roi] = make_scale_pyramid(im, net(ii).targetPosition(i-1,:), scaledInstance, p.instanceSize, net(ii).avgChans, net(ii).stats, p);
                    [newTargetPosition, newScale, scoreMap] = tracker_eval(net_x(ii), round(net(ii).s_x(i-1)), scoreId, net(ii).z_features, net(ii).x_crops, net(ii).targetPosition(i-1,:), window, p);
                else
                    im = gpuArray(single(net(1).imgFiles{nof}));
                    if(size(im, 3)==1)
                        im = repmat(im, [1 1 3]);
                    end
                    [net(ii).x_crops, net(ii).image_coord_roi, net(ii).score_coord_roi] = make_scale_pyramid(im, net(ii).targetPosition(i-1,:),...
                        scaledInstance, p.instanceSize, net(ii).avgChans, net(ii).stats, p);
                    [newTargetPosition, newScale, scoreMap] = tracker_eval(net_x(ii), round(net(ii).s_x(i-1)), scoreId, net(ii).z_features, net(ii).x_crops, net(ii).targetPosition(i-1,:), window, p);
%                     [RGB_x_crops,] = make_scale_pyramid(im, net(ii).targetPosition(i-1,:), scaledInstance, p.instanceSize, net(ii).avgChans, net(ii).stats, p);
%                     im = gpuArray(single(net(2).imgFiles{nof-1}));
%                     if(size(im, 3)==1)
%                         im = repmat(im, [1 1 3]);
%                     end
%                     [flow_x_crops, net(ii).image_coord_roi, net(ii).score_coord_roi] = make_scale_pyramid(im, net(ii).targetPosition(i-1,:),...
%                         scaledInstance, p.instanceSize, net(ii).avgChans, net(ii).stats, p);
%                     if p.subMean
%                         flow_x_crops = bsxfun(@minus, flow_x_crops, reshape(net(ii).stats.z.rgbMean, [1 1 3]));
%                     end
%                     [newTargetPosition, newScale, scoreMap] = two_trackers_eval(net_x(1), net_x(2), round(net(2).s_x(i-1)), scoreId,...
%                         net(1).z_features, net(2).z_features, RGB_x_crops, flow_x_crops, net(2).targetPosition(i-1,:), window, p, t_alpha);
                end
                
                figure(2); clf; subplot(1,3,1); h4=imshow(net(ii).x_crops(:,:,:,1)/255); axis off; axis image; title('s_win(scale1)');
                figure(2); subplot(1,3,2); h5=imshow(net(ii).x_crops(:,:,:,2)/255); axis off; axis image; title('s_win(scale2)');
                figure(2); subplot(1,3,3); h6=imshow(net(ii).x_crops(:,:,:,3)/255); axis off; axis image; title('s_win(scale3)');
                net(ii).targetPosition(i,:) = gather(newTargetPosition);
                scoreMap = imresize(scoreMap, net(ii).image_coord_roi(5)/size(scoreMap,1));
                pro_map = scoreMap;
                scoreMap = scoreMap - min(scoreMap(:)) ;
                scoreMap = scoreMap / max(scoreMap(:)) ;
                N = 256;
                IN = round(N * (scoreMap-min(scoreMap(:)))/(max(scoreMap(:))-min(scoreMap(:))));
                cmap = jet(N); % see also hot, etc.
                scoreMap = ind2rgb(IN,cmap);
                yy = net(ii).score_coord_roi(2): net(ii).score_coord_roi(4);
                xx = net(ii).score_coord_roi(1): net(ii).score_coord_roi(3);
                net(ii).map = scoreMap(yy,xx,:);
                pro_map = pro_map(yy,xx,:);
                
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
                im = gpuArray(single(net(1).imgFiles{nof}));
                im = gather(im)/255;
                color = {'red', 'blue','green'};
                rectPosition = cvt_to_rect(net(1).ground_truth(i,:));
                im = insertShape(im, 'Rectangle', gather(rectPosition), ...
                        'LineWidth', 4, 'Color', color{3});
                for ii=1:length(useNets)
                    rectPosition = [net(ii).targetPosition(i,[2,1]) - net(ii).targetSize(i,[2,1])/2, net(ii).targetSize(i,[2,1])];
                    im = insertShape(im, 'Rectangle', gather(rectPosition), ...
                        'LineWidth', 4, 'Color', color{ii});
                end
                % Draw Search-region and score map
                if nof>startFrame
                    s_map = zeros(size(im));
                    ss_map = zeros(size(im,1),size(im,2));
                    color = {'red','green'};
                    for ii=1:length(useNets)
                        rect = net(ii).image_coord_roi;
                        s_map(rect(2):rect(4),rect(1):rect(3),:) = net(ii).map;
                        im = insertShape(im, 'Rectangle', [rect(1), rect(2), rect(3)-rect(1), rect(4)-rect(2)], ...
                        'LineWidth', 5, 'Color', color{ii});
                    end
                    alpha = 0.5;
                    im = im * (1-alpha) + s_map*alpha;
                    
                    rectPosition = cvt_to_rect(net(1).ground_truth(i,:));
                    rl = [rectPosition(1), rectPosition(2), rectPosition(1)+rectPosition(3), rectPosition(2)+rectPosition(4)];
                    if rl(1) < 1
                        rl(1) = 1;
                    end 
                    if rl(2) < 1
                        rl(2) = 1;
                    end 
                    if rl(3) > size(im,2)
                        rl(3) = size(im,2);
                    end
                    if rl(4) > size(im,1)
                        rl(4) = size(im,1);
                    end 
                    ss_map(rect(2):rect(4),rect(1):rect(3)) = pro_map;
                    gt_mean = mean(mean(ss_map(rl(2):rl(4),rl(1):rl(3))));
                    search_mean = mean(mean(ss_map));
                    if gt_mean < 0.5 && deb_save
                        net(ii).targetPosition(i,:) = [rectPosition(2)+rectPosition(4)/2, rectPosition(1)+rectPosition(3)/2]
                        deb_save = 0;
                        deb_save_name = [p.path.save_path 'bad_env/' p.video '.jpg']; 
                        imwrite(im,deb_save_name);    
                        deb_save_name = [p.path.save_path 'bad_env/' p.video '_search.jpg']; 
                        imwrite(gather(net(1).x_crops(:,:,:,newScale))/255,deb_save_name);    
                    end 
                end
                % Display the annotated video frame using the video player object.
                step(videoPlayer, im);
                writeVideo(result_video, im);
            end
        end
        if p.bbox_output
            for ii=1:length(useNets)
                rectPosition = [net(ii).targetPosition(i,[2,1]) - net(ii).targetSize(i,[2,1])/2, net(ii).targetSize(i,[2,1])];
                net(ii).bboxes(i, :) = rectPosition;
%                 fprintf(p.fout(ii),'%.2f,%.2f,%.2f,%.2f\n', rectPosition);
            end
        end
    end

    if p.visualization        
        close(result_video);
    end
    % useNets = {'RGB', 'flow'};
    for ii = 1:length(useNets)
        results = cell(1,20);
        results{1}.type = 'rect';
        results{1}.res = net(ii).bboxes(1 : frames, :);
        results{1}.len = frames;
        results{1}.annoBegin = startFrame;
        results{1}.startFrame = startFrame;
        results{1}.anno = net(ii).ground_truth;
        ss = p.video;
        ss(1) = lower(ss(1));
        save_name = [p.path.save_path p.path.save_name '/' ss '_' char(useNets(ii)) '.mat']; 
        save(save_name,'results'); 
    end
    
    
end
