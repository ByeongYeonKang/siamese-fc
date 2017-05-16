% -------------------------------------------------------------------------------------------------
function [flows, imgs, pos, target_sz, video_path] = load_video_info2(seq_base_path, is_groundTruth)
%LOAD_VOT_VIDEO_INFO
% -------------------------------------------------------------------------------------------------
    %%  load all jpg files in the folder
    [img_file,video_path] = uigetfile([seq_base_path '*.*']);
    [pathstr,name,ext] = fileparts(img_file);

    %load all jpg files in the folder
    img_files = dir([video_path ['*' ext]]);
    assert(~isempty(img_files), 'No image files to load.')
    img_files = sort({img_files.name});
    img_files = strcat(video_path, img_files);
    % read all frames at once
    imgs = vl_imreadjpeg(img_files,'numThreads', 12);
    %% load all flow files for flow net
    [flow_file,video_path] = uigetfile([seq_base_path '*.*']); 
    flow_files = dir([video_path ['*' ext]]);
    assert(~isempty(flow_files), 'No image files to load.')
	flow_files = sort({flow_files.name});
    flow_files = strcat(video_path, flow_files);
    % read all frames at once
    flows = vl_imreadjpeg(flow_files,'numThreads', 12);
    
    %% if u have ground truth, let's pick up the file on the folder, otherwise do not
    if is_groundTruth  
        [gt_file,gt_path] = uigetfile([video_path '*.txt']);
        groundTruths = importdata([gt_path gt_file]);
        targetPosition = groundTruths(1,:);
    else
        % show a selected frame and get the rectangle using mouse
        imshow([video_path img_file]);
        targetPosition = getrect;
    end   
    [cx, cy, w, h] = get_axis_aligned_BB(targetPosition);
    pos = [cy cx]; % centre of the bounding box
    target_sz = [h w];
end




    