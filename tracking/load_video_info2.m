% -------------------------------------------------------------------------------------------------
function [imgs, pos, target_sz, video_path] = load_video_info2(seq_base_path, isgt)
%LOAD_VOT_VIDEO_INFO
% -------------------------------------------------------------------------------------------------
    %full path to the video's files
    [img_file,video_path] = uigetfile([seq_base_path '*']);
    [pathstr,name,ext] = fileparts(img_file); 
    % show a selected frame.
    imshow([video_path img_file]);
    
    if isgt
        % gt path
        [gt_file,gt_path] = uigetfile('*.txt');
        gt = importdata([gt_path gt_file]);
        region = gt(1,:)
    else
        % lets you select a rectangle in the current axes of figure fig using the mouse.
        region = getrect
    end    

    [cx, cy, w, h] = get_axis_aligned_BB(region);
    pos = [cy cx]; % centre of the bounding box
    target_sz = [h w];
        
	%load all jpg files in the folder
	img_files = dir([video_path ['*' ext]]);
	assert(~isempty(img_files), 'No image files to load.')
	img_files = sort({img_files.name});

	%eliminate frame 0 if it exists, since frames should only start at 1
	% img_files(strcmp('00000000.jpg', img_files)) = [];
    img_files = strcat(video_path, img_files);
    
    % read all frames at once
    imgs = vl_imreadjpeg(img_files,'numThreads', 12);
end




    