function eval_network
    eval = struct();
    
    seq_base_path = '../results/VOT/'; % (optional)
    seq = clean_dir_only_folder(seq_base_path);
  
    for i=1:numel(seq)
       eval(i).name = seq{i};       
       bbox_output = get_bbox_output([seq_base_path seq{i}]);
       ground_truth = get_ground_truth([seq_base_path seq{i}]);
       
       nums = size(bbox_output,1);
       bbox = zeros(nums,4);
       gt = zeros(nums, 4);
       for ii=1:nums
           bbox(ii,:) = get_axis_aligned_BB(bbox_output(ii,:));
           gt(ii,:) = get_axis_aligned_BB(ground_truth(ii,:));
       end         
       eval(i).IoU = cal_IoU(bbox, gt);
       eval(i).dist = cal_dist(bbox, gt);
    end
    
end

function file = get_bbox_output(file_path)
    file = csvread([file_path '/' 'bbox_output.txt']);
end

function file = get_ground_truth(file_path)
    file = csvread([file_path '/' 'groundtruth.txt']);
end

function rst = cal_IoU(groundTruth, boundingBox)
    for i=1:size(groundTruth,1)
        rst(i) = bboxOverlapRatio(rect_from_region(groundTruth(i,:)), ...
            rect_from_region(boundingBox(i,:)), 'Union');        
    end
end

function rst = cal_dist(groundTruth, bbox)
    for i=1:size(groundTruth,1)
        rst(i) = sqrt((groundTruth(i,1)-bbox(i,1)).^2 +...
            (groundTruth(i,2)-bbox(i,2)).^2);
    end
end
function rect = rect_from_region(region)
    w = region(3);
    h = region(4); 
    x = (region(1)+w)/2;
    y = (region(2)+h)/2;
    rect = [x,y,w,h];
end

function files = clean_dir_only_folder(base)
  %clean_dir just runs dir and eliminates files in a foldr
  files = dir(base);
  dirFlags = [files.isdir];
  files = files(dirFlags);
  files_tmp = {};
  for i = 1:length(files)
    if strncmpi(files(i).name, '.',1) == 0
      files_tmp{length(files_tmp)+1} = files(i).name;
    end
  end
  files = files_tmp; 
end

% -------------------------------------------------------------------------------------------------
function rect = get_axis_aligned_BB(region)
%GETAXISALIGNEDBB computes axis-aligned bbox with same area as the rotated one (REGION)
% -------------------------------------------------------------------------------------------------
    nv = numel(region);
    assert(nv==8 || nv==4);

    if nv==8
        cx = mean(region(1:2:end));
        cy = mean(region(2:2:end));
        x1 = min(region(1:2:end));
        x2 = max(region(1:2:end));
        y1 = min(region(2:2:end));
        y2 = max(region(2:2:end));
        A1 = norm(region(1:2) - region(3:4)) * norm(region(3:4) - region(5:6));
        A2 = (x2 - x1) * (y2 - y1);
        s = sqrt(A1/A2);
        w = s * (x2 - x1) + 1;
        h = s * (y2 - y1) + 1;
    else
        x = region(1);
        y = region(2);
        w = region(3);
        h = region(4);
        cx = x+w/2;
        cy = y+h/2;
    end
    rect = [cx, cy, w, h];
end
