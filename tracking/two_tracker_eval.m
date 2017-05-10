function [newTargetPosition, bestScale, responseMap] = two_tracker_eval(net_x_rgb, net_x_flow, s_x, scoreId, z_features_rgb, z_features_flow, x_crops_rgb, x_crops_flow, targetPosition, window, p)
%TRACKER_STEP
%   runs a forward pass of the search-region branch of the pre-trained Fully-Convolutional Siamese,
%   reusing the features of the exemplar z computed at the first frame.
%
%   Luca Bertinetto, Jack Valmadre, Joao F. Henriques, 2016
% -------------------------------------------------------------------------------------------------------------------------
    % forward pass, using the pyramid of scaled crops as a "batch"
    net_x_rgb.eval({p.id_feat_z, z_features_rgb, 'instance', x_crops_rgb});
    net_x_flow.eval({p.id_feat_z, z_features_flow, 'instance', x_crops_flow});
    
    responseMaps_rgb = reshape(net_x_rgb.vars(scoreId).value, [p.scoreSize p.scoreSize p.numScale]);
    responseMaps_flow = reshape(net_x_flow.vars(scoreId).value, [p.scoreSize p.scoreSize p.numScale]);
    
    [responseMap_rgb, bestScale] = choose_scale(p,responseMaps_rgb);
    [responseMap_flow, bestScale_flow] = choose_scale(p,responseMaps_flow);
    
    % make the response map sum to 1
    responseMap_rgb = responseMap_rgb - min(responseMap_rgb(:));
    responseMap_rgb = responseMap_rgb / sum(responseMap_rgb(:));
    
    % make the response map sum to 1
    responseMap_flow = responseMap_flow - min(responseMap_flow(:));
    responseMap_flow = responseMap_flow / sum(responseMap_flow(:));
    
    responseMap = responseMap_flow + responseMap_rgb;
    
    % apply windowing
    responseMap = (1-p.wInfluence)*responseMap + p.wInfluence*window;
    [r_max, c_max] = find(responseMap == max(responseMap(:)), 1);
    [r_max, c_max] = avoid_empty_position(r_max, c_max, p);
    p_corr = [r_max, c_max];
    

%     temp = round(p_corr./p.responseUp)
%     z_features = net_z.vars(zFeatId).value;
%     z_features = repmat(z_features, [1 1 1 p.numScale]);

    % Convert to crop-relative coordinates to frame coordinates
    % displacement from the center in instance final representation ...
    disp_instanceFinal = p_corr - ceil(p.scoreSize*p.responseUp/2);
    % ... in instance input ...
    disp_instanceInput = disp_instanceFinal * p.totalStride / p.responseUp;
    % ... in instance original crop (in frame coordinates)
    disp_instanceFrame = disp_instanceInput * s_x / p.instanceSize;
    % position within frame in frame coordinates
    newTargetPosition = targetPosition + disp_instanceFrame;
end

function [r_max, c_max] = avoid_empty_position(r_max, c_max, params)
    if isempty(r_max)
        r_max = ceil(params.scoreSize/2);
    end
    if isempty(c_max)
        c_max = ceil(params.scoreSize/2);
    end
end
