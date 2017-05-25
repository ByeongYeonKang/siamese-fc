function p = env_paths_tracking(p)

    p.net_rgb = '2016-08-17_gray025.net.mat';
    p.net_flow = '2017-05-23_flow090.net.mat';
    
    p.net_base_path = '../nets/';
    p.data_rgb_path = '../dataset/VOT/';
    p.data_flow_path = '../dataset/VOT_OF/';
    p.save_path = '/mnt/kist2/git/results/VOT/'
    p.rgb_bbox = 'rgb_bbox_output.txt'
    p.flow_bbox = 'flow_bbox_output.txt'
    p.seq_vot_base_path = '/path/to/VOT/evaluation/sequences/'; % (optional)
    p.stats_rgb_path = '../ILSVRC15-curation/ILSVRC2015.stats.mat'; % (optional)
    p.stats_flow_path = '../ILSVRC15-curation/optical_flow_stats.mat'; % (optional)
    %p.stats_path = ' '; % (optional)

end
