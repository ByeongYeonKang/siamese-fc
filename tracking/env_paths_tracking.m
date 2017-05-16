function p = env_paths_tracking(p)

    p.net_rgb = '2016-08-17_gray025.net.mat';
    p.net_flow = '2017-05-07_flow100.net.mat';
    
    p.net_base_path = '../nets/';
    p.seq_base_path = '../dataset/VOT/';
    p.save_path = '/mnt/kist2/git/experiment/'
    p.seq_vot_base_path = '/path/to/VOT/evaluation/sequences/'; % (optional)
    p.stats_rgb_path = '../ILSVRC15-curation/optical_flow_stats.mat'; % (optional)
    p.stats_flow_path = '../ILSVRC15-curation/optical_flow_stats.mat'; % (optional)
    %p.stats_path = ' '; % (optional)

end
