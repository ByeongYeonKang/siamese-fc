function p = env_paths_tracking(p)

    p.net = 'net-epoch-18.mat';
    p.net_rgb = '2016-08-17_gray025.net.mat';
    p.net_flow = '2017-06-04_flow050.net.mat';
    
    p.net_base_path = '../nets/';
    p.seq_vot_base_path = '/path/to/VOT/evaluation/sequences/'; % (optional)
    p.stats_rgb_path = '../ILSVRC15-curation/ilsvrc_rgb_stats.mat'; % (optional)
    p.stats_flow_path = '../ILSVRC15-curation/ilsvrc_of_stats.mat'; % (optional)
    %p.stats_path = ' '; % (optional)

end
