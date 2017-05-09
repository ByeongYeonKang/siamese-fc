function p = env_paths_tracking(p)

    p.net_base_path = '../nets/';
    p.seq_base_path = '../dataset/';
    p.save_path = '/mnt/kist2/git/experiment/'
    p.seq_vot_base_path = '/path/to/VOT/evaluation/sequences/'; % (optional)
    p.stats_path = '../ILSVRC15-curation/optical_flow_stats.mat'; % (optional)

end
