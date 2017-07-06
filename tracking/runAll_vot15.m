function runAll_vot15(dataset, startSeq)
    
    visualization = true;
    gpus = 1;

    switch dataset
        case 'OTB'
            RGB_path = '../dataset/OTB/';
            flow_path = '../dataset/OTB_OF/';
            save_path = '../results/OTB/';
            save_name = 'RGB_base';
        case 'VOT'
            RGB_path = '../dataset/VOT/';
            flow_path = '../dataset/VOT_OF/';
            save_path = '../results/VOT/';
            save_name = 'RGB_base';
    end
    
    path = struct();
    path.RGB_path = RGB_path;
    path.flow_path = flow_path;
    path.save_path = save_path;
    path.save_name = save_name;    
    seq = clean_dir_only_folder(path.RGB_path);
  
    for i=startSeq:numel(seq)
        seq{i}
        run_tracker(path, dataset, seq{i}, visualization, gpus);
    end
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