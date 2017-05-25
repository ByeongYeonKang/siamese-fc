function runAll_vot15

    visualization = true;
    gpus = 1;

    seq_base_path = '../results/VOT/'; % (optional)
    seq = clean_dir_only_folder(seq_base_path);
  
    for i=1:numel(seq)
       run_tracker(seq{i}, visualization, gpus);
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