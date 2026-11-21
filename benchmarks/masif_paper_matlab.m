function masif_paper_matlab(jobs_file, paper_runtime)
% Original all-center paper geometry, isolated failures, one MATLAB per batch.
addpath(genpath(fullfile(paper_runtime,'matlab_libs')));
jobs=jsondecode(fileread(jobs_file));
for i=1:numel(jobs)
    job=jobs(i);folder=job.host_work;receipt=fullfile(folder,'matlab-receipt.json');
    if ~exist(fullfile(folder,'surface.mat'),'file'),continue;end
    result=struct('status','failed','stage','paper_matlab');
    started=tic;
    try
        d=load(fullfile(folder,'surface.mat'));opts.single_chain=1;
        shapes=fullfile(folder,'shapes');coords=fullfile(folder,'coords');
        if ~exist(shapes,'dir'),mkdir(shapes);end
        if ~exist(coords,'dir'),mkdir(coords);end
        t=tic;
        compute_matlab_matrix(fullfile(shapes,'mesh'),d.vertices,d.faces,d.charge,d.hbond,d.hydrophobicity,0,0,0,0,0,opts);
        result.curvature_s=toc(t);
        paths.input=shapes;paths.output=coords;params.radius=12.;t=tic;
        coords_mds_batch(paths,params);
        result.coordinates_s=toc(t);result.status='complete';
    catch exception
        result.error=getReport(exception,'extended','hyperlinks','off');
    end
    result.wall_s=toc(started);fid=fopen(receipt,'w');
    if fid==-1,error('Cannot write MATLAB receipt');end
    fprintf(fid,'%s\n',jsonencode(result));fclose(fid);
end
end
