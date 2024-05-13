function tdout = readRealTimeTrackFile(trackfile)
fid = fopen(trackfile);
raw = fread(fid,inf);
str = char(raw');
fclose(fid);
tdin = jsondecode(str);
tdout = struct;
fns = fieldnames(tdin.track);
for i = 1:numel(fns),
  tdout.(fns{i}) = [td.track.(fns{i})];
end
