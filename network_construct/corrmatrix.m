close all; clc;
numSubjects=1035;

for i = 1:numSubjects

    fprintf('subject: %d\n', i)
    subject = sprintf('/vulcan/scratch/mtang/datasets/ABIDE/fmri/tensors/subject%d.mat', i);
    load(subject)

    r=corr(img', roi');
    output = sprintf('/vulcan/scratch/mtang/datasets/ABIDE/fmri/corrmatrix/subject%d.mat', i);
    save(output, 'r', 'label');

end
