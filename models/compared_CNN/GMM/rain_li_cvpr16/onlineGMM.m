function model_out = onlineGMM(I, gNum)

% I = im2double(imread('01R.png'));

finish=false;
set(gcf,'CurrentCharacter','@'); % set to a dummy character
%figure(1),imshow(I);
%sdisp('Please select the rain region on the image.');

%[x,y] = ginput(2);
patch = I;%(round(y(1)):round(y(2)),round(x(1)):round(x(2)),:);
B = im2col(patch,[8 8]);
    
B = bsxfun(@minus, B, mean(B)); %remove mean

pvars = var(B, 0, 1);
idx = pvars > 1e-5;
B = B(:, idx);

clusterNum = gNum;
[label, model, llh] = my_emgm(B, clusterNum);
model_out = struct( 'means', model.mu, 'covs',model.Sigma, 'mixweights', model.weight, 'dim', 64, 'nmodels', clusterNum);

close all;