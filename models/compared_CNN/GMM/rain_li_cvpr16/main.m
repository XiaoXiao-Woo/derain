clear;close all; clc;
addpath(genpath(pwd));

out_dir = './tmp_r/';

file_path = strcat('./tmp', '/');%'D:/ProjectSets/NDA/Attention/SwinIR/data/rain100H/rainy'
path_list = [dir(strcat(file_path,'*.jpg')); dir(strcat(file_path,'*.png'))];
img_num = length(path_list);
disp(img_num);
if img_num > 0 
    for j = 1:img_num
        image_name = path_list(j).name;
        % I = imread('case1.jpg');
        I = imread(strcat(file_path,image_name));
        [h, w, c] = size(I);
        I = imresize(I,0.5);
        YUV = rgb2ycbcr(I);
        Y = YUV(:,:,1);

       %% train GMM
        Y = im2double(Y);

        NEW_RAIN_GMM = true;
        if (NEW_RAIN_GMM)
            rainModel = onlineGMM(Y,20);
        else
            rainModel = load('rain_offline.mat');
        end
        load ('GSModel_8x8_200_2M_noDC_zeromean.mat'); %pre trained model
        bcgdModel = GS;
        clear GS;

        tic,
        [B_out R_out] = GMM_Decomp(Y, bcgdModel, rainModel);
        toc,

        YUV(:,:,1) = im2uint8((Y));

        rgb = ycbcr2rgb(YUV);
        rgb = imresize(rgb, [h, w]);
        fprintf('%d, %s\n', j, image_name);
        imwrite(rgb, strcat(out_dir,image_name));
        %figure,imshow(ycbcr2rgb(YUV));
        %print('-depsc', strcat(out_dir,image_name))
    
    end
end
% figure,imshow(ycbcr2rgb(YUV));
% figure,imshow(R_out+0.5);

