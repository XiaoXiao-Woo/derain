clear all
close all
% Path = ['E:\structureandtexture\Images\Training\Images\']


 P{1} = [-1,1];
  P{2} = [-1,0,1];

f_size = 7;
K      = 4;
MaxIter = 14;

for index = 1:12

        
    temppath = ['input\' num2str(index) '_in.png'];
    image = imread(temppath);      
    temppath = ['input\' num2str(index) '_gt.png'];
    gt_image = double(imread(temppath));
 
        
    image = double(rgb2ycbcr(image));
    in_image = double(image(:,:,1))/255;       
    in_image = padarray(in_image,[1 1]*6,'symmetric','both');
    for a = 1:4
       in_image = edgetaper(in_image,fspecial('gaussian',6,1));
    end

    iter = 1;
    T    = zeros(size(in_image));
    PreS = zeros(size(in_image));
    alpha = 0.005;
    gamma = 0.02;

    while(iter<MaxIter)

        iter
        [A,S]=AnalysisSC( in_image-T, P, alpha,1,300 );
        difference = mean((PreS(:)-S(:)).^2)
        PreS = S;

        if(iter==1)
            D = InitDictNN( in_image-S, f_size, K );
        end

        [ Z, T ] = SynthesisSC_NN( in_image-S, D, gamma, 1, 300 );

         [D]   = UpdateFilter_NN( in_image-S, single(Z), D, f_size, K, 300 );

         temp = image;
         temp(:,:,1) = S(7:end-6,7:end-6)*255;
         temp = ycbcr2rgb(temp/255);

        Structure(:,:,iter) = S;
        Texture(:,:,iter) = T;
        Dictionary(:,:,:,iter) = D;

         psnr(index,iter) = csnr(gt_image,temp*255,5,5);
         ssim(index,iter) = cal_ssim(gt_image,temp*255,5,5);

        iter = iter+1;
        if(difference<1e-10)
         iter = MaxIter+1;
        end

    end
%     iter = iter-1;
%     path = [ num2str(index) '_result_JCAS_005_01.png'];
%     imwrite(temp,path);

end
% save case2order_result_data psnr ssim