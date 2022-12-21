clear all
close all
addpath('input');


P{1} = [-1,1];
P{2} = [-1,0,1];

f_size  = 7;
K       = 4;
MaxIter = 14;


   
image     = imread('./input/rain-077.png');
image     = double(rgb2ycbcr(image));
in_image  = image(:,:,1)/255;       


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


    iter = iter+1;
    if(difference<1e-10)
     iter = MaxIter+1;
    end

end
subplot(1,3,1);
imshow(in_image);
subplot(1,3,2);
imshow(S);
subplot(1,3,3);
imshow(T+0.5);
image(:,:,1) = (S(6:326, 6:486) * 255.0);
rgb = ycbcr2rgb(image / 255.0);
imwrite(rgb, 'jcas_rain-043.png')
