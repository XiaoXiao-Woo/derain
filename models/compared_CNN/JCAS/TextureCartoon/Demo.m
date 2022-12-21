clear all
close all

% Runing 3 channels decomposition takes more time, the bottleneck is the synthesis
% model part (convolutional sparse coding and dictionary updating), using CSC_ADMM_GPU_Color
% helps to reduce some running time.


P{1} = [-1,1];
P{2} = [-1,0,1];
P{3} = [-1;1];
P{4} = [-1;0;1];

f_size  = 5;
K       = 10;
MaxIter = 14;


   
image = imread('input/5.jpg');
in_image = im2double(image);       


in_image = padarray(in_image,[1 1]*6,'symmetric','both');
for a = 1:4
   in_image = edgetaper(in_image,fspecial('gaussian',6,1));
end


iter = 1;
T = zeros(size(in_image));
PreS = zeros(size(in_image));

%You may need to adjust the parameters for different images
alpha = 0.05;
gamma = 0.2;
    
while(iter<MaxIter)
  
        iter
        [A,S]=AnalysisSC_Color( in_image-T, P, alpha,1,300 );
        difference = mean((PreS(:)-S(:)).^2)
        PreS = S;
        if(iter==1)
             D = InitDict_Color( in_image-S, f_size, K );
        end

        [ Z, T ] = SynthesisSC_Color( in_image-S, D, gamma, 1, 300 );

        [D]   = UpdateFilter_Color( in_image-S, single(Z), D, f_size, K, 250 );

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

