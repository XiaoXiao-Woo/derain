function [ D ] = InitDictNN( Image, f_size, K )
% Initialize Non-negative dictionary with PCA

[m n] = size(Image);

Patches =  im2col(Image, [f_size,f_size]);

    
Patches = Patches- repmat(mean(Patches,2),1,size(Patches,2));

temp = Patches*Patches';
[U S V] = svd(temp);


D = max(reshape(U(:,1:K),f_size*f_size,K),0);             % TO get a non-negative dictionary, we simply set the negative values as zeros
D = Normalize(D);
D   = reshape(D,f_size,f_size,K);

end

