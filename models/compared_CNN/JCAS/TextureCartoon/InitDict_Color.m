function [ D ] = InitDict_Color( Image, f_size, K )

[m n] = size(Image);

Patches = Im2Patch_Color( Image,f_size );
Patches = Patches- repmat(mean(Patches,2),1,size(Patches,2));
temp = Patches*Patches';
[U S V] = svd(temp);


D = reshape(U(:,1:K),f_size*f_size*3,K);
D = Normalize(D);
D = reshape(D,f_size,f_size,3,K);

end

