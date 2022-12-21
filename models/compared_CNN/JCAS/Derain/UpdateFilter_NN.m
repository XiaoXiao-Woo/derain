function [ D] = UpdateFilter_Color_NN( Image, Z, filters, f_size, K, MaxIter )
% Update the filters with proximal gradient descent
CutEdge = 6;
[K]     = size(Z,3);
[M N]   = size(Image);
half_f_size = (f_size-1)/2;
X = Image(half_f_size+1:end-half_f_size,half_f_size+1:end-half_f_size,:);


Zpat = ImageCube2PatchSingle(Z(CutEdge+1:end-CutEdge,CutEdge+1:end-CutEdge,:),M,N,K,f_size,1);
tempfilters = filters(:,:,:);
f_vec = tempfilters(:)';
X_vec = X(:)';

iter=1;
step = 1/100;
while(iter<MaxIter)
    Grad  = Zpat*(f_vec*Zpat - X_vec)';
    f_vec = reshape(f_vec-step*Grad',f_size*f_size,K);
    f_vec = max(f_vec,0);
    f_vec = Normalize(f_vec);
    f_vec = f_vec(:)';
    iter = iter+1;
end

f_vec = reshape(f_vec,f_size,f_size,K);
D(:,:,:)=f_vec;


