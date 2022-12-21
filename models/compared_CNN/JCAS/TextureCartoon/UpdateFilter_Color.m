function [ D] = UpdateFilter_Color( Image, Z, filters, f_size, K, MaxIter )

% We update the dictionary for each channels ono by one
CutEdge = 6;
[K] = size(Z,4);
[M N C] = size(Image);
half_f_size = (f_size-1)/2;
X = Image(half_f_size+1:end-half_f_size,half_f_size+1:end-half_f_size,:);


Zpat = ImageCube2PatchSingle(Z(CutEdge+1:end-CutEdge,CutEdge+1:end-CutEdge,1,:),M,N,K,f_size,1);
tempfilters = filters(:,:,1,:);
f_vec = tempfilters(:)';
tempX = X(:,:,1);
X_vec = tempX(:)';

iter=1;
temp = Zpat*Zpat';
[u s v] = svd(temp);
step = max(1/(s(1,1).^2+eps),1/1000);
while(iter<MaxIter)
    Grad  = Zpat*(f_vec*Zpat - X_vec)';
    f_vec = reshape(f_vec-step*Grad',f_size*f_size,K);
    f_vec = Normalize(f_vec);
    f_vec = f_vec(:)';
    iter  = iter+1;
end

f_vec = reshape(f_vec,f_size,f_size,1,K);
D(:,:,1,:)=f_vec;


Zpat = ImageCube2PatchSingle(Z(CutEdge+1:end-CutEdge,CutEdge+1:end-CutEdge,2,:),M,N,K,f_size,1);
tempfilters = filters(:,:,2,:);
f_vec = tempfilters(:)';
tempX = X(:,:,2);
X_vec = tempX(:)';

iter=1;
temp = Zpat*Zpat';
[u s v] = svd(temp);
step = max(1/(s(1,1).^2+eps),1/1000);
while(iter<MaxIter)
    Grad  = Zpat*(f_vec*Zpat - X_vec)';
    f_vec = reshape(f_vec-step*Grad',f_size*f_size,K);
    f_vec = Normalize(f_vec);
    f_vec = f_vec(:)';
    iter  = iter+1;
end

f_vec = reshape(f_vec,f_size,f_size,1,K);
D(:,:,2,:)=f_vec;


Zpat = ImageCube2PatchSingle(Z(CutEdge+1:end-CutEdge,CutEdge+1:end-CutEdge,3,:),M,N,K,f_size,1);
tempfilters = filters(:,:,3,:);
f_vec = tempfilters(:)';
tempX = X(:,:,3);
X_vec = tempX(:)';

iter=1;
temp = Zpat*Zpat';
[u s v] = svd(temp);
step = max(1/(s(1,1).^2+eps),1/1000);
while(iter<MaxIter)
    Grad  = Zpat*(f_vec*Zpat - X_vec)';
    f_vec = reshape(f_vec-step*Grad',f_size*f_size,K);
    f_vec = Normalize(f_vec);
    f_vec = f_vec(:)';
    iter  = iter+1;
end

f_vec = reshape(f_vec,f_size,f_size,1,K);
D(:,:,3,:)=f_vec;


