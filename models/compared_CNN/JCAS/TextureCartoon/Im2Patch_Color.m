function  [Patches]  =  Im2Patch_Color( Img,PatSize )
C    = size(Img,3);
TotalPatNum = (size(Img,1)-PatSize+1)*(size(Img,2)-PatSize+1);                  %Total Patch Number in the image
Patches     =   zeros(PatSize*PatSize*C, TotalPatNum, 'single');                      %Current Patches
k           =   0;
dim = PatSize*PatSize;
for j  = 1:PatSize
    for i  = 1:PatSize
        k           =  k+1;
        for c= 1:C
            temp        =  Img(i:end-PatSize+i,j:end-PatSize+j,c);
            Patches(dim*(c-1)+k,:)=  temp(:)';
        end
    end
end
 