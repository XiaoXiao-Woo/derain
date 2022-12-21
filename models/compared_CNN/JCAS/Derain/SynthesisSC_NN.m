function [ Z, T ] = SynthesisSC_NN( T, D, lambda, p, MaxIter )

CutEdge    = 6;
[M N] = size(T); 


X = padarray(T,[1 1]*CutEdge,'symmetric','both');
for a = 1:4
    X = edgetaper(X,fspecial('gaussian',CutEdge,CutEdge/6));
end
FX = fft2(X);

FilterSize = size(D,1);
K = size(D,3);

for k=1:K
    Filters(:,:,k) = psf2otf(fliplr(flipud(D(:,:,k))),[M+2*CutEdge N+2*CutEdge]);
end
[Z FT]= CSC_ADMM_CPU_NN( Filters, FX, 1000, lambda, p, 1.05, 1 );
T = real(ifft2(FT));
T = T(CutEdge+1:end-CutEdge,CutEdge+1:end-CutEdge,:);
end

