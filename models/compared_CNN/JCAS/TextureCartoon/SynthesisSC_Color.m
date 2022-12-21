function [ Z, T ] = SynthesisSC_Color( T, D, lambda, p, MaxIter )

CutEdge    = 6;
[M N C] = size(T); 

X = padarray(T,[1 1]*CutEdge,'symmetric','both');
for a = 1:4
   X = edgetaper(X,fspecial('gaussian',CutEdge,CutEdge/6));
end
FX = fft2(X);

FilterSize = size(D,1);
K = size(D,4);
for k=1:K
    for c = 1:C
        Filters(:,:,c,k) = psf2otf(fliplr(flipud(D(:,:,c,k))),[M+2*CutEdge N+2*CutEdge]);
    end
end
%[Z FT]= CSC_ADMM_GPU_Color( Filters, FX, 1000, lambda, p, 1.05, 0.1 );
[Z FT]= CSC_ADMM_CPU_Color( Filters, FX, 1000, lambda, p, 1.05, 0.1 );
T = real(ifft2(FT));
T = T(CutEdge+1:end-CutEdge,CutEdge+1:end-CutEdge,:,:);
T = reshape(T,M,N,C);
end

