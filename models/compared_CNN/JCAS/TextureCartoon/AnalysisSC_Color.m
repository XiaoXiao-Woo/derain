function [ A, S] = AnalysisSC_Color( X, P,  lambda, p, MaxIter )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
%   ||X - S||+lambda sum_i |M.*P_i*S|


K      = size(P,2);
[m n C]  = size(X);

% X(:,:,1) = X(:,:,1)*0.25;

A = zeros(m,n,C,K,'single');
L = A;
S = X;
FP = zeros(m,n,C,K,'single');
Z = A;
PS = zeros(m,n,C,K,'single');

iter = 1;
mu   = 1;
Maxmu = 1e6;
pho  = 1.05;
for k=1:K
    FP(:,:,:,k) = repmat(psf2otf(P{k},[m,n]),[1,1,C]);
end
conjFP = conj(FP);
FX   = fft2(X);
SumFPTP = sum(FP.*conjFP,4);

while(iter<MaxIter)

     Z = fft2(A+L/mu);
     S = (mu*sum(conjFP.*Z,4)+FX)./(mu*SumFPTP+1);
     PS = real(ifft2(FP.*repmat(S,[1,1,1,K])));
     A = PS-L/mu;
     temp = sqrt(sum((A).^2,3));
     A =  repmat(max(temp-lambda/mu,0)./(temp+eps),[1,1,3,1]).*(A);
     L = L+mu*(A-PS);
     if(mu<Maxmu)
        mu = mu*pho;
     end
     if(mod(iter,10)==0)
        error = mean((A(:)-PS(:)).^2);
        if(error<1e-9)
            iter=MaxIter+1;
        end
     end
     iter = iter+1;
end
S = real(ifft2(S));