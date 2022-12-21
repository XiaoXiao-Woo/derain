function [ A, S] = AnalysisSC( X, P,  lambda, p, MaxIter )
%   Solve the following problem with ADMM:

%   ||X - S||_F^2+lambda sum_i ||P_i*S||_1

%   Introducing alternative variable A:
%   ||X-S||_F^2+lambda sum_i ||A_i||_1,  s.t. P_i*S = A_i.

%   The ALM form:
%   ||X-S||_F^2+lambda sum_i ||A_i||_1 + sum_ i <L, P_i*S - A_i>+ mu/2 sum_ i ||P_i*S - A_i||_F^2,  s.t. P_i*S = A_i.   


K      = size(P,2);
[m n]  = size(X);


A = zeros(m,n,K,'single');
L = A;
S = X;
FP = zeros(m,n,K,'single');
Z = A;
PS = zeros(m,n,K,'single');

iter = 1;
mu   = 1;
Maxmu = 1e6;
pho  = 1.05;
for k=1:K
    FP(:,:,k) = psf2otf(P{k},[m,n]);
end
conjFP = conj(FP);
FX   = fft2(X);
SumFPTP = sum(FP.*conjFP,3);
while(iter<MaxIter)

    Z = fft2(A+L/mu);
    S = (mu*sum(conjFP.*Z,3)+FX)./(mu*SumFPTP+1);
    PS = real(ifft2(FP.*repmat(S,[1,1,K])));
    A = PS-L/mu;
    A = sign(A).*max(abs(A)-lambda/mu,0);
    L = L+mu*(A-PS);
    if(mu<Maxmu)
        mu = mu*pho;
    end
    if(mod(iter,10)==0)
        error = sum((A(:)-PS(:)).^2);
        if(error<1e-8)
            iter=MaxIter+1;
        end
    end
    iter = iter+1;
end
S = real(ifft2(S));
