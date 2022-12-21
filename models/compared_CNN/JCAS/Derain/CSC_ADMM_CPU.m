function [ Z  FY] = CSC_ADMM_CPU( Filters, FX, MaxIter, lambda, p, rho, mu0 )
%  Solving convolution sparse coding problem with ADMM
%  ||X-sum F_i*Z_i||_F^2 + lambda ||Z||_1
%  Here the inputings of image X and filters F are their FFT forms
%  We introducing an alternative variable S and get the ALM form
%  ||X-sum F_i*Z_i||_F^2 + lambda ||T||_1 + <Z-S,T> + mu/2 ||Z-S||_F^2  

max_mu = 1e7;
tol = 1e-10;

mu = mu0;

iter = 1;
Cond = 1;
[M N K] = size(Filters);

FR = (zeros(M,N,K));
FTX = (zeros(M,N,K));

Z = (zeros(M,N,K));
S = (zeros(M,N,K));
T = (zeros(M,N,K));

C_Filters = conj(Filters);

FTX = C_Filters.*repmat(FX,[1,1,K]);

clear FX
FTF  = sum(C_Filters.*Filters,3);
while(iter<MaxIter&Cond)


    FR = FTX+mu*fft2(S-T);
    FZ = (FR-repmat(sum(Filters.*FR,3)./(mu+FTF),[1,1,K]).*C_Filters)./mu;
    Z = real(ifft2(FZ));
    
             
    S = Z+T;
    S = sign(S).*max(abs(S)-lambda/mu,0);

    T = T + Z - S;
    
    if(mu<max_mu)
        mu = mu*rho;
    end
    if(mod(iter,10)==0)
        ConvergenceError = mean(mean(mean(mean((Z-S).^2))));
        Cond = (ConvergenceError>tol);
    end
    iter = iter+1;
   
end
Z = S;
FY = sum(fft2(Z).*Filters,3);
