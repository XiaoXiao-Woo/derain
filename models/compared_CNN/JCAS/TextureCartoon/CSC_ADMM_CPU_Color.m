function [ Z  FY] = CSC_ADMM_CPU_Color( Filters, FX, MaxIter, lambda, p, rho, mu0 )

max_mu = 1e7;
tol = 1e-8;

mu = mu0;

iter = 1;
Cond = 1;
[M N C K] = size(Filters);

FR = zeros(M,N,C,K);
FTX = zeros(M,N,C,K);

Z = zeros(M,N,C,K);
S = zeros(M,N,C,K);
T = zeros(M,N,C,K);

C_Filters = conj(Filters);

FTX = C_Filters.*repmat(FX,[1,1,1,K]);

clear FX
FTF  = sum(C_Filters.*Filters,4);
while(iter<MaxIter&Cond)


    FR = FTX+mu*fft2(S-T);
    FZ = (FR-repmat(sum(Filters.*FR,4)./(mu+FTF),[1,1,1,K]).*C_Filters)./mu;
    Z  = real(ifft2(FZ));

    temp = sqrt(sum((Z+T).^2,3));
    S =  repmat(max(temp-lambda/mu,0)./(temp+eps),[1,1,3,1]).*(Z+T);

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
Z  = S;
FY = sum(FZ.*Filters,4);
