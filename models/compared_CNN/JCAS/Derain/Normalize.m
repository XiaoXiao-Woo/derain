function [ Mat ] = Normalize( Mat )

dim = size(Mat,1);
Mat = Mat./repmat(sqrt(max(sum(Mat.^2),1)),dim,1);

end

