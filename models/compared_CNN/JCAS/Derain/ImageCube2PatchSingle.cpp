/*

 *
 */
// #include "stdafx.h"
#include "mex.h"
#include <malloc.h>
#include <math.h>
#include <string.h>

///////////////////////////////////////////////////////////

//void ImageCube2Patchs(double* Image, double* PatchMat, int h, int w, int patchsize, int step)
//{
//Transform a 3D images to patches
//}


// 1 Image 2 H 3 W 4 K 5 patchsize 6 step
void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[]) {

	float *Image,*PatchMat,*CurImage;
	int h = mxGetScalar(prhs[1]);
	int w = mxGetScalar(prhs[2]);
	int k = mxGetScalar(prhs[3]);
	int patchsize = mxGetScalar(prhs[4]);
	int step = mxGetScalar(prhs[5]);

	double* temppointer = mxGetPr(prhs[0]);
    Image =  reinterpret_cast<float *>(temppointer);
    
	float temp1 = h-patchsize+1;
	temp1 = ceil(temp1/step);
	float temp2 = w-patchsize+1;
	temp2 = ceil(temp2/step);
	int patchnum = temp1*temp2;

	int patchpixelnum = patchsize*patchsize;
	int output_h = patchpixelnum*k;
    
    plhs[0] = mxCreateNumericMatrix(patchsize*patchsize*k,  patchnum, mxSINGLE_CLASS, mxREAL);
    PatchMat = (float *) mxGetData(plhs[0]);
// 	plhs[0] = mxCreateDoubleMatrix(patchsize*patchsize*k,patchnum,mxREAL);
//     double* temppointer = mxGetPr(plhs[0]);
//     PatchMat = mxGetPr(plhs[0]);
//     PatchMat =  reinterpret_cast<float *>(temppointer);
 	float* tempimage;
	float* temppatch = PatchMat;
	int index;
	for(int iter=0;iter<k;iter++)
	{
		index = 0;
		CurImage = Image + iter*h*w;
		for(int n=0;n<w-patchsize+1;)
		{
			for(int m=0;m<h-patchsize+1;)
			{
				temppatch = PatchMat+index*output_h+iter*patchpixelnum;
				index++;
				for(int i=0;i<patchsize;i++)
				{
					tempimage = CurImage + n*h+i*h+m;					
					for(int j=0;j<patchsize;j++)
					{
						*temppatch++ = *tempimage++;
					}
				} 
				m = m+step;
			}
			n = n+step;
		}
	}
}