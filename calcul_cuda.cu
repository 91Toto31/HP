#include "calcul_cuda.h"

struct GlobalVarMatrix{
	float *matriceLeftd;
	float *matriceRightd;
	float *matriceResultd;
};

__global__ void matrixMulKernel(float * matriceResultd, float * matriceLeftd, float * matriceRightd, int width){
	// identifiant de thread à deux dimensions, comme la matrice
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	// Pvaleur sert au stockage de la valeur calculée par le thread
	float pResult = 0;
	for(int i = 0; i < width; ++i){
		float mdElement = matriceLeftd[ty*width + i];
		float ndElement = matriceRightd[i*width + tx];
		pResult        += mdElement * ndElement;
	}
	// écrit la valeur calculée dans la matrice de résultat
	// chaque thread ne peut écrire qu'une valeur !
	matriceResultd[ty*width + tx] = pResult;
}

///fonction qui appelle la fonction en cuda de multiplication de matrice
extern "C" void matrixMulOnDevice(float * matriceResult, float * matriceLeft, float * matriceRight, int width){
	//calcul de la taille des matrices
	int size = width*width*sizeof(float);
	//allocation des matrices et leur remplissage
	cudaMalloc(&globalVarMatrix.matriceLeftd, size);
	cudaMemcpy(globalVarMatrix.matriceLeftd, matriceLeft, size, cudaMemcpyHostToDevice) ;
	cudaMalloc(&globalVarMatrix.matriceRightd, size);
	cudaMemcpy(globalVarMatrix.matriceRightd, matriceRight, size, cudaMemcpyHostToDevice);
	//allocation de la matrice de résultat
	cudaMalloc(&globalVarMatrix.matriceResultd, size);
	//multiplication d'une seule matrice
	dim3 dimGrid(1, 1);
	//matrice carrée
	dim3 dimBlock(width, width);
	//produit matriciel proprement dit
	matrixMulKernel<<<dimGrid, dimBlock>>>(globalVarMatrix.matriceLeftd, globalVarMatrix.matriceRightd, globalVarMatrix.matriceResultd, width);
	//récupération du résultat du calcul
	cudaMemcpy(matriceResult, globalVarMatrix.matriceResultd, size, cudaMemcpyDeviceToHost);
	//destruction des matrices, désormais inutilisées
	cudaFree(globalVarMatrix.matriceLeftd);
	cudaFree(globalVarMatrix.matriceRightd);
	cudaFree(globalVarMatrix.matriceResultd);
}

