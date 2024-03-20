#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "calcul_cuda.h"

///fonction qui initialise le premier terme de la série pseudo-aléatoire
void initRandom(){
	time_t t = time(NULL);
	srand(t);
}

///fonction qui tire un nombre aléatoire entre deux bornes
/**	@param inf : borne inférieure
 * 	@param sup : borne supérieur
 * 	@return nombre aléatoire entre deux bornes
*/
float getRandFloat(float inf, float sup){
	return inf + (((float)rand())*(sup - inf))/((float)RAND_MAX);
}

///fonction qui initialise une matrice carrée avec des nombres aléatoires
/**	@param matrix : matrice que l'on veut initialiser de manière aléatoire
 * 	@param size : taille de la matrice
 * 	@param inf : borne inférieure
 * 	@param sup : borne supérieur
*/
void initRandomMatrix(float* matrix, size_t size, float inf, float sup){
	if(matrix == NULL) return;
	size_t i,j;
	for(i = 0; i < size*size; ++i){
		matrix[i] = getRandFloat(inf, sup);
	}
}

int main(int argc, char** argv){
	initRandom();
	clock_t temps = clock();
	float chrono;
	int width = 10000;                    //on défini la taille de la matrice
	int size = width*width*sizeof(float);
	//on alloue les matrices
	float * matriceLeft = malloc(size);
	float * matriceRight = malloc(size);
	float * matriceResult = malloc(size);
	//on initialise les matrices aléatoirement
	initRandomMatrix(matriceLeft, width, -10.0, 10.0);
	initRandomMatrix(matriceRight, width, -10.0, 10.0);
	temps = clock() - temps;
	chrono = ((float)temps)/((float)CLOCKS_PER_SEC);
	printf("Temps de l'initialisation : %fs\n", chrono);
	temps = clock();
	//on appelle la fonction qui fait la multiplication à notre place
	matrixMulOnDevice(matriceResult, matriceLeft, matriceRight, width);
	temps = clock() - temps;
	chrono = ((float)temps)/((float)CLOCKS_PER_SEC);
	printf("Temps du calcul de la multiplication : %fs\n", chrono);
	//on désaloue les matrices
	free(matriceResult);
	free(matriceRight);
	free(matriceLeft);
	return 0;
}


