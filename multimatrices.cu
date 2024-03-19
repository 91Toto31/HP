/* 
  PAH (Programación de Arquitecturas Heterogéneas)
  
    multimatrices.c

 Multiply two square matrices on CPU and GPU
   Optional parameters (in this order): multimat #n #blk
   #n: number of elements in each vector
   #blk: threads per CUDA block
   
 */

#include <stdio.h>
#include <stdlib.h>



const int N = 1024;       //  Number of rows and columns

const int CUDA_BLK = 16;  // Size block CUDA_BLK * CUDA_BLK 


/* 
   Para medir el tiempo transcurrido (elapsed time):

   resnfo: tipo de dato definido para abstraer la métrica de recursos a usar
   timenfo: tipo de dato definido para abstraer la métrica de tiempo a usar

   timestamp: abstrae función usada para tomar las muestras del tiempo transcurrido

   printtime: abstrae función usada para imprimir el tiempo transcurrido

   void myElapsedtime(resnfo start, resnfo end, timenfo *t): función para obtener 
   el tiempo transcurrido entre dos medidas
*/

#include <sys/time.h>
#include <sys/resource.h>

#ifdef _noWALL_
typedef struct rusage resnfo;
typedef struct _timenfo {
  double time;
  double systime;
} timenfo;
#define timestamp(sample) getrusage(RUSAGE_SELF, (sample))
#define printtime(t) printf("%15f s (%f user + %f sys) ",		\
			    t.time + t.systime, t.time, t.systime)
#else
typedef struct timeval resnfo;
typedef double timenfo;
#define timestamp(sample)     gettimeofday((sample), 0)
#define printtime(t) printf("%15f s ", t)
#endif

void myElapsedtime(resnfo start, resnfo end, timenfo *t)
{
#ifdef _noWALL_
  t->time = (end.ru_utime.tv_sec + (end.ru_utime.tv_usec * 1E-6)) 
    - (start.ru_utime.tv_sec + (start.ru_utime.tv_usec * 1E-6));
  t->systime = (end.ru_stime.tv_sec + (end.ru_stime.tv_usec * 1E-6)) 
    - (start.ru_stime.tv_sec + (start.ru_stime.tv_usec * 1E-6));
#else
  *t = (end.tv_sec + (end.tv_usec * 1E-6)) 
    - (start.tv_sec + (start.tv_usec * 1E-6));
#endif /*_noWALL_*/
}


/*
  Función para inicializar las matrices que vamos a utilizar
*/
void populating_matrices(float matrixA[], float matrixB[], 
			 float matrixR[], const unsigned int n)
{
  unsigned int idx = 0;

  for(idx = 0; idx < n*n; idx++) {
    matrixR[idx] = 0;
    matrixA[idx] = 1.0;
    matrixB[idx] = 0.5;
  }
}


/*
  Función que devuelve la suma de todos los elementos de una matriz, 
  y que usaremos para comprobar el resultado. 
  De paso inicializa la matriz con ceros.
*/
float checkini_matrix(float matrix[], const unsigned int n)
{
  unsigned int idx = 0;
  float result = 0;

  for(idx = 0; idx < n*n; idx++) {
    result += matrix[idx];
    matrix[idx] = (float) 0;
  }

  return(result);
}


/*
  Función para multiplicar dos matrices en la CPU  veces
  Versión NAIVE
*/
void matrices_product(const float mA[], const float mB[], 
		      float mR[], const unsigned int n)
{
  unsigned int i, j, k;
  float acum;

    for(i = 0; i < n; i++) {
      for(j = 0; j < n; j++) {
	acum = mR[i*n+j];

	for (k = 0; k < n; k++)
          acum += mA[i*n+k] * mB[k*n+j];

	mR[i*n+j] = acum;
      }
    }

  
}


// Declación de kernels, definición más abajo
__global__ void mulmatrices_cuda(const float *const mA, 
				 const float *const mB, 
				 float *const mR, const int n);
__global__ void mulmatrices_cuda_blk(const float *const mA, 
				     const float *const mB, 
				     float *const mR, const int n);


/*
  Función para multiplicar dos matrices en la GPU *r* veces
*/
cudaError_t matrices_product_GPU(const int which, float mA[], float mB[], 
				 float mR[], const unsigned int n, 
				 const unsigned int blk_size, 
				 resnfo *start, resnfo *end)
{
  cudaError_t success;
  
  // Número de bytes de cada una de nuestras matrices
  unsigned int numBytes = n * n * sizeof(float);

  // Reservamos memoria global del device (GPU) para nuestras 
  // matrices y las copiamos
  float *cA;
  cudaMalloc((void **) &cA, numBytes);
  cudaMemcpy(cA, mA, numBytes, cudaMemcpyHostToDevice); // CPU -> GPU

  float *cB;
  cudaMalloc((void **) &cB, numBytes);
  cudaMemcpy(cB, mB, numBytes, cudaMemcpyHostToDevice); // CPU -> GPU

  float *cR;
  cudaMalloc((void **) &cR, numBytes);
  cudaMemset(cR, 0, numBytes); // Inicializamos (a 0) matriz para el resultado

  // Bloque bidimensional de hilos (*blk_sizexblk_size* hilos)
  dim3 dimBlock(blk_size, blk_size);

  // Rejilla bidimensional (*ceil(n/blk_size)* bloques x 2)
  dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x, 
	       (n + dimBlock.y - 1) / dimBlock.y);

  printf("Grid %d %d Threads per block %d \n", dimGrid.x, dimGrid.y, blk_size * blk_size);
  // Lanzamos ejecución del kernel en la GPU *r* veces
  // Varios kernels disponibles para el cálculo
  timestamp(start);            // Medimos tiempo de cálculo en GPU
  switch(which) {
    // naive
  case 0:
         mulmatrices_cuda<<<dimGrid, dimBlock>>>(cA, cB, cR, n);
    break;

    // blk usando *shared memory* dinámicamente 
    //y fusionando accesos (coalescing)
  case 1:
         mulmatrices_cuda_blk<<<dimGrid, dimBlock, 2 * dimBlock.x * dimBlock.y * sizeof(float)>>>(cA, cB, cR, n);
    break;
	
  }
  
  success = cudaDeviceSynchronize( );
  timestamp(end);

  cudaMemcpy(mR, cR, numBytes, cudaMemcpyDeviceToHost); // GPU -> CPU

  cudaFree (cA);
  cudaFree (cB);
  cudaFree (cR);

  return( success );
}




/*
  Función principal
*/
int main(int argc, char *argv[])
{
  // Para medir tiempos
  resnfo start, end, startgpu, endgpu;
  timenfo time, timegpu;

  // Aceptamos algunos parámetros
  
  // Número de elem/fila de las matrices cuadradas (predeterminado: N)
  unsigned int n = (argc > 1)?atoi (argv[1]):N;

  // Número de hilos en cada bloque CUDA (predeterminado: CUDA_BLK)
  unsigned int cb = (argc > 2)?atoi (argv[2]):CUDA_BLK;

  // Número de bytes a reservar para nuestras matrices
  unsigned int numBytes = n * n * sizeof(float);

  // Reservamos e inicializamos matrices
  timestamp(&start);
  float *matrixA = (float *) malloc(numBytes);
  float *matrixB = (float *) malloc(numBytes);
  float *matrixR = (float *) malloc(numBytes);
  populating_matrices(matrixA, matrixB, matrixR, n);
  timestamp(&end);

  myElapsedtime(start, end, &time);
  printtime(time);
  printf(" -> Reservar e inicializar matrices (%ux%u)\n\n", n, n);


  // Multiplicamos matrices en CPU con versión NAIVE
  timestamp(&start);
  matrices_product(matrixA, matrixB, matrixR, n);
  timestamp(&end);

  // Sumamos elementos de matriz resultante, para comprobar cálculos posteriores
  float result = checkini_matrix(matrixR, n);

  myElapsedtime(start, end, &time);
  printtime(time);
  printf(" -> Mult. matrices en CPU NAIVE \n");


  // Multiplicamos matrices en GPU con versión NAIVE
  timestamp(&start);
  cudaError_t success = matrices_product_GPU(0, matrixA, matrixB, matrixR,   n, cb, &startgpu, &endgpu);
  timestamp(&end);

  // Sumamos elementos de matriz resultante, para comprobar cálculo en GPU
  float result2 = checkini_matrix(matrixR, n);

  myElapsedtime(start, end, &time);
  printtime(time);
  printf(" -> Mult. matrices GPU NAIVE (%d hilos, %dx%d hilos/bloq )\n", n*n, cb, cb);
  printf("\n");
  if (success == cudaSuccess) // Algún problema en la GPU?
    if (result2 == result)    // Comprobamos si resultado numérico es OK
      printf("\t\t      OK!\n\n");
    else
      printf("\t\t      mec\n\n");
  else
    printf("\t\t      CUDA ERROR: %d!\n\n", success);

  // Separamos tiempo de cálculo en GPU de tiempo de transferencia
  myElapsedtime(startgpu, endgpu, &timegpu);
  printf("\t\t");	
  printtime(timegpu);
  printf("tiempo cálculo en GPU\n\t\t%15f s alloc y comm\n", time - timegpu);


  // Multiplicamos matrices en GPU con versión BLK
  timestamp(&start);
  success = matrices_product_GPU(1, matrixA, matrixB, matrixR, n, cb, &startgpu, &endgpu);
  timestamp(&end);

  // Sumamos elementos de matriz resultante, para comprobar cálculo en GPU
  result2 = checkini_matrix(matrixR, n);

  myElapsedtime(start, end, &time);
  printtime(time);
  printf(" -> Mult. matrices GPU BLK (%d hilos, %dx%d hilos/bloq \n", n*n, cb, cb);
  printf("\t\t      ");  printf("\n");
  if (success == cudaSuccess) // Algún problema en la GPU?
    if (result2 == result) // Comprobamos si resultado numérico es OK
      printf("\t\t      OK!\n\n");
    else
      printf("\t\t      mec\n\n");
  else
    printf("\t\t      CUDA ERROR: %d!\n\n", success);

  // Separamos tiempo de cálculo en GPU de tiempo de transferencia
  myElapsedtime(startgpu, endgpu, &timegpu);
  printf("\t\t");	
  printtime(timegpu);
  printf("tiempo cálculo en GPU\n\t\t%15f s alloc y comm\n", time - timegpu);

  free(matrixA);
  free(matrixB);
  free(matrixR);

  return(0);
}


/*
  Definición de nuestro kernel NAIVE para multiplicar dos matrices
*/
__global__ void mulmatrices_cuda(const float *const mA, 
				 const float *const mB, 
				 float *const mR, const int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int row = i * n;
  //  int col = j;    Usamos j directamente
  int id = row + j;
  int count = 0;
  float acum =0;

  if (i < n && j < n) {
    acum = mR[id];
    while (count < n) {
      acum += mA[row] * mB[j];
      row++;
      j += n;
      count++;
    }
    mR[id] = acum;
  }
}


/*
  Kernel para multiplicar dos matrices usando *shared memory*
*/
extern __shared__ char array[]; // Para uso dinámico de shared mem
__global__ void mulmatrices_cuda_blk(const float *const mA, 
				     const float *const mB, 
				     float *const mR, const int n)
{
  // Block size in number of rows (= number of columns)
  int blockSize = blockDim.y;

  // Índice del bloque
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Índice del hilo
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Index of first sub-array of A to be processed by block of threads
  int aBegin = by * blockSize * n;

  // Index of last sub-array of A to be processed by block of threads
  int aEnd = aBegin + n - 1;

  // Stride to iterate through the sub-arrays of A
  int aStep = blockSize;

  //Index of first sub-array of B to be processed by block of threads
  int bBegin = bx * blockSize;

  // Stride to iterate through the sub-arrays of B
  int bStep = blockSize * n;

   // Read sub-array of global memory block
  // Each thread reads one element
  int c = by * n * blockSize + bx * blockSize;
  float Csub = 0;
  
  // Loop through all the sub-matrices of A and B needed for the computation
  // of the sub-array assigned to the block of threads
  for (int a = aBegin, b = bBegin;
       a <= aEnd;
       a += aStep, b += bStep) {

    //Shared memory for sub-matrix of A
    float *As = (float *)array;
    // Statically it would be:    
    // __shared__ float As[blockSize][blockSize];

    // Shared memory for sub-matrix of  B
    float *Bs = (float *)&As[blockSize * blockSize];
    // Statically it would be:    
    // __shared__ float Bs[blockSize][blockSize];

    // Load arrays into shared memory from global memory
    // each thread loads a different element of the sub-array
    As[ty*blockSize + tx] = mA[a + n*ty + tx];
    Bs[ty*blockSize + tx] = mB[b + n*ty + tx];
    // Statically it would be:    
    // As[ty][tx] = mA[a + n*ty + tx];
    // Bs[ty][tx] = mB[b + n*ty + tx];

    // We synchronize threads to ensure the loading of the entire sub-array
    __syncthreads( );

    // We multiply the two sub-matrices ;
    // each thread calculates an element of the sub-array of the block
    for (int k = 0; k < blockSize; ++k)
      Csub += As[ty*blockSize + k] * Bs[k*blockSize + tx];
    // Statically it would be:     Csub += As[ty][k] * Bs[k][tx];

    // We synchronize to ensure complete calculation of the sub-matrix
    // before loading new sub-arrays of A and B in next iteration
    __syncthreads();
  }


// We write the sub-array of the block in global memory
   // each thread writes an element
  mR[c + ty*n + tx] = Csub;
}


