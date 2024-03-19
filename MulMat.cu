/* 
   Programación de GPUs (General Purpose Computation on Graphics Processing Unit)

   Margarita Amor López
   
   2014 (modificado 2020)

   MulMat.c

   Multiplicacion de matrices en CPU y GPU
   Parámetros opcionales (en este orden):   #n #blk
   
      #n: número de elementos en cada fila y columna
      #blk: hilos por bloque CUDA
*/

#include <stdio.h>
#include <stdlib.h>

// Tipo de los elementos en los vectores
#ifdef _INT_
typedef int basetype;     // Tipo para elementos: int
#define labelelem    "ints"
#elif _DOUBLE_
typedef double basetype;  // Tipo para elementos: double
#define labelelem    "doubles"
#else
typedef float basetype;   // Tipo para elementos: float     PREDETERMINADO
#define labelelem    "floats"
#endif

const int N = 512;    // Número predeterm. de elementos en los vectores



const int CUDA_BLK = 128;  // Tamaño predeterm. de bloque de hilos CUDA


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
			    t.time + t.systime, t.time, t.systime);
#else
typedef struct timeval resnfo;
typedef double timenfo;
#define timestamp(sample)     gettimeofday((sample), 0)
#define printtime(t) printf("%15f s ", t);
#endif

void myElapsedtime(const resnfo start, const resnfo end, timenfo *const t)
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
  Función para inicializar los vectores que vamos a utilizar
*/
void populating_arrays(basetype arrayA[], basetype arrayB[], 
		       basetype arrayR[], const unsigned int n)
{
  unsigned int i;

  for(i = 0; i < n*n; i++) {
    arrayA[i] = 5;
    arrayB[i] = 5;
    arrayR[i] = 0;
  }
}


/*
  Función que devuelve la suma de todos los elementos de un vector, 
  y que usaremos para comprobar el resultado. 
  De paso inicializa el array.
*/
basetype checkini_array(basetype array[], const unsigned int n)
{
  unsigned int i;
  basetype res = 0;

  for(i = 0; i < n*n; i++) {
    res += array[i];
    array[i] = 5.0;
  }

  return(res);
}


/*
  Multiplicación de matrices en la CPU 
*/
void MultMat_CPU(const basetype arrayA[], const basetype arrayB[], 
		    basetype arrayR[], const unsigned int n)
{
  unsigned int i, j, k;
  basetype res;

      for(i = 0; i < n; i++) 
	for(j= 0; j<n; j++){
		res = 0;
	  for(k=0; k<n; k++)
      		res += arrayA[i*n+k] * arrayB[k*n+j];
	
	arrayR[i*n+j]= res;
	/* printf("%d %d %f \n", i, j, res); */
    }
  
}


// Declación de kernel, definición más abajo
__global__ void multmat_kernel_cuda(const basetype *const mA, 
				       const basetype *const mB, 
				       basetype *const mR, const int n);


/*
  Función para sumar dos vectores en la GPU *r* veces
*/
void multmat_GPU(const basetype arrayA[], const basetype arrayB[], 
		    basetype arrayR[], const unsigned int n, 
		    const unsigned int blk_size, 
		    resnfo *const start, resnfo *const end)
{
 
  // Número de bytes de cada uno de nuestros vectores
  unsigned int numBytes = n * n* sizeof(basetype);

  // Reservamos memoria global del device (GPU) para nuestros 
  // arrays y los copiamos
  basetype *cA;
  cudaMalloc((void **) &cA, numBytes);
  cudaMemcpy(cA, arrayA, numBytes, cudaMemcpyHostToDevice); // CPU -> GPU

  basetype *cB;
  cudaMalloc((void **) &cB, numBytes);
  cudaMemcpy(cB, arrayB, numBytes, cudaMemcpyHostToDevice); // CPU -> GPU

  basetype *cR;
  cudaMalloc((void **) &cR, numBytes);
  cudaMemset(cR, 0, numBytes); // Inicializamos (a 0) array para el resultado

  // Bloque unidimensional de hilos (*blk_size* hilos)
  dim3 dimBlock(blk_size);

  // Rejilla unidimensional (*ceil(n/blk_size)* bloques)
  dim3 dimGrid((n*n + dimBlock.x - 1) / dimBlock.x);

  // Lanzamos ejecución del kernel en la GPU *r* veces
  timestamp(start);            // Medimos tiempo de cálculo en GPU
  
    multmat_kernel_cuda<<<dimGrid, dimBlock>>>(cA, cB, cR, n);
    
   
  cudaDeviceSynchronize();
  timestamp(end);

  cudaMemcpy(arrayR, cR, numBytes, cudaMemcpyDeviceToHost); // GPU -> CPU
  

     for(int i = 0; i < n; i++) {
	int j=0;
	 printf("%f \n", arrayR[i*n+j]); }
    

  cudaFree (cA);
  cudaFree (cB);
  cudaFree (cR);
}


// Declaración de función para ver recursos del device
void devicenfo(void);


// Declaración de función para comprobar y ajustar los parámetros de
// ejecución del kernel a las restricciones de la GPU
void checkparams(unsigned int *n, unsigned int *cb);


/*
  Función principal
*/
int main(int argc, char *argv[])
{
  // Para medir tiempos
  resnfo start, end, startgpu, endgpu;
  timenfo time, timegpu;

  // Aceptamos algunos parámetros


 

  // Número de elementos en los vectores (predeterminado: N)
  unsigned int n = (argc > 1)?atoi (argv[1]):N;
  
   if (n == 0) {
    devicenfo();
    return(0);
  }

  // Número de hilos en cada bloque CUDA (predeterminado: CUDA_BLK)
  unsigned int cb = (argc > 2)?atoi (argv[2]):CUDA_BLK;

  checkparams(&n, &cb);

  // Número de bytes a reservar para nuestros vectores
  unsigned int numBytes = n * n* sizeof(basetype);

  // Reservamos e inicializamos vectores
  timestamp(&start);
  basetype *vectorA = (basetype *) malloc(numBytes);
  basetype *vectorB = (basetype *) malloc(numBytes);
  basetype *vectorR = (basetype *) malloc(numBytes);
  populating_arrays(vectorA, vectorB, vectorR, n);
  timestamp(&end);

  myElapsedtime(start, end, &time);
  printtime(time);
  printf(" -> Reservar e inicializar vectores (%u %s)\n\n", n, labelelem);


  // Multiplicamos matrices en CPU
  timestamp(&start);
  MultMat_CPU(vectorA, vectorB, vectorR, n);
  timestamp(&end);

  myElapsedtime(start, end, &time);
  printtime(time);
  printf(" -> Multiplicacion de dos matrices en la CPU(%u %s)\n\n", n*n, labelelem);

  // Sumamos elementos de vector resultante, para comprobar cálculo en GPU
  basetype result = checkini_array(vectorR, n);


  // Multiplicamos en GPU
  timestamp(&start);
  multmat_GPU(vectorA, vectorB, vectorR, n, cb, &startgpu, &endgpu);
  timestamp(&end);

  // Sumamos elementos de vector resultante, para comprobar cálculo en GPU
  basetype result_gpu = checkini_array(vectorR, n);

  myElapsedtime(start, end, &time);
  printtime(time);
  printf(" -> Multiplicacion de matrices en GPU (%d datos, %d hilos/bloq)\n", n*n, cb);
  if (result_gpu == result) // Comprobamos si resultado numérico es OK
    printf("\t\t      Resultado  OK\n\n");
  else
    printf("\t\t      mec!\n\n");

  // Separamos tiempo de cálculo en GPU de tiempo de transferencia
  myElapsedtime(startgpu, endgpu, &timegpu);
  printf("\t\tDesglose:\n\t\t");	
  printtime(timegpu);
  printf("tiempo cálculo en GPU\n\t\t%15f s alloc y comm\n", time - timegpu);

  free(vectorA);
  free(vectorB);
  free(vectorR);

  return(0);
}


/*
  Definición de nuestro kernel para multiplicar dos matrices en CUDA
*/
__global__ void multmat_kernel_cuda(const basetype *const mA, 
				       const basetype *const mB, 
				       basetype *const mR, const int n)
{
   //2D Thread ID
    int row = (blockIdx.x * blockDim.x+ threadIdx.x)/n;
    int column = (blockIdx.x * blockDim.x+ threadIdx.x) % n;

    //Pvalue stores the Pd element that is computed by the thread
    float Pvalue = 0;

    for(int k = 0; k < n; ++k) {
        float Mdelement = mA[row*n + k];
        float Ndelement = mB[k*n + column];
        Pvalue += (Mdelement*Ndelement);
    }

    mR[row*n + column] = Pvalue;


}


/*
  Sacar por pantalla información del *device*
*/
void devicenfo(void)
{
  struct cudaDeviceProp capabilities;

  cudaGetDeviceProperties (&capabilities, 0);

  printf("->CUDA Platform & Capabilities\n");
  printf("Name: %s\n", capabilities.name);
  printf("totalGlobalMem: %.2f MB\n", capabilities.totalGlobalMem/1024.0f/1024.0f);
  printf("sharedMemPerBlock: %.2f KB\n", capabilities.sharedMemPerBlock/1024.0f);
  printf("regsPerBlock (32 bits): %d\n", capabilities.regsPerBlock);
  printf("warpSize: %d\n", capabilities.warpSize);
  printf("memPitch: %.2f KB\n", capabilities.memPitch/1024.0f);
  printf("maxThreadsPerBlock: %d\n", capabilities.maxThreadsPerBlock);
  printf("maxThreadsDim: %d x %d x %d\n", capabilities.maxThreadsDim[0], 
	 capabilities.maxThreadsDim[1], capabilities.maxThreadsDim[2]);
  printf("maxGridSize: %d x %d\n", capabilities.maxGridSize[0], 
	 capabilities.maxGridSize[1]);
  printf("totalConstMem: %.2f KB\n", capabilities.totalConstMem/1024.0f);
  printf("major.minor: %d.%d\n", capabilities.major, capabilities.minor);
  printf("clockRate: %.2f MHz\n", capabilities.clockRate/1024.0f);
  printf("textureAlignment: %lu\n", capabilities.textureAlignment);
  printf("deviceOverlap: %d\n", capabilities.deviceOverlap);
  printf("multiProcessorCount: %d\n", capabilities.multiProcessorCount);
}


/*
  Función que ajusta el número de hilos, de bloques, y de bloques por hilo 
  de acuerdo a las restricciones de la GPU
*/
void checkparams(unsigned int *n, unsigned int *cb)
{
  struct cudaDeviceProp capabilities;

  // Si menos numero total de hilos que tamaño bloque, reducimos bloque
  if (*cb > *n)
    *cb = *n;

  cudaGetDeviceProperties (&capabilities, 0);

  if (*cb > capabilities.maxThreadsDim[0]) {
    *cb = capabilities.maxThreadsDim[0];
    printf("->Núm. hilos/bloq cambiado a %d (máx por bloque para dev)\n\n", 
	   *cb);
  }

  if (((*n + *cb - 1) / *cb) > capabilities.maxGridSize[0]) {
    *cb = 2 * (*n - 1) / (capabilities.maxGridSize[0] - 1);
    if (*cb > capabilities.maxThreadsDim[0]) {
      *cb = capabilities.maxThreadsDim[0];
      printf("->Núm. hilos/bloq cambiado a %d (máx por bloque para dev)\n", 
	     *cb);
      if (*n > (capabilities.maxGridSize[0] * *cb)) {
	*n = capabilities.maxGridSize[0] * *cb;
	printf("->Núm. total de hilos cambiado a %d (máx por grid para \
dev)\n\n", *n);
      } else {
	printf("\n");
      }
    } else {
      printf("->Núm. hilos/bloq cambiado a %d (%d máx. bloq/grid para \
dev)\n\n", 
	     *cb, capabilities.maxGridSize[0]);
    }
  }
}
