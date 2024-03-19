/* 
bSuma los elementos de un vectore en CPU y GPU
   Parámetros opcionales (en este orden): sumavectores #n #blk

   #n: número de elementos en cada vector
   #blk: hilos por bloque CUDA
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

const int N = 2048 ;    // Número predeterm. de elementos en los vectores

const int CUDA_BLK = 1024;  // Tamaño predeterm. de bloque de hilos ƒCUDA


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
void populating_arrays(float arrayA[], const unsigned int n)
{
  unsigned int i;

  for(i = 0; i < n; i++) {
    arrayA[i] = 1.0;
  }
}

/*
  Función para sumar dos vectores en la CPU *r* veces
*/
void rarrays_CPU(float arrayA[], const unsigned int n)
{
  unsigned int i;

    for(i = 1; i < n; i++) {
      arrayA[0] += arrayA[i] ;
    }
  
}


/*
  Definición de nuestro kernel para los elementos de un  vector en CUDA usando funciones atómicas
*/
__global__ void kernel_cuda_atomic(float *mA, const int n)
{
  int 		global_id = blockIdx.x * blockDim.x + threadIdx.x;
  float 	val;

  val = mA[global_id];
  
  if ( global_id > 0)
  	atomicAdd(&mA[0], val);
    
}


/*
  Definición de nuestro kernel para los elementos de un  vector en CUDA usando shuffles
*/
__global__ void kernel_cuda_shuffles(float *mA, const int n)
{

  // shared memory

    __shared__ float temp[32];

    int   tid = threadIdx.x;
    int   global_id = tid + blockIdx.x*blockDim.x;
    float val = mA[global_id];

    if (global_id ==0) val =0.0f;
    // first, do reduction within each warp

    for (int i=1; i<32; i=2*i)
      val += __shfl_xor_sync((unsigned int)-1, val, i);

    // put warp sums into shared memory, then read back into first warp

    if (tid%32==0) temp[tid/32] = val;

    __syncthreads();

    if (tid<32) {
      val = 0.0f;
      if (tid<blockDim.x/32) val = temp[tid];

    // second, do final reduction within first warp

      for (int i=1; i<32; i=2*i)
        val += __shfl_xor_sync((unsigned int)-1, val, i);

    // finally, first thread atomically adds result to global sum

      if (tid==0) atomicAdd(&mA[0], val);
      }
  
   }

   
/*
  Función para sumar dos vectores en la GPU *r* veces
*/
void rarrays_GPU(float arrayA[], const unsigned int n, const unsigned int blk_size, 
		    resnfo *const start, resnfo *const end)
{

  // Número de bytes de cada uno de nuestros vectores
  unsigned int 	numBytes = n * sizeof(float);

  // Reservamos memoria global del device (GPU) para nuestros 
  // arrays y los copiamos
  float *cA;
  cudaMalloc((void **) &cA, numBytes);
  cudaMemcpy(cA, arrayA, numBytes, cudaMemcpyHostToDevice); // CPU -> GPU


  // Bloque unidimensional de hilos (*blk_size* hilos)
  dim3 dimBlock(blk_size);

  // Rejilla unidimensional (*ceil(n/blk_size)* bloques)
  dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);

  // Lanzamos ejecución del kernel en la GPU *r* veces
  timestamp(start);            // Medimos tiempo de cálculo en GPU
    // kernel_cuda_atomic<<<dimGrid, dimBlock>>>(cA, n);
    kernel_cuda_shuffles<<<dimGrid, dimBlock>>>(cA, n);
   
 
  timestamp(end);

  cudaMemcpy(&arrayA[0], cA, sizeof(float), cudaMemcpyDeviceToHost); // GPU -> CPU

  cudaFree (cA);

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

  // Número de elementos en los vectores (predeterminado: N)
  unsigned int n = (argc > 1)?atoi (argv[1]):N;

   // Número de hilos en cada bloque CUDA (predeterminado: CUDA_BLK)
  unsigned int cb = (argc > 2)?atoi (argv[2]):CUDA_BLK;

  // Número de bytes a reservar para nuestros vectores
  unsigned int numBytes = n * sizeof(float);

  // Reservamos e inicializamos vectores
  timestamp(&start);
  float *vectorA = (float *) malloc(numBytes);
   
  populating_arrays(vectorA,  n);
  timestamp(&end);

  myElapsedtime(start, end, &time);
  printtime(time);
  printf(" -> Reservar e inicializar vectores (%u)\n\n", n);


  // Sumamos vectores en CPU
  timestamp(&start);
  rarrays_CPU(vectorA,  n);
  timestamp(&end);

  myElapsedtime(start, end, &time);
  printtime(time);
  printf(" -> Sumar elementos de un vector en CPU %f \n\n", vectorA[0]);

   populating_arrays(vectorA,  n);

  // Sumamos vectores en GPU
  timestamp(&start);
  rarrays_GPU(vectorA,  n, cb, &startgpu, &endgpu);
  timestamp(&end);

  printf("resultado %f\n", vectorA[0]);
 
  myElapsedtime(start, end, &time);
  printtime(time);
 
  // Separamos tiempo de cálculo en GPU de tiempo de transferencia
  myElapsedtime(startgpu, endgpu, &timegpu);
  printf("\t\tDesglose:\n\t\t");	
  printtime(timegpu);
  printf("tiempo cálculo en GPU\n\t\t%15f s alloc y comm\n", time - timegpu);

  free(vectorA);
  
  return(0);
}



