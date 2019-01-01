#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>

#include <set>
#include <string>
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <random>

inline void GPUassert(cudaError_t code, char * file, int line, bool Abort = true)
{
	if (code != 0) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (Abort) return;
	}
}

#define GPUerrchk(ans) { GPUassert((ans), __FILE__, __LINE__); }


__device__ int factorial(int n) {
	if (n == 1) {
		return 1;
	}
	return n * factorial(n - 1);
}



__global__ void permute_kernel_MC(char* d_A, int size, unsigned long long seed, char* svePermutacije) {
	
	int jednoD = blockIdx.x;
	int dvoD = jednoD + blockIdx.y*gridDim.x;
	int troD = dvoD + gridDim.x*gridDim.y*blockIdx.z;
	int tid;
	tid = troD * blockDim.x + threadIdx.x;
	
	int fakt = 720;

	if (tid < fakt) {
		curandState state;

		char* kopija = new char[size];

		for (int i = 0; i < size; i++) {
			kopija[i] = d_A[i];
		}

		for (int i = 0; i < size; i++) {
			curand_init(seed + tid + i, fakt, 0, &state);

			unsigned int r1 = curand_uniform(&state)*size;
			unsigned int r2 = curand_uniform(&state)*size;

			char temp = kopija[r1];
			kopija[r1] = kopija[r2];
			kopija[r2] = temp;
		}
		for (int i = 0; i < size; i++) {

			svePermutacije[size * tid + i] = kopija[i];
		}
		delete[] kopija;
	}
}

int factorialHost(int n) {
	if (n == 1) {
		return 1;
	}
	return n * factorialHost(n - 1);
}



void funkcija(FILE *fp, int n, double *sum, double *maxi, double *mini) {
	clock_t begin = clock();
	char h_a[] = "ABCDEF";

	char* d_a;
	int duzina = 6;
	cudaMalloc((void**)&d_a, sizeof(h_a));
	GPUerrchk(cudaMemcpy(d_a, h_a, sizeof(h_a), cudaMemcpyHostToDevice));

	int fakt = factorialHost(duzina);
	int threadNoMC = fakt; 
	char* h_svePermutacije = new char[threadNoMC * duzina];

	char* svePermutacije;
	cudaMalloc((void**)&svePermutacije, sizeof(char)* threadNoMC * duzina);
	cudaMemset(svePermutacije, '0', sizeof(char) * threadNoMC * duzina);



	std::set<std::string> unikatno;
	printf("\n\n LV\n");
	int number = 1;
	while (threadNoMC / number > 320) number++;
	while (1.0*threadNoMC / number - int(threadNoMC / number) > 0) number++;
	int a = threadNoMC / number;
	
	int brojac = 0;
	while (unikatno.size() != fakt) {
		permute_kernel_MC << <number, a >> > (d_a, duzina, (123456 + rand()), svePermutacije);
		GPUerrchk(cudaMemcpy(h_svePermutacije, svePermutacije, sizeof(char)* threadNoMC * duzina, cudaMemcpyDeviceToHost));
		brojac++;

		std::string temp;
		for (int i = 0; i < threadNoMC; i++) {
			temp = "";
			for (int j = 0; j < duzina; j++) {
				char c = *(h_svePermutacije + duzina * i + j);
				temp = temp + c;
			}
			unikatno.insert(temp);
		}
	}

	for (std::string s : unikatno) {
		std::cout << s << std::endl;
	}
	delete[] h_svePermutacije; 

	GPUerrchk(cudaPeekAtLastError());
	GPUerrchk(cudaDeviceSynchronize());
	time_t end = clock();
	printf("Vrijeme izvrsenja u sekundama je: %f\n", (double)(end - begin) / CLOCKS_PER_SEC);
	if (n != 0) {
		fprintf(fp, "%d,%f\n", n, (double)(end - begin) / CLOCKS_PER_SEC);
		*sum += (double)(end - begin) / CLOCKS_PER_SEC;
		if (*maxi < (double)(end - begin) / CLOCKS_PER_SEC) *maxi = (double)(end - begin) / CLOCKS_PER_SEC;
		if (*mini > (double)(end - begin) / CLOCKS_PER_SEC) *mini = (double)(end - begin) / CLOCKS_PER_SEC;
	}
}


int main()
{
	srand(time(NULL));
	FILE *fp;
	fp = fopen("C:\\Users\\ismar\\Desktop\\LVP.csv", "w");
	double sum = 0.0;
	double maxi = -999999.9;
	double mini = 999999.9;
	for (int i = 0; i <= 100; i++) {
		if (fp == NULL) {
			printf("Couldn't open file\n");
			return;
		}
		funkcija(fp, i, &sum, &maxi, &mini);
	}
	fprintf(fp, "%s,%f\n", "Minimum", mini);
	fprintf(fp, "%s,%f\n", "Maximum", maxi);
	fprintf(fp, "%s,%f\n", "Prosjek", 1.0*sum / 100);
	printf("Prosjecno vrijeme izvrsavanja je: %f", 1.0*sum / 100);
	fclose(fp);
	return 0;
}
