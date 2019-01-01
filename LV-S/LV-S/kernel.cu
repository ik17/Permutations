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



void LVS(FILE *fp, int k, double *sum, double *maxi, double *mini, std::string rijec) {
	clock_t begin = clock();
	std::set<std::string> permutacije;
	permutacije.insert(rijec);
	int n = 1;
	for (int i{ 2 }; i <= rijec.size(); i++)
		n *= i;
	std::mt19937 rng;
	rng.seed(std::random_device()());
	std::uniform_int_distribution<std::mt19937::result_type> permut(0, rijec.size() - 1);

	while (permutacije.size() != n) {
		std::swap(rijec[permut(rng)], rijec[permut(rng)]);
		permutacije.insert(rijec);
	}

	for (std::string d : permutacije)
		std::cout << d << std::endl;
	time_t end = clock();
	printf("Vrijeme izvrsenja u sekundama je: %f\n", (double)(end - begin) / CLOCKS_PER_SEC);
	if (k != 0) {
		fprintf(fp, "%d,%f\n", k, (double)(end - begin) / CLOCKS_PER_SEC);
		*sum += (double)(end - begin) / CLOCKS_PER_SEC;
		if (*maxi < (double)(end - begin) / CLOCKS_PER_SEC) *maxi = (double)(end - begin) / CLOCKS_PER_SEC;
		if (*mini > (double)(end - begin) / CLOCKS_PER_SEC) *mini = (double)(end - begin) / CLOCKS_PER_SEC;
	}
}


int main()
{
	srand(time(NULL));
	FILE *fp;
	fp = fopen("C:\\Users\\ismar\\Desktop\\LVS.csv", "w");
	double sum = 0.0;
	double maxi = -999999.9;
	double mini = 999999.9;
	for (int i = 0; i <= 100; i++) {
		if (fp == NULL) {
			printf("Couldn't open file\n");
			return;
		}
		LVS(fp, i, &sum, &maxi, &mini, "ABCDEF");

	}
	fprintf(fp, "%s,%f\n", "Minimum", mini);
	fprintf(fp, "%s,%f\n", "Maximum", maxi);
	fprintf(fp, "%s,%f\n", "Prosjek", 1.0*sum / 100);
	printf("Prosjecno vrijeme izvrsavanja je: %f", 1.0*sum / 100);
	fclose(fp);
	return 0;
}
