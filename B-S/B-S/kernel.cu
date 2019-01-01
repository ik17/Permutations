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

int factorialHost(int n) {
	if (n == 1) {
		return 1;
	}
	return n * factorialHost(n - 1);
}



void countNumber(int* counter, int size, int* maxNumber) {
	int i = size - 1;
	bool residue = true;
	while (residue) {
		counter[i] += 1;
		if (counter[i] == maxNumber[i] + 1) {
			counter[i] = 0;
			i -= 1;
		}
		else residue = false;
	}
}

char* createString(char* someList, int size, int* counter) {
	char* s = new char[size + 1]();
	for (int i = 0; i < size; i++) {
		if (counter[i] > 0) {
			for (int j = i; j > i - counter[i]; j--) {
				s[j] = s[j - 1];
			}
		}
		s[i - counter[i]] = someList[i];
	}
	s[size] = '\0';
	return s;
}

char** permutations(char* someList, int size) {
	int* counter = new int[size]();
	int permutationNumber = factorialHost(size);
	char** permutationList = new char*[permutationNumber]();
	for (int i = 0; i < size; i++) {
		counter[i] = 0;
	}
	permutationList[0] = createString(someList, size, counter);
	int* maxNumber = new int[size]();
	for (int i = 0; i < size; i++) {
		maxNumber[i] = i;
	}
	for (int i = 1; i < permutationNumber; i++) {
		countNumber(counter, size, maxNumber);
		permutationList[i] = createString(someList, size, counter);
	}
	delete[]counter;
	delete[]maxNumber;
	return permutationList;
}



void BS(FILE *fp, int k, double *sum, double *maxi, double *mini) {
	clock_t begin = clock();
	char letters[7] = { 'a', 'b', 'c', 'd', 'e', 'f', '\0' };
	int size = 6;
	char** perm = permutations(letters, size);
	for (int i = 0; i < factorialHost(size); i++) std::cout << perm[i] << std::endl;
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
	//proba << <3, 10 >> > ();
	srand(time(NULL));
	FILE *fp;
	fp = fopen("C:\\Users\\ismar\\Desktop\\BS.csv", "w");
	double sum = 0.0;
	double maxi = -999999.9;
	double mini = 999999.9;
	for (int i = 0; i <= 100; i++) {
		if (fp == NULL) {
			printf("Couldn't open file\n");
			return;
		}
		BS(fp, i, &sum, &maxi, &mini);
	}
	fprintf(fp, "%s,%f\n", "Minimum", mini);
	fprintf(fp, "%s,%f\n", "Maximum", maxi);
	fprintf(fp, "%s,%f\n", "Prosjek", 1.0*sum / 100);
	printf("Prosjecno vrijeme izvrsavanja je: %f", 1.0*sum / 100);
	fclose(fp);
	return 0;
}
