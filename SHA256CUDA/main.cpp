
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <thread>
#include <iomanip>
#include <string>
#include "main.h"
#include "sha256.cuh"

#define SHOW_INTERVAL_MS 1000
#define BLOCK_SIZE 1024
#define SHA_PER_ITERATIONS 3200

size_t difficulty = 7;

// Output string by the device read by host
char *g_out = nullptr;
unsigned char *g_hash_out = nullptr;
int *g_found = nullptr;

uint64_t nonce = 0;
uint64_t user_nonce = 0;

// First timestamp when program starts
static std::chrono::high_resolution_clock::time_point t1;

// Last timestamp we printed debug infos
static std::chrono::high_resolution_clock::time_point t_last_updated;

__device__ bool checkZeroPadding(unsigned char* sha, size_t difficulty) {

	for (size_t cur_byte = 0; cur_byte < difficulty / 2; ++cur_byte) {
		if (sha[cur_byte] != 0) {
			return false;
		}
	}

	bool isOdd = difficulty % 2 != 0;
	size_t last_byte_check = static_cast<size_t>(difficulty / 2);
	if (isOdd) {
		if (sha[last_byte_check] > 0x0F || sha[last_byte_check] == 0) {
			return false;
		}
	}
	else if (sha[last_byte_check] < 0x0F) return false;

	return true;
}

// Does the same as sprintf(char*, "%d%s", int, const char*) but a bit faster
__device__ size_t concatenate_nonce(uint64_t nonce, const char* str, size_t strlen, unsigned char* out) {
	uint64_t result = nonce;
	uint8_t remainder;
	size_t nonce_size = nonce == 0 ? 1 : floor(log10((double)nonce)) + 1;
	size_t i = nonce_size;
	while (result >= 10) {
		remainder = result % 10;
		result /= 10;
		out[--i] = remainder + '0';
	}

	out[0] = result + '0';
	i = nonce_size;

	for (size_t c = 0; c < strlen; ++c) {
		out[i++] = str[c];
	}

	out[i] = 0;
	return i;
}

__global__ void sha256_kernel(char* out_input_string_nonce, unsigned char* out_found_hash, int *out_found, const char* in_input_string, size_t in_input_string_size, size_t difficulty, uint64_t nonce_offset) {
	if (*out_found) return;
	int i = blockIdx.x * blockDim.x + threadIdx.x + nonce_offset;

	unsigned char *out = (unsigned char*)malloc(in_input_string_size + 32);
	size_t size = concatenate_nonce(i, in_input_string, in_input_string_size, out);

	unsigned char sha[32];

	SHA256_CTX ctx;
	sha256_init(&ctx);
	sha256_update(&ctx, out, size);
	sha256_final(&ctx, sha);
	//atomicAdd(g_calculated_hashes, 1);
	if (checkZeroPadding(sha, difficulty)) {
		memcpy(out_found_hash, sha, 32);
		memcpy(out_input_string_nonce, out, size);
		atomicExch(out_found, 1);
	}

	free(out);

}

void pre_sha256() {
	// compy symbols
	checkCudaErrors(cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice));
}

// Prints a 32 bytes sha256 to the hexadecimal form filled with zeroes
void print_hash(const unsigned char* sha256) {
	for (size_t i = 0; i < 32; ++i) {
		std::cout << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(sha256[i]);
	}
	std::cout << std::dec << std::endl;
}

void print_state() {
	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double, std::milli> last_show_interval = t2 - t_last_updated;
	if (last_show_interval.count() > SHOW_INTERVAL_MS) {
		t_last_updated = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> span = t2 - t1;
		float ratio = span.count() / 1000;
		std::cout << std::fixed << static_cast<int>((nonce - user_nonce) / ratio) << " hash(es)/s" << std::endl;

		std::cout << std::fixed << "Nonce : " << nonce << std::endl;
	}

	if (*g_found) {
		std::cout << g_out << std::endl;
		print_hash(g_hash_out);
	}
}

int main() {

	cudaSetDevice(0);

	t1 = std::chrono::high_resolution_clock::now();
	t_last_updated = std::chrono::high_resolution_clock::now();

	std::string in;
	
	std::cout << "Entrez un message : ";
	std::cin >> in;

#ifndef _DEBUG
	std::cout << "Nonce : ";
	std::cin >> nonce;
	user_nonce = nonce;

	std::cout << "Difficulte : ";
	std::cin >> difficulty;
	std::cout << std::endl;
#endif

	const size_t input_size = in.size();

	// Input string for the device
	char *d_in = nullptr;

	// Create the input string for the device
	cudaMalloc(&d_in, input_size);
	cudaMemcpy(d_in, in.c_str(), input_size, cudaMemcpyHostToDevice);

	cudaMallocManaged(&g_out, input_size + 32);
	cudaMallocManaged(&g_hash_out, 32);
	cudaMallocManaged(&g_found, sizeof(int));
	*g_found = 0;

	nonce += user_nonce;

	pre_sha256();

	for (;;) {
		int numBlocks = (SHA_PER_ITERATIONS + BLOCK_SIZE - 1) / BLOCK_SIZE;
		sha256_kernel << < numBlocks, BLOCK_SIZE >> > (g_out, g_hash_out, g_found, d_in, input_size, difficulty, nonce);

		cudaDeviceSynchronize();

		nonce += numBlocks * BLOCK_SIZE;

		print_state();

		if (*g_found) {
			break;
		}
	}

	cudaDeviceReset();

	system("pause");

	return 0;
}