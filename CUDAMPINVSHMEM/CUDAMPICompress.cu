/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//Sriram Madhivanan
//CUDA-MPI Implementation
/*---------------------------------------------------------------------------------------------------------------------------------------------*/

//#include "device/nvshmem_coll_defines.cuh"
#include "device_host/nvshmem_types.h"
#include "host/nvshmem_api.h"
#include "host/nvshmem_coll_api.h"
#include "mpi.h"
#include "nvshmem.h"
#include "nvshmemx.h"
#include <cstdint>
#include <cstdio>
#include <ctime>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include <limits.h>
#include<time.h>
#include<math.h>
#include "../include/parallelHeader.h"

#define block_size 1024
#define MIN_SCRATCH_SIZE 50 * 1024 * 1024
#define CUDA_CHECK(stmt)                                  \
do {                                                      \
    cudaError_t result = (stmt);                          \
    if (cudaSuccess != result) {                          \
        fprintf(stderr, "[%s:%d] CUDA failed with %s \n", \
         __FILE__, __LINE__, cudaGetErrorString(result)); \
        exit(-1);                                         \
    }                                                     \
} while (0)



struct huffmanTree *head_huffmanTreeNode;
struct huffmanTree huffmanTreeNode[512];
unsigned char bitSequenceConstMemory[256][255];
struct huffmanDictionary huffmanDictionary;
unsigned int constMemoryFlag = 0;

int main(int argc, char* argv[]){
	double start, end, compressStart, compressEnd;
	double beforeFcollect, afterFcollect;
	double beforeBcast, afterBcast;
	int rank, numProcesses;
	unsigned int cpu_time_used;
	unsigned int i, blockLength;
	unsigned int *compressedDataOffset, *compBlockLengthArray;
	unsigned int distinctCharacterCount, combinedHuffmanNodes, inputFileLength, compBlockLength, frequency[256];
	unsigned char *inputFileData, bitSequence[255], bitSequenceLength = 0;
	FILE *inputFile;
	unsigned int integerOverflowFlag;
	long unsigned int mem_free, mem_total;
	long unsigned int mem_req, mem_offset, mem_data;
	int numKernelRuns;
	
	MPI_Init( &argc, &argv);
	MPI_File mpi_inputFile, mpi_compressedFile;
	MPI_Status status;
	
	// get rank and number of processes value
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
	// init nvshmem
	MPI_Comm mpi_comm = MPI_COMM_WORLD;
	nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
	int mype;
	attr.mpi_comm = &mpi_comm;
	nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    mype = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
	int npes  = numProcesses;
    CUDA_CHECK(cudaSetDevice(mype));
	/*----------------------------------*/
	// get file size
	if(rank == 0){
		inputFile = fopen(argv[1], "rb");
		fseek(inputFile, 0, SEEK_END);
		inputFileLength = ftell(inputFile);
		fseek(inputFile, 0, SEEK_SET);
		fclose(inputFile);
	}

	//broadcast size of file to all the processes 
	//MPI_Bcast(&inputFileLength, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

	/*use nvshmem to broadcast the file size*/
	int64_t *buffer = NULL;
    int64_t *h_buffer = NULL;
    int64_t *d_source, *d_dest;
    int64_t *h_source, *h_dest;
	int PE_root = 0;
	size_t num_elems = 1;
	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	size_t alloc_size = num_elems * 2 * sizeof(int64_t);

    CUDA_CHECK(cudaHostAlloc(&h_buffer, alloc_size, cudaHostAllocDefault));
    h_source = (int64_t *)h_buffer;
    h_dest = (int64_t *)&h_source[num_elems];
	// 关键修复点 1: 确保所有进程都初始化h_source和h_dest
	// 关键修复点 1: 确保所有进程都初始化h_source和h_dest
	if (rank == 0) {
		*h_source = inputFileLength;  // 只有root进程设置实际的值
	} 
    buffer = (int64_t *)nvshmem_malloc(alloc_size);
	d_source = (int64_t *)buffer;
    d_dest = (int64_t *)&d_source[num_elems];
	// 关键修复点 2: 明确的同步障碍，确保所有进程都就绪
	// MPI_Barrier(MPI_COMM_WORLD);
    // *h_buffer = inputFileLength;
    CUDA_CHECK(cudaMemcpyAsync(d_source, h_source, 1*sizeof(int64_t), cudaMemcpyHostToDevice, stream));                      
    CUDA_CHECK(cudaMemcpyAsync(d_dest, h_dest, 1*sizeof(int64_t),cudaMemcpyHostToDevice, stream));
	// 关键修复点 3: 确保所有进程都在同一时间点进行广播
	CUDA_CHECK(cudaStreamSynchronize(stream));  // 这里需要同步，确保复制完成
	// 广播操作
	//nvshmem_barrier_all();  // 关键修复点 4: 确保所有进程都准备好执行广播
    nvshmem_int64_broadcast(NVSHMEM_TEAM_WORLD, d_dest, d_source, num_elems, PE_root);
    CUDA_CHECK(cudaMemcpyAsync(h_source, d_source, 1*sizeof(int64_t),cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_dest,d_dest, 1*sizeof(int64_t),cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    nvshmem_barrier_all();  
	//printf("h_dest = %ld\n", *h_dest);
	
	inputFileLength = *h_dest;
	
	//printf("inputFileLength = %d\n", inputFileLength);
	/*-----------------------*/
	// get file chunk siz

	blockLength = inputFileLength / numProcesses;

	if(rank == (numProcesses-1)){
		blockLength = inputFileLength - ((numProcesses-1) * blockLength);	
	}
	
	// open file in each process and read data and allocate memory for compressed data
	MPI_File_open(MPI_COMM_WORLD, argv[1], MPI_MODE_RDONLY, MPI_INFO_NULL, &mpi_inputFile);
	MPI_File_seek(mpi_inputFile, rank * blockLength, MPI_SEEK_SET);

	inputFileData = (unsigned char *)malloc(blockLength * sizeof(unsigned char));	
	MPI_File_read(mpi_inputFile, inputFileData, blockLength, MPI_UNSIGNED_CHAR, &status);

	// start clock
	if(rank == 0){
		start = MPI_Wtime();
	}
	
	// find the frequency of each symbols
	for (i = 0; i < 256; i++){
		frequency[i] = 0;
	}
	for (i = 0; i < blockLength; i++){
		frequency[inputFileData[i]]++;
	}
	
	compBlockLengthArray = (unsigned int *)malloc(numProcesses * sizeof(unsigned int));
	
	// initialize nodes of huffman tree
	distinctCharacterCount = 0;
	for (i = 0; i < 256; i++){
		if (frequency[i] > 0){
			huffmanTreeNode[distinctCharacterCount].count = frequency[i];
			huffmanTreeNode[distinctCharacterCount].letter = i;
			huffmanTreeNode[distinctCharacterCount].left = NULL;
			huffmanTreeNode[distinctCharacterCount].right = NULL;
			distinctCharacterCount++;
		}
	}

	// build tree 
	for (i = 0; i < distinctCharacterCount - 1; i++){
		combinedHuffmanNodes = 2 * i;
		sortHuffmanTree(i, distinctCharacterCount, combinedHuffmanNodes);
		buildHuffmanTree(i, distinctCharacterCount, combinedHuffmanNodes);
	}

	if(distinctCharacterCount == 1){
  	  head_huffmanTreeNode = &huffmanTreeNode[0];
	}
	
	// build table having the bitSequence sequence and its length
	buildHuffmanDictionary(head_huffmanTreeNode, bitSequence, bitSequenceLength);

	// calculate memory requirements
	// GPU memory
	cudaMemGetInfo(&mem_free, &mem_total);
	
	// offset array requirements
	mem_offset = 0;
	for(i = 0; i < 256; i++){
		mem_offset += frequency[i] * huffmanDictionary.bitSequenceLength[i];
	}
	mem_offset = mem_offset % 8 == 0 ? mem_offset : mem_offset + 8 - mem_offset % 8;
	
	// other memory requirements
	mem_data = blockLength + (blockLength + 1) * sizeof(unsigned int) + sizeof(huffmanDictionary);
	
	if(mem_free - mem_data < MIN_SCRATCH_SIZE){
		printf("\nExiting : Not enough memory on GPU\nmem_free = %lu\nmin_mem_req = %lu\n", mem_free, mem_data + MIN_SCRATCH_SIZE);
		return -1;
	}
	mem_req = mem_free - mem_data - 10 * 1024 * 1024;
	numKernelRuns = ceil((double)mem_offset / mem_req);
	integerOverflowFlag = mem_req + 255 <= UINT_MAX || mem_offset + 255 <= UINT_MAX ? 0 : 1;
	
	// generate data offset array
	compressedDataOffset = (unsigned int *)malloc((blockLength + 1) * sizeof(unsigned int));
	if(rank==0){
		compressStart = MPI_Wtime();
	}
	// launch kernel
	lauchCUDAHuffmanCompress(inputFileData, compressedDataOffset, 
			blockLength, numKernelRuns, integerOverflowFlag, mem_req);
	
	if(rank==0){
		compressEnd = MPI_Wtime();
	}	
	// calculate length of compressed data
	compBlockLengthArray = (unsigned int *)malloc(numProcesses * sizeof(unsigned int));
	compBlockLength = mem_offset / 8 + 1024;
	compBlockLengthArray[rank] = compBlockLength;
	// send the length of each compressed chunk to process 0
	//MPI_Gather(&compBlockLength, 1, MPI_UNSIGNED, compBlockLengthArray, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
	/* use nvshmem to gather*/
	PE_root = 0;
	num_elems = 1;
	CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	alloc_size = num_elems * (1+numProcesses) * sizeof(int64_t);

    CUDA_CHECK(cudaHostAlloc(&h_buffer, alloc_size, cudaHostAllocDefault));
    h_source = (int64_t *)h_buffer;
    h_dest = (int64_t *)&h_source[num_elems];
	//h_dest = (int64_t *)compBlockLengthArray;
    buffer = (int64_t *)nvshmem_malloc(alloc_size);
	*h_source = compBlockLength;
	d_source = (int64_t *)buffer;
    d_dest = (int64_t *)&d_source[num_elems];
	// 关键修复点 2: 明确的同步障碍，确保所有进程都就绪
	// MPI_Barrier(MPI_COMM_WORLD);
    // *h_buffer = inputFileLength;
    CUDA_CHECK(cudaMemcpyAsync(d_source, h_source, num_elems*sizeof(int64_t), cudaMemcpyHostToDevice, stream));                      
    CUDA_CHECK(cudaMemcpyAsync(d_dest, h_dest, num_elems*numProcesses*sizeof(int64_t),cudaMemcpyHostToDevice, stream));
	// 关键修复点 3: 确保所有进程都在同一时间点进行广播
	CUDA_CHECK(cudaStreamSynchronize(stream));  // 这里需要同步，确保复制完成
	// 广播操作
	//nvshmem_barrier_all();  // 关键修复点 4: 确保所有进程都准备好执行广播
	if(rank==0){
		beforeFcollect = MPI_Wtime();
	}
	for(int i=0;i<1;i++){
		nvshmem_fcollectmem(NVSHMEM_TEAM_WORLD, d_dest, d_source, num_elems*sizeof(int64_t));
	}
	if(rank==0){
		afterFcollect = MPI_Wtime();
		//printf("beforeFcollect = %f, afterFcollect = %f\n", beforeFcollect, afterFcollect);
		//printf("Time taken for fcollect: %f s\n", afterFcollect - beforeFcollect);
	}
    CUDA_CHECK(cudaMemcpyAsync(h_source, d_source, num_elems*sizeof(int64_t),cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_dest,d_dest, num_elems*numProcesses*sizeof(int64_t),cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    nvshmem_barrier_all();  
	for(int i=0;i<numProcesses;i++){
		compBlockLengthArray[i] = h_dest[i];
		//printf("compBlockLengthArray[%d] = %d\n", i, compBlockLengthArray[i]);
	}
	/*-----------------------*/
	
	// update the data to reflect offsets
	if(rank == 0){
		compBlockLengthArray[0] = (numProcesses + 2) * 4 + compBlockLengthArray[0];
		for(i = 1; i < numProcesses; i++)
			compBlockLengthArray[i] = compBlockLengthArray[i] + compBlockLengthArray[i-1];
		for(i = (numProcesses - 1); i > 0; i--)
			compBlockLengthArray[i] = compBlockLengthArray[i - 1];
		compBlockLengthArray[0] = (numProcesses + 2) * 4;
	}
	// broadcast size of each compressed chunk back to all the processes
	// MPI_Bcast(compBlockLengthArray, numProcesses, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
	/* use nvshmem to gather*/
	h_buffer = (int64_t*)compBlockLengthArray;
	PE_root = 0;
	num_elems = numProcesses;
	CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	alloc_size = num_elems * 2 * sizeof(int64_t);

	CUDA_CHECK(cudaHostAlloc(&h_buffer, alloc_size, cudaHostAllocDefault));
	h_source = (int64_t *)h_buffer;
	h_dest = (int64_t *)&h_source[num_elems];

	// 关键修复点 1: 确保所有进程都初始化h_source和h_dest
	if (rank == 0) {
		for(int i=0;i<numProcesses;i++){
			h_source[i] = compBlockLengthArray[i];  // 只有root进程设置实际的值
		}	
	} 
	buffer = (int64_t *)nvshmem_malloc(alloc_size);
	d_source = (int64_t *)buffer;
	d_dest = (int64_t *)&d_source[num_elems];
	// 关键修复点 2: 明确的同步障碍，确保所有进程都就绪
	// MPI_Barrier(MPI_COMM_WORLD);
	// *h_buffer = inputFileLength;
	CUDA_CHECK(cudaMemcpyAsync(d_source, h_source, num_elems*sizeof(int64_t), cudaMemcpyHostToDevice, stream));                      
	CUDA_CHECK(cudaMemcpyAsync(d_dest, h_dest, num_elems*sizeof(int64_t),cudaMemcpyHostToDevice, stream));
	// 关键修复点 3: 确保所有进程都在同一时间点进行广播
	CUDA_CHECK(cudaStreamSynchronize(stream));  // 这里需要同步，确保复制完成
	// 广播操作
	//nvshmem_barrier_all();  // 关键修复点 4: 确保所有进程都准备好执行广播
	if(rank==0){
		beforeBcast = MPI_Wtime();
	}
	nvshmem_int64_broadcast(NVSHMEM_TEAM_WORLD, d_dest, d_source, num_elems, PE_root);
	if(rank==0){
		afterBcast = MPI_Wtime();
	}
	CUDA_CHECK(cudaMemcpyAsync(h_source, d_source, num_elems*sizeof(int64_t),cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(h_dest,d_dest, num_elems*sizeof(int64_t),cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaStreamSynchronize(stream));
	nvshmem_barrier_all();  
	for(int i=0;i<num_elems;i++){
		compBlockLengthArray[i] = h_dest[i];
		//printf("compBlockLengthArray[%d] = %d\n", i, compBlockLengthArray[i]);
	}
	/*-----------------------*/
	
	// get time
	if(rank == 0){
		end = MPI_Wtime();
		// cpu_time_used = ((end - start)) * 1000 / CLOCKS_PER_SEC;
		// int toCompressStart = (compressStart - start) * 1000 / CLOCKS_PER_SEC;
		// int toCompressEnd = (compressEnd - compressStart) * 1000 / CLOCKS_PER_SEC;
		printf("Time taken: %f s\n", end-start);
		printf("Time taken to compressStart: %f s\n", compressStart-start);
		printf("Time taken to compressEnd: %f s\n", compressEnd-start);
		printf("Time taken for fcollectStart: %f s\n", beforeFcollect - start);
		printf("Time taken for fcollectEnd: %f s\n", afterFcollect-start);
		printf("Time taken for bcastStart: %f s\n", beforeBcast - start);
		printf("Time taken for bcastEnd: %f s\n", afterBcast-start);
	}
	
	// MPI file I/O: write
	MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &mpi_compressedFile);
	if(rank == 0){
		MPI_File_write(mpi_compressedFile, &inputFileLength, 1, MPI_UNSIGNED, MPI_STATUS_IGNORE);
		MPI_File_write(mpi_compressedFile, &numProcesses, 1, MPI_UNSIGNED, MPI_STATUS_IGNORE);
		MPI_File_write(mpi_compressedFile, compBlockLengthArray, numProcesses, MPI_UNSIGNED, MPI_STATUS_IGNORE);
	}
	MPI_File_seek(mpi_compressedFile, compBlockLengthArray[rank], MPI_SEEK_SET);
	MPI_File_write(mpi_compressedFile, frequency, 256, MPI_UNSIGNED, MPI_STATUS_IGNORE);
	MPI_File_write(mpi_compressedFile, inputFileData, (compBlockLength - 1024), MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);
	
	//close open files
	MPI_File_close(&mpi_compressedFile); 	
	MPI_File_close(&mpi_inputFile);
	
	free(inputFileData);
	free(compressedDataOffset);
	free(compBlockLengthArray);
	MPI_Finalize();
}
