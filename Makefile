NVCC	:=	nvcc

CUDA_CC	:=  $(COMPUTE_CAPABILITY)

CUDA_OPTIONS	:=	-O3 -Xptxas -v -arch=${CUDA_CC} -maxrregcount 255
OPTIONS	:=	-O3
#LIB := -L/s/chopin/e/proj/AlphaZ/waruna/lib -L/s/chopin/e/proj/AlphaZ/waruna/papi/installation/5.3.0/lib -lnvidia-ml -lpapi
LIB := -L/s/chopin/e/proj/AlphaZ/waruna/papi/installation/5.3.0/lib -lnvidia-ml -lpapi -lcublas

EXES	:=  jacobi_power

OBJECTS	:=	jacobi_kernel.o high_resolution_power.o gpu_heater.o

all:	${EXES}

	
jacobi_power:	jacobi_host.cu jacobi_kernel.hu jacobi_kernel.o high_resolution_power.o gpu_heater.o
	${NVCC}  -o $@ $< ${OBJECTS} ${CUDA_OPTIONS} ${LIB}
	
high_resolution_power.o:	high_resolution_power.cu high_resolution_power.h gpu_heater.o
	$(NVCC) $< ${OPTIONS} -c -o $@

jacobi_kernel.o:	jacobi_kernel.cu jacobi_kernel.hu
	${NVCC} -c -o $@ $< ${CUDA_OPTIONS} 

gpu_heater.o: gpu_heater.cu gpu_heater.h
	${NVCC} -c -o $@ $< ${CUDA_OPTIONS}

clean:
	rm -rf *.o ${EXES} *.ptx
	
ptx:	jacobi_kernel.cu
	${NVCC} ${CUDA_OPTIONS} -ptx $<
