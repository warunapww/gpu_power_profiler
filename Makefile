NVCC	:=	nvcc

CUDA_CC	:=  $(COMPUTE_CAPABILITY)

CUDA_OPTIONS	:=	-O3 -Xptxas -v -arch=${CUDA_CC} -maxrregcount 255
OPTIONS	:=	-O3
LIB := -L/s/chopin/e/proj/AlphaZ/waruna/lib -L/s/chopin/e/proj/AlphaZ/waruna/papi/installation/5.3.0/lib -lnvidia-ml -lpapi

EXES	:= 	smith_waterman

OBJECTS	:=	smith_waterman_kernel.o high_resolution_power.o

all:	${EXES}

smith_waterman_kernel.o:	smith_waterman_kernel.cu smith_waterman.h
	${NVCC} -c -o $@  $< ${CUDA_OPTIONS} 
	
smith_waterman:	smith_waterman.cu smith_waterman.h smith_waterman_kernel.o high_resolution_power.o
	${NVCC}  -o $@ $< ${OBJECTS} ${CUDA_OPTIONS} ${LIB}
	
high_resolution_power.o:	high_resolution_power.cu high_resolution_power.h
	$(NVCC) $< ${OPTIONS} -c -o $@

clean:
	rm -rf *.o ${EXES} *.ptx
	
ptx:	smith_waterman_kernel.cu
	${NVCC} ${CUDA_OPTIONS} -ptx $<
