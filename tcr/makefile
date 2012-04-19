GIR_INC_DIR := ${GIR_SRC}/src
GIR_LIB_DIR := ${GIR_SRC}/lib
#GIR_PLUGINS_DIR := ../../main/plugins

CUDA_LIB_DIR=/opt/NVIDIA_CUDA_SDK/cuda/lib
CUDA_INC_DIR=/opt/NVIDIA_CUDA_SDK/cuda/include

# fftw_flags
FFTW_PATH := /home/mirl/jhulet/opt/fftw3
FFTW_LIB := -L${FFTW_PATH}/lib -lfftw3f

CXX_ARGS := -I./ -I${GIR_INC_DIR}

Plugin_TCR.so: Plugin_TCR.cpp TCRIterator.o TCRIteratorCPU.o KernelCode_CPU.o
	${CXX} ${CXX_ARGS} -DNO_CUDA -L${GIR_LIB_DIR} -I${CUDA_INC_DIR} TCRIterator.o TCRIteratorCPU.o KernelCode_CPU.o -lgir-base -shared -fPIC -o $@ $<
	#cp $@ ${GIR_PLUGINS_DIR}

%.o: %.cpp
	${CXX} ${CXX_ARGS} -fPIC -c -o $@ $<

KernelCode_CPU.o: KernelCode.cpp
	${CXX} ${CXX_ARGS} -DTCR_KERNEL_CPU -fPIC -c -o $@ $<

TCRIteratorCPU.o: TCRIteratorCPU.cpp
	${CXX} ${CXX_ARGS} -DTCR_KERNEL_CPU -fPIC -c -o $@ $<

KernelCode_CUDA.o: KernelCode.cu
	nvcc ${CXX_ARGS} -DTCR_KERNEL_CUDA -c -o $@ $< --compiler-options -fPIC

TCRIteratorCUDA.o: TCRIteratorCUDA.cu
	nvcc ${CXX_ARGS} -DTCR_KERNEL_CUDA -c -o $@ $< --compiler-options -fPIC

# CPU / CUDA
mpi-tcr: mpi-tcr.cpp TCRIterator.o TCRIteratorCPU.o MPITools.cpp KernelCode_CPU.o KernelCode_CUDA.o MPIPartitioner.o TCRIteratorCUDA.o
	mpicxx TCRIterator.o MPITools.cpp TCRIteratorCPU.o TCRIteratorCUDA.o KernelCode_CPU.o KernelCode_CUDA.o MPIPartitioner.o -limf -lm -I${CUDA_INC_DIR} ${CXX_ARGS} -L${GIR_LIB_DIR} -L${CUDA_LIB_DIR} -L${FFTW_PATH}/lib -lcudart -lcufft -lgir-base -lfftw3f -ldl -o $@ $<

serial-tcr: serial-tcr.cpp TCRIterator.o TCRIteratorCPU.o MPITools.cpp KernelCode_CPU.o KernelCode_CUDA.o MPIPartitioner.o TCRIteratorCUDA.o
	${CXX} TCRIterator.o TCRIteratorCPU.o TCRIteratorCUDA.o KernelCode_CPU.o KernelCode_CUDA.o MPIPartitioner.o -I${CUDA_INC_DIR} ${CXX_ARGS} -L${GIR_LIB_DIR} -L${CUDA_LIB_DIR} -L${FFTW_PATH}/lib -lgir-base -lcudart -lcufft -lfftw3f -ldl -o $@ $<
	#${CXX} TCRIterator.o TCRIteratorCPU.o TCRIteratorCUDA.o KernelCode_CPU.o KernelCode_CUDA.o MPIPartitioner.o -limf -lm -I${CUDA_INC_DIR} ${CXX_ARGS} -L${GIR_LIB_DIR} -L${CUDA_LIB_DIR} -L${FFTW_PATH}/lib -lcudart -lcufft -lgir-base -lfftw3f -ldl -o $@ $<

clean:
	rm -f *.o
	rm -f *.so
	rm -f mpi-tcr
	rm -f serial-tcr
