GIR_INC_DIR := ${GIR_SRC}/src
GIR_LIB_DIR := ${GIR_SRC}/lib
FFTW_LIB_DIR := /home/mirl/jhulet/opt/fftw3/lib

aws-recon: aws-recon.cpp
	${CXX} -I ${GIR_INC_DIR} -L${GIR_LIB_DIR} -L${FFTW_LIB_DIR} -lgir-base -lfftw3f -o $@ $<

clear-data:
	rm -f /DATA/aws-data/g_5ch_rep0_24.ser.ch_*
	rm -f /DATA/aws-data/g_5ch.ser.ch_*
	rm -rf /DATA/aws-data/aws-gpu*

clean:
	rm -f aws-recon
