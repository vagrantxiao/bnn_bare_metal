# Makefile for BNN of Rosetta benchmarks
#
# Author: Yuanlong Xiao (ylxiao@seas.upenn.edu)
#
# Targets:
#   all   - Builds hardware and software in SDSoC.

OBJ=bin_conv.o bin_dense.o  fp_conv.o  host.o\
    bin_conv_wt_gen_0.o bin_conv_wt_gen_1.o bin_conv_wt_gen_2.o bin_conv_wt_gen_3.o\
    bin_dense_wt_gen_0.o bin_dense_wt_gen_1.o bin_dense_wt_gen_2.o bin_dense_wt_gen_3.o\
    bin_dense_wt_gen_4.o bin_dense_wt_gen_5.o bin_dense_wt_gen_6.o bin_dense_wt_gen_7.o\
    bin_dense_wt_gen_8.o bin_dense_wt_gen_9.o bin_dense_wt_gen_10.o\
    data_in_gen_0.o data_in_gen_1.o data_in_gen_2.o data_in_gen_3.o data_in_gen_4.o





INCLUDE=-I /opt/Xilinx/Vivado/2018.2/include 
OPT_LEVEL=-O3
CFLAGS=$(INCLUDE) $(OPT_LEVEL)
CXX=g++
VPATH=src




all: main

main:$(OBJ)
	$(CXX) $(CFLAGS) -o main $(OBJ) 

$(OBJ):%.o:%.cpp
	$(CXX) $(CFLAGS) -o $@ -c $^



install:
	echo hello

print: 
	ls ./src

tar:
	tar -czvf ./src.tar.gz ./src/ 


try:
	echo $(notdir $(wildcard ./src)) 



clean:
	rm -rf ./*.o main


















