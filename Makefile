# Makefile for BNN of Rosetta benchmarks
#
# Author: Yuanlong Xiao (ylxiao@seas.upenn.edu)
#
# Targets:
#   all   - Builds hardware and software in SDSoC.

OBJ=bin_conv.o bin_conv_gen.o bin_dense.o bin_dense_gen.o data_gen_num.o fp_conv.o fp_conv_gen.o host.o bin_conv_gen1.o
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


















