
void fp_conv(
	hls::stream< Word > & Input_1,//Word wt_mem[CONVOLVERS][C_WT_WORDS],
	hls::stream< DMA_Word > & Input_2,
	hls::stream< Word > & Output_1
);
