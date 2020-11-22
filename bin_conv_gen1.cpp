#include "Typedefs.h"


void bin_conv_gen1(
	hls::stream< Word > & Output_1
) {
#pragma HLS INTERFACE ap_hs port=Output_1
#include "bin_conv_new.h"
  bin_conv_gen1_loop: for(int i=0; i<75868; i++){
#pragma HLS PIPELINE II=1
	  Output_1.write(bin_conv_new[i]);
  }
}








