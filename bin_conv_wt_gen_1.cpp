#include "Typedefs.h"
void bin_conv_wt_gen_1(hls::stream< Word > & Input_1, hls::stream< Word > & Output_1){
#pragma HLS INTERFACE ap_hs port=Input_1
#pragma HLS INTERFACE ap_hs port=Output_1
#include "bin_conv_wt_par_1.h"
 loop_redir: for(int i=0; i<20480; i++){
#pragma HLS PIPELINE II=1
    Output_1.write(Input_1.read());
  }
 loop_0: for(int i=0; i<16384; i++){
#pragma HLS PIPELINE II=1
  Output_1.write(bin_conv_wt_1_0[i]);
  }
 loop_1: for(int i=0; i<4096; i++){
#pragma HLS PIPELINE II=1
  Output_1.write(bin_conv_wt_1_1[i]);
  }
}