#include "Typedefs.h"
void bin_dense_wt_gen_8(hls::stream< Word > & Input_1, hls::stream< Word > & Output_1){
#pragma HLS INTERFACE ap_hs port=Input_1
#pragma HLS INTERFACE ap_hs port=Output_1
#include "bin_dense_wt_par_8.h"
 loop_redir: for(int i=0; i<124928; i++){
#pragma HLS PIPELINE II=1
    Output_1.write(Input_1.read());
  }
 loop_0: for(int i=0; i<8192; i++){
#pragma HLS PIPELINE II=1
  Output_1.write(bin_dense_wt_8_0[i]);
  }
 loop_1: for(int i=0; i<4096; i++){
#pragma HLS PIPELINE II=1
  Output_1.write(bin_dense_wt_8_1[i]);
  }
 loop_2: for(int i=0; i<2048; i++){
#pragma HLS PIPELINE II=1
  Output_1.write(bin_dense_wt_8_2[i]);
  }
}
