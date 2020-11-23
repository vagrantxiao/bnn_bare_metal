#include "Typedefs.h"
void bin_dense_wt_gen_2(hls::stream< Word > & Input_1, hls::stream< Word > & Output_1){
#pragma HLS INTERFACE ap_hs port=Input_1
#pragma HLS INTERFACE ap_hs port=Output_1
#include "bin_dense_wt_par_2.h"
 loop_redir: for(int i=0; i<20482; i++){
#pragma HLS PIPELINE II=1
    Output_1.write(Input_1.read());
  }
 loop_0: for(int i=0; i<8192; i++){
#pragma HLS PIPELINE II=1
  Output_1.write(bin_dense_wt_2_0[i]);
  }
 loop_1: for(int i=0; i<1; i++){
#pragma HLS PIPELINE II=1
  Output_1.write(bin_dense_wt_2_1[i]);
  }
}
