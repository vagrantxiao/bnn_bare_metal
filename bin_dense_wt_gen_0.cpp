#include "Typedefs.h"
void bin_dense_wt_gen_0(hls::stream< Word > & Output_1){
#pragma HLS INTERFACE ap_hs port=Output_1
#include "bin_dense_wt_par_0.h"
 loop_0: for(int i=0; i<16384; i++){
#pragma HLS PIPELINE II=1
  Output_1.write(bin_dense_wt_0_0[i]);
  }
 loop_1: for(int i=0; i<1; i++){
#pragma HLS PIPELINE II=1
  Output_1.write(bin_dense_wt_0_1[i]);
  }
}
