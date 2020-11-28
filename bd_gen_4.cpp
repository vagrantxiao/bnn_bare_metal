#include "Typedefs.h"
void bd_gen_4(hls::stream< Word > & Input_1, hls::stream< Word > & Output_1){
#pragma HLS INTERFACE ap_hs port=Input_1
#pragma HLS INTERFACE ap_hs port=Output_1
#include "bd_par_4.h"
 loop_redir: for(int i=0; i<81920; i++){
#pragma HLS PIPELINE II=1
    Output_1.write(Input_1.read());
  }
 loop_0: for(int i=0; i<16384; i++){
#pragma HLS PIPELINE II=1
  Output_1.write(bd_4_0[i]);
  }
 loop_1: for(int i=0; i<1024; i++){
#pragma HLS PIPELINE II=1
  Output_1.write(bd_4_1[i]);
  }
}
