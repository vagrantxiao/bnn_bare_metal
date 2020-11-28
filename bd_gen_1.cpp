#include "Typedefs.h"
void bd_gen_1(hls::stream< Word > & Input_1, hls::stream< Word > & Output_1){
#pragma HLS INTERFACE ap_hs port=Input_1
#pragma HLS INTERFACE ap_hs port=Output_1
#include "bd_par_1.h"
 loop_redir: for(int i=0; i<24576; i++){
#pragma HLS PIPELINE II=1
    Output_1.write(Input_1.read());
  }
 loop_0: for(int i=0; i<16384; i++){
#pragma HLS PIPELINE II=1
  Output_1.write(bd_1_0[i]);
  }
 loop_1: for(int i=0; i<8192; i++){
#pragma HLS PIPELINE II=1
  Output_1.write(bd_1_1[i]);
  }
}
