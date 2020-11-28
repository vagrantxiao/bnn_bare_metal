#include "Typedefs.h"
void bd_gen_5(hls::stream< Word > & Input_1, hls::stream< Word > & Output_1){
#pragma HLS INTERFACE ap_hs port=Input_1
#pragma HLS INTERFACE ap_hs port=Output_1
#include "bd_par_5.h"
 loop_redir: for(int i=0; i<122880; i++){
#pragma HLS PIPELINE II=1
    Output_1.write(Input_1.read());
  }
 loop_0: for(int i=0; i<16384; i++){
#pragma HLS PIPELINE II=1
  Output_1.write(bd_5_0[i]);
  }
 loop_1: for(int i=0; i<8192; i++){
#pragma HLS PIPELINE II=1
  Output_1.write(bd_5_1[i]);
  }
}
