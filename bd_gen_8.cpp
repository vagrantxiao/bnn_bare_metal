#include "Typedefs.h"
void bd_gen_8(hls::stream< Word > & Input_1, hls::stream< Word > & Output_1){
#pragma HLS INTERFACE ap_hs port=Input_1
#pragma HLS INTERFACE ap_hs port=Output_1
#include "bd_par_8.h"
 loop_redir: for(int i=0; i<141810; i++){
#pragma HLS PIPELINE II=1
    Output_1.write(Input_1.read());
  }
 loop_0: for(int i=0; i<16384; i++){
#pragma HLS PIPELINE II=1
  Output_1.write(bd_8_0[i]);
  }
 loop_1: for(int i=0; i<512; i++){
#pragma HLS PIPELINE II=1
  Output_1.write(bd_8_1[i]);
  }
}
