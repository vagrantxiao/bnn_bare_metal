#include "Typedefs.h"
void bc2_gen_1(hls::stream< Word > & Input_1, hls::stream< Word > & Output_1){
#pragma HLS INTERFACE ap_hs port=Input_1
#pragma HLS INTERFACE ap_hs port=Output_1
#include "bc2_par_1.h"
 loop_redir: for(int i=0; i<20480; i++){
#pragma HLS PIPELINE II=1
    Output_1.write(Input_1.read());
  }
 loop_0: for(int i=0; i<8192; i++){
#pragma HLS PIPELINE II=1
  Output_1.write(bc2_1_0[i]);
  }
 loop_1: for(int i=0; i<512; i++){
#pragma HLS PIPELINE II=1
  Output_1.write(bc2_1_1[i]);
  }
 loop_2: for(int i=0; i<48; i++){
#pragma HLS PIPELINE II=1
  Output_1.write(bc2_1_2[i]);
  }
}
