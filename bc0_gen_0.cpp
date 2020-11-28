#include "Typedefs.h"
void bc0_gen_0(hls::stream< Word > & Output_1){
#pragma HLS INTERFACE ap_hs port=Output_1
#include "bc0_par_0.h"
 loop_0: for(int i=0; i<8192; i++){
#pragma HLS PIPELINE II=1
  Output_1.write(bc0_0_0[i]);
  }
 loop_1: for(int i=0; i<4096; i++){
#pragma HLS PIPELINE II=1
  Output_1.write(bc0_0_1[i]);
  }
 loop_2: for(int i=0; i<512; i++){
#pragma HLS PIPELINE II=1
  Output_1.write(bc0_0_2[i]);
  }
 loop_3: for(int i=0; i<116; i++){
#pragma HLS PIPELINE II=1
  Output_1.write(bc0_0_3[i]);
  }
}
