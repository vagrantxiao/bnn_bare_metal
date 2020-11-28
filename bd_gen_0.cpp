#include "Typedefs.h"
void bd_gen_0(hls::stream< Word > & Output_1){
#pragma HLS INTERFACE ap_hs port=Output_1
#include "bd_par_0.h"
 loop_0: for(int i=0; i<16384; i++){
#pragma HLS PIPELINE II=1
  Output_1.write(bd_0_0[i]);
  }
 loop_1: for(int i=0; i<4096; i++){
#pragma HLS PIPELINE II=1
  Output_1.write(bd_0_1[i]);
  }
 loop_2: for(int i=0; i<498; i++){
#pragma HLS PIPELINE II=1
  Output_1.write(bd_0_2[i]);
  }
}
