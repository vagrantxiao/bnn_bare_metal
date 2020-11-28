#include "Typedefs.h"
void bc1_gen_1(hls::stream< Word > & Input_1, hls::stream< Word > & Output_1){
#pragma HLS INTERFACE ap_hs port=Input_1
#pragma HLS INTERFACE ap_hs port=Output_1
#include "bc1_par_1.h"
 loop_redir: for(int i=0; i<20480; i++){
#pragma HLS PIPELINE II=1
    Output_1.write(Input_1.read());
  }
 loop_0: for(int i=0; i<8192; i++){
#pragma HLS PIPELINE II=1
  Output_1.write(bc1_1_0[i]);
  }
}
