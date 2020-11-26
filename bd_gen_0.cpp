#include "Typedefs.h"
void bd_gen_0(hls::stream< Word > & Output_1){
#pragma HLS INTERFACE ap_hs port=Output_1
#include "bd_par_0.h"
 loop_0: for(int i=0; i<8192; i++){
#pragma HLS PIPELINE II=1
  Output_1.write(bd_0_0[i]);
  }
}
