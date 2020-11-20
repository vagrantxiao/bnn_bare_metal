#include "Typedefs.h"

void fp_conv_gen(
	hls::stream< Word > & Output_1
) {
#pragma HLS INTERFACE ap_hs port=Output_1  
#include "fp_conv_par.h"
 fp_conv_gen_label0: fp_conv_gen_loop: for(int kh_i=0; kh_i<192; kh_i++){
#pragma HLS PIPELINE II=1
	  Output_1.write(fp_conv_wt[kh_i]);
  }
}
