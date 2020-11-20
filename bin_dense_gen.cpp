#include "Typedefs.h"

void bin_dense_gen(
	hls::stream< Word > & Output_1
) {
#pragma HLS INTERFACE ap_hs port=Output_1
#include "bin_dense_par.h"
  bin_dense_gen_loop: for(int i=0; i<175602; i++){
#pragma HLS PIPELINE II=1
	  Output_1.write(bin_dense_wt[i]);
  }
}
