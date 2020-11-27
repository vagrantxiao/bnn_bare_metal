#include "Typedefs.h"
void bc_gen_3(hls::stream< Word > & Input_1, hls::stream< DMA_Word > & Output_1){
#pragma HLS INTERFACE ap_hs port=Input_1
#pragma HLS INTERFACE ap_hs port=Output_1
DMA_Word out_tmp;

	#include "bc_par_3.h"
 loop_redir: for(int i=0; i<61440/2; i++){
#pragma HLS PIPELINE II=1
	out_tmp(127, 64) = Input_1.read();
	out_tmp(63,  0)  = Input_1.read();
    Output_1.write(out_tmp);
  }
 loop_0: for(int i=0; i<8192/2; i++){
#pragma HLS PIPELINE II=1
  out_tmp(127, 64) = bc_3_0[2*i];
  out_tmp(63,  0)  = bc_3_0[2*i+1];
  Output_1.write(out_tmp);
  }
 loop_1: for(int i=0; i<4096/2; i++){
#pragma HLS PIPELINE II=1
	  out_tmp(127, 64) = bc_3_1[2*i];
	  out_tmp(63,  0)  = bc_3_1[2*i+1];
	  Output_1.write(out_tmp);
  }
 loop_2: for(int i=0; i<2048/2; i++){
#pragma HLS PIPELINE II=1
	  out_tmp(127, 64) = bc_3_2[2*i];
	  out_tmp(63,  0)  = bc_3_2[2*i+1];
	  Output_1.write(out_tmp);
  }
 loop_3: for(int i=0; i<1024/2; i++){
#pragma HLS PIPELINE II=1
	  out_tmp(127, 64) = bc_3_3[2*i];
	  out_tmp(63,  0)  = bc_3_3[2*i+1];
	  Output_1.write(out_tmp);
  }
 loop_4: for(int i=0; i<92/2; i++){
#pragma HLS PIPELINE II=1
	  out_tmp(127, 64) = bc_3_4[2*i];
	  out_tmp(63,  0)  = bc_3_4[2*i+1];
	  Output_1.write(out_tmp);
  }
}
