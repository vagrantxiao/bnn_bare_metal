#include "Typedefs.h"
#include "load_kh.h"
#include "bin_conv.h"

void bin_conv_wrapper_1(

	hls::stream< Word > & Input_1,
	hls::stream< Word > & Input_2,
	hls::stream< Word > & Output_1
) {
#pragma HLS INTERFACE ap_hs port=Input_1
#pragma HLS INTERFACE ap_hs port=Input_2
#pragma HLS INTERFACE ap_hs port=Output_1

	static unsigned int bin_conv_cnt = 0;
	static Word dmem[2][CONVOLVERS][C_DMEM_WORDS];
#pragma HLS ARRAY_PARTITION variable=dmem complete dim=2
#pragma HLS ARRAY_PARTITION variable=dmem complete dim=1
    Word wt_mem[CONVOLVERS][C_WT_WORDS];
#pragma HLS ARRAY_PARTITION variable=wt_mem complete dim=1
	Word kh_mem[KH_WORDS];

    ap_uint<1> d_i_idx_list[] =          {0,  1,  1,  1,  1,  0,  0  };
    ap_uint<1> d_o_idx_list[]  =         {1,  0,  0,  0,  0,  1,  1  };
    const Address n_inputs_list[] =      {256,256,256,256,256,512,512};
    const Address o_index_list[] =       {128,0,  128,256,384,0,  64 };
    const ap_uint<2> width_mode_list[] = {1,  0,  0,  0,  0,  0,  0  };
    const ap_uint<2> norm_mode_list[] =  {2,  1,  1,  1,  1,  2,  2  };
    const Address n_outputs_list[] =     {128,128,128,128,128,64, 64 };

    Address o_index = o_index_list[bin_conv_cnt];
    Address n_outputs = n_outputs_list[bin_conv_cnt];
    Address kh_index = 0;

    //printf("bin_conv_cnt=%d\n", bin_conv_cnt);


    for(unsigned int kh_i=0; kh_i<KH_WORDS; kh_i++)
    {
#pragma HLS PIPELINE
    	kh_mem[kh_i] = Input_1.read();

    	//printf("0x%08x%08x,\n", (unsigned int) kh_mem[kh_i](63,32), (unsigned int) kh_mem[kh_i](31,0));
    }


    if(bin_conv_cnt == 0)
    {
		for(unsigned int dmem_i=0; dmem_i<2; dmem_i++)
		  for(unsigned int dmem_j=0; dmem_j<CONVOLVERS; dmem_j++)
			for(unsigned int dmem_k=0; dmem_k<C_DMEM_WORDS; dmem_k++)
			{
#pragma HLS PIPELINE
				dmem[dmem_i][dmem_j][dmem_k] = Input_2.read();
			}

    }


    LOOP_IMG_BATCH:
    for (IdxType i = 0; i < n_outputs; ++i) {
      // Load the batch-norm parameters for this output
      NormComp nc;
      load_kh(nc, kh_mem, kh_index);

      bin_conv(
    	  Input_1,
          wt_mem,
          nc,
          dmem,
          d_i_idx_list[bin_conv_cnt],
		  d_o_idx_list[bin_conv_cnt],
          n_inputs_list[bin_conv_cnt],
          o_index,
          i == 0 ? 1 : 0,         // new_batch
          width_mode_list[bin_conv_cnt],
          norm_mode_list[bin_conv_cnt]
      );

      kh_index++;
      o_index++;
    }


    if(bin_conv_cnt == 6)
    {
	  for(unsigned int dmem_i=0; dmem_i<2; dmem_i++)
		for(unsigned int dmem_j=0; dmem_j<CONVOLVERS; dmem_j++)
		  for(unsigned int dmem_k=0; dmem_k<C_DMEM_WORDS; dmem_k++){
#pragma HLS PIPELINE
Output_1.write(dmem[dmem_i][dmem_j][dmem_k]);
			}
    }

    bin_conv_cnt++;
    if(bin_conv_cnt==7) bin_conv_cnt = 0;

}





// -----------------------------------------------------------------------
// Module to do the first conv layer
// -----------------------------------------------------------------------
