#include "Typedefs.h"

void data_gen_num(
	int image_num,
	hls::stream< DMA_Word > & Output_1
) {
  #include "data.h"
  for(int i=0; i<image_num*1024; i++)
  {
	  DMA_Word out_tmp;
	  out_tmp(127,64) = 0;
	  out_tmp(63,  0) = data_in[i];
	  Output_1.write(out_tmp);
  }
}


