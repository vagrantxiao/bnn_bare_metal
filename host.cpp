#include "stdio.h"
#include "Accel.h"
#include "input.h"
#include "kh.h"
#include "wt.h"
#include "data.h"
#include "label.h"

int main(int argc, char** argv) {
  hls::stream< Word > data_gen_out1("data_gen_out1");
  unsigned N_IMG;
  if (argc < 2) {
    printf ("We will use default N_IMG = 1\n");
    N_IMG  = 1;
  }else{
	N_IMG  = std::stoi(argv[1]);
  }

	printf("Hello world\n");
	int i;
	int j;
	int err_cnt = 0;

	Word dmem_o[DMEM_O_WORDS];

	data_gen(N_IMG, data_gen_out1);

	for(i=0; i<N_IMG; i++)
	{
		printf("Processing Numer %d image\n", i);
		for(j=0; j<38; j++)
		{
			printf("Processing layer %d\n", j);
			top(
				data_gen_out1,
				&wt_i[WT_WORDS*j],
				&kh_i[KH_WORDS*j],
				&data_in[1024*i],
				dmem_o,
				n_inputs[j],
				n_outputs[j],
				input_words[j],
				output_words[j],
				layer_mode[j],  // [0]='new layer', [2:1]='conv1,conv,dense,last'
				dmem_mode[j],   // 0 means dmem[0] is input
				width_mode[j],  // 0=8'b, 1=16'b, 2=32'b
				norm_mode[j]    // 0='do nothing', 1='do norm', 2='do pool'
				);
		}
		ap_int<8> p = 0;
        p(7,0) = dmem_o[0](7,0);

        int prediction = p.to_int();
        if(prediction == y[i]){
        	printf("Pred/Label: %d/%d [ OK ]\n", prediction, y[i]);
        }else{
        	printf("Pred/Label: %d/%d [FAIL]\n", prediction, y[i]);
        	err_cnt++;
        }
	}

	printf("We got %d/%d errors\nDone\n", err_cnt, N_IMG);

	return 0;
}
