#include "stdio.h"
#include "Accel.h"
#include "input.h"
#include "kh.h"
#include "wt.h"
#include "data.h"
#include "label.h"

#define N_IMG 10
int main()
{
	printf("Hello world\n");
	int i;
	int j;
	int err_cnt = 0;

	Word dmem_o[DMEM_O_WORDS];

	for(i=0; i<N_IMG; i++)
	{
		printf("Processing Numer %d image\n", i);
		for(j=0; j<54; j++)
		{
			//printf("Processing layer %d\n", i);
			top(
				&wt_i[WT_WORDS*j],
				&kh_i[KH_WORDS*j],
				&data_in[DMEM_WORDS*i],
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

	printf("We got %d/1000 errors\n", err_cnt);

	return 0;
}
