#include "stdio.h"
#include "label.h"
#include "Typedefs.h"
#include "data_gen_num.h"
#include "fp_conv.h"
#include "bc_gen_0.h"
#include "bc_gen_1.h"
#include "bc_gen_2.h"
#include "bc_gen_3.h"
#include "bd_gen_0.h"
#include "bd_gen_1.h"
#include "bd_gen_2.h"
#include "bd_gen_3.h"
#include "bd_gen_4.h"
#include "bd_gen_5.h"
#include "bd_gen_6.h"
#include "bd_gen_7.h"
#include "bd_gen_8.h"
#include "bd_gen_9.h"
#include "bin_conv_0.h"
#include "bin_conv_1.h"
#include "bin_conv_2.h"
#include "bin_dense_wrapper.h"
#include "data_in_gen_0.h"
#include "data_in_gen_1.h"
#include "data_in_gen_2.h"
#include "data_in_gen_3.h"
#include "data_in_gen_4.h"
#include "bin_conv_gen.h"
#include "bin_conv_gen0.h"
#include "bc_gen_0.h"
#include "bin_conv_gen1.h"
#include "bin_conv_gen2.h"

#include "bc0_gen_0.h"

#include "bc1_gen_0.h"
#include "bc1_gen_1.h"
#include "bc1_gen_2.h"

#include "bc2_gen_0.h"
#include "bc2_gen_1.h"


int main(int argc, char** argv) {
  hls::stream< Word > data_gen_out0("data_gen_out0");
  hls::stream< Word > data_gen_out1("data_gen_out1");
  hls::stream< Word > data_gen_out2("data_gen_out2");
  hls::stream< Word > data_gen_out3("data_gen_out3");
  hls::stream< DMA_Word > data_gen_out4("data_gen_out4");
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
	hls::stream< Word > fp_conv_in1("fp_conv_in1");
	hls::stream< Word > fp_conv_in2("fp_conv_in2");
	hls::stream< Word > fp_conv_out1("fp_conv_out1");
	hls::stream< Word > bin_conv_in1("bin_conv_in1");
	hls::stream< Word > bin_conv_in2("bin_conv_in2");
	hls::stream< Word > bin_conv_out1("bin_conv_out1");
	hls::stream< Word > bin_conv0_out1("bin_conv0_out1");
	hls::stream< Word > bin_conv1_out1("bin_conv1_out1");
	hls::stream< Word > bin_conv2_out1("bin_conv2_out1");
	hls::stream< Word > bin_dense_in1("bin_dense_in1");
	hls::stream< Word > bin_dense_in2("bin_dense_in2");
	hls::stream< DMA_Word > bin_dense_out1("bin_dense_out1");
	hls::stream< Word > bin_conv_gen_out0("bin_conv_gen_out0");
	hls::stream< Word > bin_conv_gen_out1("bin_conv_gen_out1");
	hls::stream< Word > bin_conv_gen_out2("bin_conv_gen_out2");
	hls::stream< Word > bin_conv_gen_out3("bin_conv_gen_out3");
	hls::stream< Word > bin_dense_gen_out0("bin_dense_gen_out0");
	hls::stream< Word > bin_dense_gen_out1("bin_dense_gen_out1");
	hls::stream< Word > bin_dense_gen_out2("bin_dense_gen_out2");
	hls::stream< Word > bin_dense_gen_out3("bin_dense_gen_out3");
	hls::stream< Word > bin_dense_gen_out4("bin_dense_gen_out4");
	hls::stream< Word > bin_dense_gen_out5("bin_dense_gen_out5");
	hls::stream< Word > bin_dense_gen_out6("bin_dense_gen_out6");
	hls::stream< Word > bin_dense_gen_out7("bin_dense_gen_out7");
	hls::stream< Word > bin_dense_gen_out8("bin_dense_gen_out8");
	hls::stream< Word > bin_dense_gen_out9("bin_dense_gen_out9");
	hls::stream< Word > bin_dense_gen_out10("bin_dense_gen_out10");

	hls::stream< Word > bin_conv_gen0_out1("bin_conv_gen0_out1");
	hls::stream< Word > bin_conv_gen1_out1("bin_conv_gen1_out1");
	hls::stream< Word > bin_conv_gen2_out1("bin_conv_gen2_out1");

	Word dmem_o[2*2*64];

	data_in_gen_0(data_gen_out0);
	data_in_gen_1(data_gen_out0, data_gen_out1);
	data_in_gen_2(data_gen_out1, data_gen_out2);
	data_in_gen_3(data_gen_out2, data_gen_out3);
	data_in_gen_4(N_IMG, data_gen_out3, data_gen_out4);



	//data_gen_num(N_IMG, data_gen_out1);


	for(i=0; i<N_IMG; i++)
	{
		printf("We are processing %d images\n", i);

		//bc_gen_0(bin_conv_gen_out0);
		//bc_gen_1(bin_conv_gen_out0, bin_conv_gen_out1);
		//bc_gen_2(bin_conv_gen_out1, bin_conv_gen_out2);
		//bc_gen_3(bin_conv_gen_out2, bin_conv_gen_out3);
		bc0_gen_0(bin_conv_gen0_out1);

		hls::stream< Word > bc1_gen_0_out1("bc1_gen_0_out1");
		hls::stream< Word > bc1_gen_1_out1("bc1_gen_1_out1");
		hls::stream< Word > bc1_gen_2_out1("bc1_gen_2_out1");
		bc1_gen_0(bc1_gen_0_out1);
		bc1_gen_1(bc1_gen_0_out1, bc1_gen_1_out1);
		bc1_gen_2(bc1_gen_1_out1, bc1_gen_2_out1);


		hls::stream< Word > bc2_gen_0_out1("bc2_gen_0_out1");
		hls::stream< Word > bc2_gen_1_out1("bc2_gen_1_out1");
		bc2_gen_0(bc2_gen_0_out1);
		bc2_gen_1(bc2_gen_0_out1, bc2_gen_1_out1);


		hls::stream< Word > bd_gen_0_out1("bd_gen_0_out1");
		hls::stream< Word > bd_gen_1_out1("bd_gen_1_out1");
		hls::stream< Word > bd_gen_2_out1("bd_gen_2_out1");
		hls::stream< Word > bd_gen_3_out1("bd_gen_3_out1");
		hls::stream< Word > bd_gen_4_out1("bd_gen_4_out1");
		hls::stream< Word > bd_gen_5_out1("bd_gen_5_out1");
		hls::stream< Word > bd_gen_6_out1("bd_gen_6_out1");
		hls::stream< Word > bd_gen_7_out1("bd_gen_7_out1");
		hls::stream< Word > bd_gen_8_out1("bd_gen_8_out1");
		hls::stream< Word > bd_gen_9_out1("bd_gen_9_out1");

		bd_gen_0(bd_gen_0_out1);
		bd_gen_1(bd_gen_0_out1, bd_gen_1_out1);
		bd_gen_2(bd_gen_1_out1, bd_gen_2_out1);
		bd_gen_3(bd_gen_2_out1, bd_gen_3_out1);
		bd_gen_4(bd_gen_3_out1, bd_gen_4_out1);
		bd_gen_5(bd_gen_4_out1, bd_gen_5_out1);
		bd_gen_6(bd_gen_5_out1, bd_gen_6_out1);
		bd_gen_7(bd_gen_6_out1, bd_gen_7_out1);
		bd_gen_8(bd_gen_7_out1, bd_gen_8_out1);


		fp_conv(data_gen_out4,
				fp_conv_out1
			    );

		for(j=0; j<3; j++){
			//printf("bin_conv_wrapper_0=%d\n", j);
			bin_conv_0(bin_conv_gen0_out1,
					 fp_conv_out1,
					 bin_conv0_out1);
		}

		for(j=0; j<7; j++){
			bin_conv_1(bc1_gen_2_out1,
					bin_conv0_out1,
					 bin_conv1_out1);
		}

		for(j=0; j<6; j++){
			bin_conv_2(bc2_gen_1_out1,
					bin_conv1_out1,
					 bin_conv2_out1);
		}

		for(j=0; j<37; j++){
			bin_dense_wrapper(bd_gen_8_out1,
					  bin_conv2_out1,
					  bin_dense_out1);
		}

		for(j=0; j<2*2*64; j++)
		{
			dmem_o[j] = bin_dense_out1.read();
		}

		int recv_cnt = 0;
		recv_cnt = (int) dmem_o[0](31,0);

		printf("We will receive %d\n", recv_cnt);

		ap_int<8> p = 0;
        p(7,0) = dmem_o[1](7,0);

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
