#include <iostream>
#include <iomanip>
#include <hls_stream.h>
#include "Accel.h"
#include "AccelPrint.h"
#include "stdio.h"

#include "bin_conv_par.h"
#include "bin_dense_par.h"

const static Word m1("0x5555555555555555", 16);
const static Word m2("0x3333333333333333", 16);
const static Word m4("0x0f0f0f0f0f0f0f0f", 16);
const static Word h01("0x0101010101010101", 16);

int fp_conv_cnt = 0;
int bin_conv_cnt = 0;
int bin_dense_cnt = 0;

// -----------------------------------------------------------------------
// Hardware-specific print helpers
// -----------------------------------------------------------------------
template<typename T>
void print_ap_bits(const T& in, const unsigned W) {
  printf ("   ");
  for (unsigned i = 0; i < W; ++i)
    printf ("%3d", in[i] ? -1 : 0);
  printf ("\n");
}

template<typename T>
void print_params(T params[CONVOLVERS][K][K]) {
  for (unsigned m = 0; m < CONVOLVERS; ++m) {
    for (unsigned wr = 0; wr < K; ++wr) {
      for (unsigned wc = 0; wc < K; ++wc) {
        printf ("%3d", (params[m][wr][wc]==0) ? 0 : 1);
      }
      printf("\n");
    }
    printf("--\n");
  }
}

template<typename T>
void print_line_buffer_m(T lbuf[CONV_BANKS]) {
  for (unsigned wr = 0; wr < CONV_ROWS; ++wr) {
  for (unsigned bank = 0; bank < CONV_BANKS; ++bank) {
    for (unsigned wc = 0; wc < CONV_COLS; ++wc) {
      printf ("%3d", lbuf[bank][wr][wc].to_int());
    }
    printf (" |");
  }
  printf ("\n");
  }
}

TwoBit encode_bit(const Bit& b) {
  return (b == 0) ? TwoBit(1) : TwoBit(-1);
}

// -----------------------------------------------------------------------
// Conv
// -----------------------------------------------------------------------
ConvOut conv3x3b(
    const TwoBit line_buffer_m[CONV_BANKS][CONV_ROWS][CONV_COLS],
    const Bit conv_params_m[K][K],
    const ap_uint<4> bank,
    const IdxType cc
) {
  ConvOut sum = 0;
  for (ap_uint<2> kr = 0; kr < K; ++kr) {
    for (ap_uint<2> kc = 0; kc < K; ++kc) {
      TwoBit data = line_buffer_m[bank][kr][cc+kc];
      const Bit& wt = conv_params_m[2-kr][2-kc];
      data[1] = (wt & data[0]) ^ data[1];
      sum += data;
    }
  }
  return sum;
}

// -----------------------------------------------------------------------
// Produce 32 elements of conv results
// -----------------------------------------------------------------------
void conv_word(
    const TwoBit line_buffer_m[CONV_BANKS][CONV_ROWS][CONV_COLS],
    const Bit conv_params_m[K][K],
    ConvOut conv_out_buffer_m[WORD_SIZE]
) {
  for (ap_uint<4> bank = 0; bank < CONV_BANKS; ++bank) {
    for (ap_uint<4> cc = 0; cc < BANK_WIDTH; ++cc) {
      conv_out_buffer_m[bank*BANK_WIDTH+cc] = conv3x3b( line_buffer_m, conv_params_m, bank, cc );
    }
  }
}

// -----------------------------------------------------------------------
// Process each line in a word, we need to outline this loop to
// avoid false control dependencies in Vivado HLS
// -----------------------------------------------------------------------
void process_word(
    const TwoBit  word_buffer_m[CONV_BANKS][CONV_COLS],
    const TwoBit  old_word_buffer_m[CONV_BANKS][CONV_COLS],
    const bool lb[CONV_BANKS],
    const bool rb[CONV_BANKS],
    TwoBit  line_buffer_m[CONV_BANKS][CONV_ROWS][CONV_COLS],
    const   Bit conv_params_m[K][K],
    ConvOut conv_out_buffer_m[WORD_SIZE],
    const   ap_uint<3> log_width,
    const   ap_uint<6> words_per_image,
    const   IdxType wrd
) {
  // slices_per_line = width / BANK_WIDTH
  const ap_uint<5> slices_per_line = 1 << (log_width - LOG_BANK_WIDTH);
  const bool first_wrd = (wrd == 0);
  const bool last_wrd = (wrd == words_per_image);
  DB_PRINT(4, "process word %d, spl=%d\n", wrd.to_int(), slices_per_line.to_int());

  // Prologue
  // Update bottom row, slices are shifted left. Some slices copied from previous word (middle row)
  for (ap_uint<4> bank = 0; bank < CONV_BANKS; ++bank) {
    ap_int<6> s_idx = bank + slices_per_line - CONV_BANKS;
    if (s_idx < 0) {
      // set to zero or copy from old word (middle row)
      for (ap_uint<4> cc = 1; cc < CONV_COLS-1; ++cc) {
        line_buffer_m[bank][CONV_ROWS-1][cc] = old_word_buffer_m[CONV_BANKS+s_idx][cc];
      }
      line_buffer_m[bank][CONV_ROWS-1][0          ] = lb[bank] ? TwoBit(0) : old_word_buffer_m[CONV_BANKS+s_idx][0];
      line_buffer_m[bank][CONV_ROWS-1][CONV_COLS-1] = rb[bank] ? TwoBit(0) : old_word_buffer_m[CONV_BANKS+s_idx][CONV_COLS-1];
    } else {
      // fill from new word
      for (ap_uint<4> cc = 1; cc < CONV_COLS-1; ++cc) {
        line_buffer_m[bank][CONV_ROWS-1][cc] = (last_wrd) ? TwoBit(0) : word_buffer_m[s_idx][cc];
      }
      line_buffer_m[bank][CONV_ROWS-1][0          ] = (last_wrd || lb[bank]) ? TwoBit(0) : word_buffer_m[s_idx][0];
      line_buffer_m[bank][CONV_ROWS-1][CONV_COLS-1] = (last_wrd || rb[bank]) ? TwoBit(0) : word_buffer_m[s_idx][CONV_COLS-1];
    }
  }
  
  DB(4,
    printf("Accel lbuf wrd%d before conv:\n", wrd.to_int());
    print_line_buffer_m(line_buffer_m);
  );

  // Convolution
  conv_word( line_buffer_m, conv_params_m, conv_out_buffer_m );
  
  // Update
  // Fill line buffer with lines from the new word
  for (ap_uint<4> bank = 0; bank < CONV_BANKS; ++bank) {
    // --------------------------------------------------------------
    // Top row, slices are shifted right by slices_per_line
    ap_int<6> s_idx0 = bank - slices_per_line;
    if (s_idx0 >= 0) {
      // slice from input word
      for (ap_uint<4> cc = 1; cc < CONV_COLS-1; ++cc) {
        line_buffer_m[bank][0][cc] = word_buffer_m[s_idx0][cc];
      }
      line_buffer_m[bank][0][0          ] = lb[bank] ? TwoBit(0) : word_buffer_m[s_idx0][0];
      line_buffer_m[bank][0][CONV_COLS-1] = rb[bank] ? TwoBit(0) : word_buffer_m[s_idx0][CONV_COLS-1];
    } else {
      // set to zero or copy from old word (middle row)
      for (ap_uint<4> cc = 1; cc < CONV_COLS-1; ++cc) {
        line_buffer_m[bank][0][cc] = (first_wrd) ? TwoBit(0) : old_word_buffer_m[CONV_BANKS+s_idx0][cc];
      }
      line_buffer_m[bank][0][0          ] = (first_wrd || lb[bank]) ? TwoBit(0) : old_word_buffer_m[CONV_BANKS+s_idx0][0];
      line_buffer_m[bank][0][CONV_COLS-1] = (first_wrd || rb[bank]) ? TwoBit(0) : old_word_buffer_m[CONV_BANKS+s_idx0][CONV_COLS-1];
    }

    // --------------------------------------------------------------
    // Middle row, simply copy the word into the line buffer
    for (ap_uint<4> cc = 1; cc < CONV_COLS-1; ++cc) {
      line_buffer_m[bank][1][cc] = word_buffer_m[bank][cc];
    }
    // Fill end buffer bits
    line_buffer_m[bank][1][0          ] = lb[bank] ? TwoBit(0) : word_buffer_m[bank][0];
    line_buffer_m[bank][1][CONV_COLS-1] = rb[bank] ? TwoBit(0) : word_buffer_m[bank][CONV_COLS-1];
  }

  DB(4,
    printf("Accel lbuf wrd%d after conv:\n", wrd.to_int());
    print_line_buffer_m(line_buffer_m);
  );
}

// -----------------------------------------------------------------------
// A single PE reads from all inputs and weights to generate a single
// output feature map.
// * Make sure this function gets inlined by VHLS, or cosim may fail!
// -----------------------------------------------------------------------
void bin_conv(
    Word wt_mem[CONVOLVERS][C_WT_WORDS],
    NormComp nc,
    Word dmem[2][CONVOLVERS][C_DMEM_WORDS],
    ap_uint<1> d_i_idx,
    ap_uint<1> d_o_idx,
    const unsigned   n_inputs,
    const Address    o_index,
    const ap_uint<1> new_batch,
    const ap_uint<2> width_mode,  // 0=8'b, 1=16'b, 2=32'b
    const ap_uint<2> norm_mode    // 0='do nothing', 1='do norm', 2='do pool'
) {
  const ap_uint<3> log_width = width_mode + LOG_BANK_WIDTH;
  const ap_uint<5> words_per_image = 1 << (2*width_mode);
  const unsigned n_phases = n_inputs / CONVOLVERS;
  const unsigned images_per_phase = PIX_PER_PHASE >> (2*log_width);
  const unsigned WORDS_PER_PHASE = PIX_PER_PHASE / WORD_SIZE;




  assert(n_phases % images_per_phase == 0);
  assert(n_inputs % images_per_phase == 0);
  assert(images_per_phase*words_per_image == WORDS_PER_PHASE);

  // ---------------------------------------------------------------------
  // buffers
  // ---------------------------------------------------------------------
  TwoBit  line_buffer[CONVOLVERS][CONV_BANKS][CONV_ROWS][CONV_COLS];
  Bit     conv_params[CONVOLVERS][K][K];
  ConvSum fixed_buffer[WORDS_PER_PHASE][WORD_SIZE];
  ConvSum fixed_temp[WORD_SIZE];
  // per-convolver buffers
  TwoBit  word_buffer[CONVOLVERS][CONV_BANKS][CONV_COLS];
  TwoBit  old_word_buffer[CONVOLVERS][CONV_BANKS][CONV_COLS];
  ConvOut conv_out_buffer[CONVOLVERS][WORD_SIZE];
  // edge padding flag bits
  bool lb[CONV_BANKS];
  bool rb[CONV_BANKS];

  static Address wt_addr = 0;           // address of weight word
  static ap_uint<3> wt_offset = 0;      // offset 0..6 of param
  if (new_batch != 0) { wt_addr = 0; wt_offset = 0; }

  // ---------------------------------------------------------------------
  // Calculate edge padding flag bits
  const ap_uint<4> log_slice = log_width - LOG_BANK_WIDTH;
  const ap_uint<4> w_div_8 = (1 << log_width) >> 3;
  assert (w_div_8 > 0);
  ap_uint<4> mask = ~ap_uint<4>(0);   // set mask to all 1s
  mask = mask >> (4-log_slice);
  for (ap_uint<4> bank = 0; bank < CONV_BANKS; ++bank) {
    #pragma HLS unroll
    const ap_uint<4> x = bank & mask;
    lb[bank] = (x == 0);          // (bank % w_div_8) == 0
    rb[bank] = (x+1 == w_div_8);  // (bank % w_div_8) == w_div_8-1
  }

  // ---------------------------------------------------------------------
  // Reset conv buffer
  for (IdxType i = 0; i < WORDS_PER_PHASE; ++i) {
    for (IdxType j = 0; j < WORD_SIZE; ++j) {
      #pragma HLS UNROLL
      fixed_buffer[i][j] = 0;
    }
  }

  // ---------------------------------------------------------------------
  // Compute in phases
  // Each phase processes CONVOLVERS * WORDS_PER_PHASE input words
  // ---------------------------------------------------------------------
  LOOP_PHASES:
  for (ap_uint<10> p = 0; p < n_phases; p += images_per_phase) {
    DB(3, printf ("=== PHASE %d ===\n", p.to_int()) );

    // wrd = which word in the current image
    // wrd_phase = which wrd in the current phase
    ap_uint<8> wrd = 0;
    ap_uint<8> wrd_phase = 0;

    // Load a word each iteration, and then process it
    // We load WORDS_PER_PHASE words per phase, however we also need 1 extra "empty"
    // iteration per image in the phase to do the loop epilogue, so the loop bound
    // is WORDS_PER_PHASE + images_per_phase
    LOOP_WORDS_IN_PHASE:
    for (ap_uint<8> count = 0; count < WORDS_PER_PHASE+images_per_phase; ++count) {
      // First word of an image
      if (wrd == 0) {
        Word wt_word_buffer[CONVOLVERS];

        // -------------------------------------------------------------------
        // Load param word
        // Each word contains CONV_W_PER_WORD weight filters, after we use
        // them all we should load the next word
        // -------------------------------------------------------------------
        LOOP_WT_WORDS:
        for (IdxType m = 0; m < CONVOLVERS; ++m) {
          /*if (wt_offset == 0)
            wt_word_buffer[m] = wt_mem[m][wt_addr];
          else
            wt_word_buffer[m] = wt_word_buffer[m] >> WT_SIZE;
          */
          wt_word_buffer[m] = wt_mem[m][wt_addr] >> ap_uint<6>(WT_SIZE*wt_offset);
          //printf("m=%d, wt_addr=%d\n", (unsigned int)m, (unsigned int)wt_addr);
        }
        if (wt_offset == CONV_W_PER_WORD-1) {
          ++wt_addr;
          wt_offset = 0;
        } else {
          ++wt_offset;
        }
        //print_wt_word(wt_word_buffer[0]);

        // -------------------------------------------------------------------
        // Load params
        // Each word contains CONV_W_PER_WORD weight filters packed into the first
        // 63 bits, the last bit is unused. Wts are stored in output-major order.
        // -------------------------------------------------------------------
        LOOP_LOAD_WTS:
        for (IdxType m = 0; m < CONVOLVERS; ++m) {
          for (ap_uint<2> kr = 0; kr < K; ++kr) {
            for (ap_uint<2> kc = 0; kc < K; ++kc)
              conv_params[m][kr][kc] = wt_word_buffer[m][kr*K+kc];
          }
        }

        DB(3, print_params(conv_params) );
      }

      // -------------------------------------------------------------------
      // Every word in an image
      // -------------------------------------------------------------------
      // Load word
      // (wrd_phase-wrd) is which wrd in the current phase, aligned to img boundary
      if (wrd != words_per_image) {
        LOOP_CONVOLVER_LOAD:
        for (IdxType m = 0; m < CONVOLVERS; ++m) {
          Word word = dmem[d_i_idx][m][p*words_per_image + wrd_phase];
          for (IdxType bank = 0; bank < CONV_BANKS; ++bank) {
            for (IdxType cc = 0; cc < CONV_COLS-2; ++cc) {
              word_buffer[m][bank][cc+1] = encode_bit(word[ap_uint<6>(bank*BANK_WIDTH+cc)]);
            }
            word_buffer[m][bank][0          ] = (bank==0)            ?
              TwoBit(0) : encode_bit(word[ap_uint<6>(bank*BANK_WIDTH-1)]);
            word_buffer[m][bank][CONV_COLS-1] = (bank==CONV_BANKS-1) ?
              TwoBit(0) : encode_bit(word[ap_uint<6>(bank*BANK_WIDTH+BANK_WIDTH)]);
          }
        }
      }

      // Compute
      LOOP_CONVOLVERS:
      for (IdxType m = 0; m < CONVOLVERS; ++m) {
        // Do the following for each word in an image
        process_word( word_buffer[m], old_word_buffer[m], lb, rb, line_buffer[m], conv_params[m],
            conv_out_buffer[m], log_width, words_per_image, wrd );
      } // CONVOLVERS

      for (IdxType m = 0; m < CONVOLVERS; ++m) {
        for (IdxType bank = 0; bank < CONV_BANKS; ++bank) {
          for (IdxType cc = 0; cc < CONV_COLS; ++cc) {
            old_word_buffer[m][bank][cc] = word_buffer[m][bank][cc];
        } }
      }

      // -------------------------------------------------------------------
      // Sum results across convolvers
      // -------------------------------------------------------------------
      for (IdxType i = 0; i < WORD_SIZE; ++i) {
        // Ignore conv results after processing the first word
        if (wrd > 0) {
          ConvSum s = 0;
          for (IdxType m = 0; m < CONVOLVERS; ++m)
            s += conv_out_buffer[m][i];
          fixed_buffer[wrd_phase-1][i] += s;
        }
      }

      // -------------------------------------------------------------------
      // Increment counters
      // -------------------------------------------------------------------
      if (wrd != words_per_image) {
        wrd++;
        wrd_phase++;
      } else {
        wrd = 0;
      }
    } // wrd_phase = 0 .. WORDS_PER_PHASE

  } // n_phases

  LOOP_ACC_PHASES:
  for (ap_uint<5> w = 0; w < words_per_image; ++w) {
    for (IdxType b = 0; b < WORD_SIZE; ++b) {
      #pragma HLS unroll
      fixed_temp[b] = fixed_buffer[w][b];
    }

    LOOP_ACC_PHASES_I:
    for (ap_uint<8> i = words_per_image; i < WORDS_PER_PHASE; i += words_per_image) {
      for (IdxType b = 0; b < WORD_SIZE; ++b) {
        fixed_temp[b] += fixed_buffer[w+i][b];
    } }

    for (IdxType b = 0; b < WORD_SIZE; ++b) {
      #pragma HLS unroll
      fixed_buffer[w][b] = fixed_temp[b];
    }
  }

  const Address bank_idx = o_index % CONVOLVERS;
  const Address bank_off = o_index / CONVOLVERS;
  const ap_uint<5> pool_width = 1 << (log_width-1);
  DB(4,
    unsigned width = 1 << log_width;
    printf ("=== conv result ===\n");
    print_mat(fixed_buffer[0], width, 8, width);
  );
  DB_PRINT(2, "  o_idx=%3d: nc=%6d\n", o_index.to_int(), nc.to_int());

  static Word outword;
  Word poolword;
  LOOP_BATCH_NORM:
  for (ap_uint<6> w = 0; w < words_per_image; ++w) {
    Word binword;
    Address o_bank_idx = bank_idx;
    Address o_bank_offset = bank_off*words_per_image + w;
    const ap_uint<6> out_offset = (w % 4) << 4;

    for (ap_uint<7> i = 0; i < WORD_SIZE; ++i) {
      binword[i] = (fixed_buffer[w][i] >= nc) ? 0 : 1;
    }

    if (norm_mode == 1) {
      outword = binword;
    }
    else if (norm_mode == 2) {
      // horizontal pooling first
      ap_int<WORD_SIZE/2> poolword_h;
      for (ap_uint<6> i = 0; i < WORD_SIZE/2; ++i) {
        poolword_h[i] = binword[2*i] & binword[2*i+1];
      }

      // vertical pooling
      for (ap_uint<6> i = 0; i < WORD_SIZE/4; ++i) {
        // source indices
        ap_uint<5> i0 = i >> (log_width-1);
        i0 = (i0 << log_width) + i(log_width-2,0);
        ap_uint<5> i1 = i0 + pool_width;
        // dest index
        ap_uint<6> d0 = out_offset + i;
        poolword[d0] = poolword_h[i0] & poolword_h[i1];
      }

      // For log_width > 3 we can just assign the word, but log_width = 3 means width = 8,
      // which means pooled width = 4, which is only 16 bits, which is less than 1 Word.
      // So each time we get here we only have 16 bits, meaning we have to accumulate four
      // of these 16-bit batches before writing a word out.
      if (log_width != LOG_BANK_WIDTH) {
        o_bank_offset /= 4;
        outword = poolword;
      } else {
        outword = outword >> WORD_SIZE/4;
        outword(63,48) = poolword(15,0);
        o_bank_idx = (o_index/4)%CONVOLVERS;
        o_bank_offset = (o_index/4)/CONVOLVERS;
      }
    }

    dmem[d_o_idx][o_bank_idx][o_bank_offset] = outword;
  }
}

void bin_conv_wrapper(

	hls::stream< Word > & Input_1,
	hls::stream< Word > & Input_2,
	hls::stream< Word > & Output_1
) {
	static unsigned int bin_conv_cnt = 0;
	Word dmem[2][CONVOLVERS][C_DMEM_WORDS];
    Word wt_mem[CONVOLVERS][C_WT_WORDS];
	Word kh_mem[KH_WORDS];

    ap_uint<1> d_i_idx_list[] =          {0,  1,  0,  0,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0  };
    ap_uint<1> d_o_idx_list[]  =         {1,  0,  1,  1,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1  };
    const Address n_inputs_list[] =      {128,128,256,256,256,256,256,256,512,512,512,512,512,512,512,512};
    const Address o_index_list[] =             {0,  0,  0,  128,0,  128,256,384,0,  64, 128,192,256,320,384,448};
    const ap_uint<2> width_mode_list[] = {2,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0  };
    const ap_uint<2> norm_mode_list[] =  {2,  1,  2,  2,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2  };
    const Address n_outputs_list[] =     {128,256,128,128,128,128,128,128,64, 64, 64,64,  64, 64, 64, 64 };

    Address o_index = o_index_list[bin_conv_cnt];
    Address n_outputs = n_outputs_list[bin_conv_cnt];
    Address kh_index = 0;



    for(unsigned int wt_mem_i=0; wt_mem_i<CONVOLVERS; wt_mem_i++)
      for(unsigned int wt_mem_j=0; wt_mem_j<C_WT_WORDS; wt_mem_j++)
      {
    	wt_mem[wt_mem_i][wt_mem_j] = Input_1.read();
    	//printf("%08x%08x,\n", (unsigned int) wt_mem[wt_mem_i][wt_mem_j](63,32), (unsigned int) wt_mem[wt_mem_i][wt_mem_j](31,0));
      }

    for(unsigned int kh_i=0; kh_i<KH_WORDS; kh_i++)
    {
    	kh_mem[kh_i] = Input_1.read();
    	//printf("%08x%08x,\n", (unsigned int) kh_mem[kh_i](63,32), (unsigned int) kh_mem[kh_i](31,0));
    }


    for(unsigned int dmem_i=0; dmem_i<2; dmem_i++)
  	  for(unsigned int dmem_j=0; dmem_j<CONVOLVERS; dmem_j++)
        for(unsigned int dmem_k=0; dmem_k<C_DMEM_WORDS; dmem_k++)
          dmem[dmem_i][dmem_j][dmem_k] = Input_2.read();


    LOOP_IMG_BATCH:
    for (IdxType i = 0; i < n_outputs; ++i) {
      // Load the batch-norm parameters for this output
      NormComp nc;
      load_kh(nc, kh_mem, kh_index);

      bin_conv(
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
    bin_conv_cnt++;
    if(bin_conv_cnt==16) bin_conv_cnt = 0;

    for(unsigned int dmem_i=0; dmem_i<2; dmem_i++)
  	  for(unsigned int dmem_j=0; dmem_j<CONVOLVERS; dmem_j++)
        for(unsigned int dmem_k=0; dmem_k<C_DMEM_WORDS; dmem_k++)
          Output_1.write(dmem[dmem_i][dmem_j][dmem_k]);

}
// -----------------------------------------------------------------------
// Module to do the first conv layer
// -----------------------------------------------------------------------
void fp_conv(
	hls::stream< Word > & Input_1,//Word wt_mem[CONVOLVERS][C_WT_WORDS],
	hls::stream< Word > & Input_2,
	hls::stream< Word > & Output_1
) {
#pragma HLS INTERFACE ap_hs port=Input_1
#pragma HLS INTERFACE ap_hs port=Input_2
#pragma HLS INTERFACE ap_hs port=Output_1

  const unsigned M = 3;
  const unsigned S = 32;
  const unsigned OUTWORDS = 16; // words per output image
  Word kh_mem[KH_WORDS];
  C1InputType win[M][K][K];
  C1InputType lbuf[M][K-1][S];
  Word outwords[OUTWORDS];
  WtType wtbuf[M];

  Address wt_offset = 0;
  ap_uint<3> wt_addr = 0;

  ap_uint<1> d_i_idx = 1;
  ap_uint<1> d_o_idx = 0;
  const Address kh_index = 0;
  const Address o_index = 0;
  const unsigned N = 128;
  Word dmem[2][CONVOLVERS][C_DMEM_WORDS];

  for(int in_data_cnt=0; in_data_cnt<1024; in_data_cnt++) {
    dmem[1][0][in_data_cnt] = Input_2.read();
  }

  for(int kh_i=0; kh_i<KH_WORDS; kh_i++)
  {
  	kh_mem[kh_i] = Input_1.read();
  	//printf("%08x%08x,\n", (unsigned int) kh_mem[kh_i](63,32), (unsigned int) kh_mem[kh_i](31,0));
  }

  // Parallelized across m, better for HLS
  LOOP_FP_CONV_O:
  for (IdxType n = 0; n < N; ++n) {

    // clear linebuffers for each new output map
    LOOP_RESET_LINEBUFFERS:
    for (IdxType m = 0; m < M; ++m) {
      PROLOG_COLS: for (IdxType c = 0; c < S; ++c) {
        PROLOG_ROWS: for (IdxType r = 0; r < K/2; ++r) {
          for (IdxType lr = 0; lr < K-2; ++lr) {
            lbuf[m][lr][c] = lbuf[m][lr+1][c];
          }
          lbuf[m][K-2][c] = 0;
      } }
    }

    // The weights for the 1st conv layer are just laid out
    // linearly across wt_mem, 3 weights per 64-bit word
    DB_PRINT(3, "n = %u\n", n.to_int());
    Word wt_word =  Input_1.read();//wt_mem[n % CONVOLVERS][n / CONVOLVERS];
    //printf("%08x%08x,\n", (unsigned int) wt_word(63,32), (unsigned int) wt_word(31,0));
    LOOP_LOAD_WTS:
    for (ap_uint<2> m = 0; m < M; ++m) {
      wtbuf[m] = wt_word((m+1)*WT_SIZE-1, m*WT_SIZE);
      DB(3, print_wt(wtbuf[m]));
      DB(3, printf("--\n"));
    }

    // load batch norm params
    C1Comp nc;

    load_kh(nc, kh_mem, (kh_index+n));
    //printf ("  n=%3d, nc=%6.3f\n", n.to_int(), nc.to_float());

    // begin convolution
    LOOP_CONV_ROWS: for (IdxType r = 0; r < S+1; ++r) {
      LOOP_CONV_COLS: for (IdxType c = 0; c < S+1; ++c) {
        // load input word
        Word inword = 0;
        if (r < S && c < S) {
          const Address addr = r*S + c;
          inword = dmem[d_i_idx][addr/C_DMEM_WORDS][addr%C_DMEM_WORDS];

        }

        for (ap_uint<2> m = 0; m < M; ++m) {
          // load data: the value of pix is either the pixel at [r,c]
          // 0 -> +1, -1 -> -1
          // or -> 0 for padding around the boundaries
          C1InputType pix;
          const unsigned W = pix.length();
          pix(W-1,0) = inword(W-1+m*W, m*W);

          // window: shift left, leaving rightmost col for new data
          for (IdxType wr = 0; wr < K; ++wr) {
            for (IdxType wc = 0; wc < K-1; ++wc) {
              win[m][wr][wc] = win[m][wr][wc+1];
          } }

          // window: fill top K-1 pixels of rightmost column from lbuf
          for (IdxType wr = 0; wr < K-1; ++wr) {
            C1InputType val = (c != S) ? lbuf[m][wr][c] : C1InputType(0);
            win[m][wr][K-1] = val;
          }

          // window: fill bottom right with new input pixel
          win[m][K-1][K-1] = pix;

          // lbuf: shift up column c
          if (c != S) {
            for (IdxType lr = 0; lr < K-2; ++lr) {
              lbuf[m][lr][c] = lbuf[m][lr+1][c];
            }
            lbuf[m][K-2][c] = pix;
          }
        } // m

        // only perform the conv and store if legal position
        if (r > 0 && c > 0) {
          C1ConvType res = 0;
          for (ap_uint<2> m = 0; m < M; ++m) {
            for (ap_uint<2> wr = 0; wr < K; ++wr) {
              for (ap_uint<2> wc = 0; wc < K; ++wc) {
                const C1InputType& pix = win[m][wr][wc];
                const Bit& b = wtbuf[m][8-(wr*K+wc)];
                res += (b==0) ? pix : (C1InputType)(-pix);
            } }
          }

          // perform normalization right here
          outwords[(r-1)/2][((r-1)%2)*S + (c-1)] =
            (res >= nc) ? Bit(0) : Bit(-1);
        }

      } // CONV_COLS
    } // CONV_ROWS

    // Here i is the word offset within the outwords buffer
    LOOP_OUTPUT:
    for (IdxType i = 0; i < OUTWORDS; ++i) {
      Address img_idx = o_index+n;
      Address bank_idx = img_idx % CONVOLVERS;
      Address bank_off = img_idx / CONVOLVERS;
      dmem[d_o_idx][bank_idx][bank_off*OUTWORDS + i] = outwords[i];
      //printf("d_o_idx = %d, bank_idx = %d, bank_off*OUTWORDS + i = %d\n",
      //        		  (unsigned int)(d_o_idx),
      //				  (unsigned int)(bank_idx),
      //				  (unsigned int)(bank_off*OUTWORDS + i));
    }
  } // n
  for(unsigned int dmem_i=0; dmem_i<2; dmem_i++)
	for(unsigned int dmem_j=0; dmem_j<CONVOLVERS; dmem_j++)
      for(unsigned int dmem_k=0; dmem_k<C_DMEM_WORDS; dmem_k++)
        Output_1.write(dmem[dmem_i][dmem_j][dmem_k]);

}


void bin_dense(
    const Word wt_mem[CONVOLVERS][C_WT_WORDS],
    const Word kh_mem[KH_WORDS],
    Word dmem[2][CONVOLVERS][64]
) {
  static char ctrl_i = 0;
  ap_uint<2> layer_type;
  ap_uint<1> d_i_idx;
  ap_uint<1> d_o_idx;
  Address o_index;
  unsigned n_inputs;
  unsigned n_outputs;
  ap_uint<32> tmp1_list[] = {0x21000000,0x21002000,0x21004000,0x21006000,0x21008000,0x2100a000,0x2100c000,0x2100e000,
		                      0x21010000,0x21012000,0x21014000,0x21016000,0x21018000,0x2101a000,0x2101c000,0x2101e000,
						   	  0x21020000,0x21022000,0x21024000,0x21026000,0x21028000,0x2102a000,0x2102c000,0x2102e000,
							  0x21030000,0x21032000,0x21034000,0x21036000,0x21038000,0x2103a000,0x2103c000,0x2103e000,
							  0x20100000,0x20110000,0x20120000,0x20130000,0x31000000};


  ap_uint<32> tmp2_list[] = {0x20000020,0x20000020,0x20000020,0x20000020,0x20000020,0x20000020,0x20000020,0x20000020,
		                      0x20000020,0x20000020,0x20000020,0x20000020,0x20000020,0x20000020,0x20000020,0x20000020,
							  0x20000020,0x20000020,0x20000020,0x20000020,0x20000020,0x20000020,0x20000020,0x20000020,
							  0x20000020,0x20000020,0x20000020,0x20000020,0x20000020,0x20000020,0x20000020,0x20000020,
							  0x04000100,0x04000100,0x04000100,0x04000100,0x0400000a};

  layer_type  = tmp1_list[ctrl_i](31,28);
  d_i_idx     = tmp1_list[ctrl_i](27,24);
  d_o_idx     = tmp1_list[ctrl_i](23,20);
  o_index     = tmp1_list[ctrl_i](19, 8);

  n_inputs    = tmp2_list[ctrl_i](31,16);
  n_outputs   = tmp2_list[ctrl_i](15, 0);

  //assert(n_outputs % WORD_SIZE == 0);
  assert(layer_type == LAYER_DENSE || n_outputs == 10);
  assert(n_inputs/WORD_SIZE % CONVOLVERS == 0);


  ctrl_i++;
  if(ctrl_i==37) ctrl_i=0;

  DenseSum sum_m[CONVOLVERS];
  // for last layer
  DenseNorm best_out = -1024;
  ap_int<8> prediction = -1;

  // read words from dmem and the wt store, dot them
  // o is the output bit, i is the input bit
  LOOP_DENSE_O:
  for (Address o = 0; o < n_outputs; ++o) {
    const Address o_addr = (o_index+o)/WORD_SIZE;
    const ap_uint<6> o_offset = (o_index+o) % WORD_SIZE;
    Word o_word = dmem[d_o_idx][o_addr%CONVOLVERS][o_addr/CONVOLVERS];
    //printf("i,%d, j,%d, k,%d\n", (unsigned int) d_o_idx, (unsigned int) (o_addr%CONVOLVERS), (unsigned int)(o_addr/CONVOLVERS));

    DenseSum sum = 0;

    LOOP_DENSE_I:
    for (Address i = 0; i < n_inputs; i+=CONVOLVERS*WORD_SIZE) {
      const Address wt_addr = (o*n_inputs+i) / WORD_SIZE;

      for (IdxType j = 0; j < CONVOLVERS; ++j) {
        // in_wrd addr = [(i/WORD_SIZE+j) % CONVOLVERS][(i/WORD_SIZE+j) / CONVOLVERS]
        // wt_wrd addr = [wt_addr % CONVOLVERS][wt_addr / CONVOLVERS]
        const Word in_wrd = dmem[d_i_idx][j][i/WORD_SIZE/CONVOLVERS];
        //printf("i,%d, j,%d, k,%d\n", (unsigned int) d_i_idx, (unsigned int) (j), (unsigned int)(i/WORD_SIZE/CONVOLVERS));
        const Word wt_wrd = wt_mem[j][wt_addr / CONVOLVERS];


        Word x = wt_wrd ^ in_wrd;

        // count_set bit for 64 bits, returns 2*cnt
        x -= (x >> 1) & m1;
        x = (x & m2) + ((x >> 2) & m2);
        x = (x + (x >> 4)) & m4;
        x += x >> 8;
        x += x >> 16;
        x += x >> 32;
        x = x & 0x7f;

        sum_m[j] = WORD_SIZE - (DenseSum)(x<<1);
      }

      for (IdxType j = 0; j < CONVOLVERS; ++j)
        sum += sum_m[j];
    } // n_inputs

    // not last layer -> biniarize,
    // otherwise just store the value as a 64bit word
    if (layer_type == LAYER_DENSE) {
      Address kh_addr = o / KH_PER_WORD;
      Word kh_word = kh_mem[kh_addr];

      NormComp nc;
      IdxType kh_off = o % KH_PER_WORD;
      if (kh_off == 0)
        nc(15,0) = kh_word(15, 0);
      else if (kh_off == 1)
        nc(15,0) = kh_word(31,16);
      else if (kh_off == 2)
        nc(15,0) = kh_word(47,32);
      else
        nc(15,0) = kh_word(63,48);

      o_word[o_offset] = (sum >= nc) ? 0 : 1;
    } else {
      Address kh_addr = o / (const unsigned)2;
      Word kh_word = kh_mem[kh_addr];

      KType ki;  HType hi;
      IdxType kh_off = o % 2;
      if (kh_off == 0) {
        ki(15,0) = kh_word(15, 0);
        hi(15,0) = kh_word(31,16);
      } else {
        ki(15,0) = kh_word(47,32);
        hi(15,0) = kh_word(63,48);
      }

      //printf (" >> %d * %f + %f\n", sum.to_int(), ki.to_float(), hi.to_float());
      ap_fixed<20,10> out = ap_fixed<20,10>(sum)*ki + hi;

      if (o == 0 || out > best_out) {
        prediction = o;
        best_out = out;
      }
    }

    dmem[d_o_idx][o_addr%CONVOLVERS][o_addr/CONVOLVERS] = o_word;
    //printf("i,%d, j,%d, k,%d\n", (unsigned int) d_o_idx, (unsigned int) (o_addr%CONVOLVERS), (unsigned int)(o_addr/CONVOLVERS));
  } // n_outputs

  // Here we are using o_index as a bit index, not a word index!
  if (layer_type == LAYER_LAST) {
    Word o_word;
    o_word(7,0) = prediction(7,0);
    o_word(WORD_SIZE-1, 8) = 0;
    dmem[d_o_idx][0][0] = o_word;
    //printf("i,%d, j,%d, k,%d\n", (unsigned int) d_o_idx, (unsigned int) (0), (unsigned int)(0));
  }
}

void bin_dense_wrapper(
	hls::stream< Word > & Input_1,
	hls::stream< Word > & Input_2,
	hls::stream< Word > & Output_1
) {
	Word dmem[2][CONVOLVERS][64];
	Word wt_mem[CONVOLVERS][C_WT_WORDS];
	Word kh_mem[KH_WORDS];

    for(unsigned int wt_mem_i=0; wt_mem_i<CONVOLVERS; wt_mem_i++)
      for(unsigned int wt_mem_j=0; wt_mem_j<C_WT_WORDS; wt_mem_j++)
      {
    	wt_mem[wt_mem_i][wt_mem_j] = Input_1.read();
    	//printf("%08x%08x,\n", (unsigned int) wt_mem[wt_mem_i][wt_mem_j](63,32), (unsigned int) wt_mem[wt_mem_i][wt_mem_j](31,0));
      }

    for(unsigned int kh_i=0; kh_i<KH_WORDS; kh_i++)
    {
    	kh_mem[kh_i] = Input_1.read();
    	//printf("%08x%08x,\n", (unsigned int) kh_mem[kh_i](63,32), (unsigned int) kh_mem[kh_i](31,0));
    }

	for(int i=0; i<2; i++)
	  for(int j=0; j<2; j++)
		for(int k=0; k<64; k++){
			dmem[i][j][k] = Input_2.read();
		}

	bin_dense(
		wt_mem,
	    kh_mem,
	    dmem
	);

	for(int i=0; i<2; i++)
	  for(int j=0; j<2; j++)
		for(int k=0; k<64; k++){
			Output_1.write(dmem[i][j][k]);
		}
}

void data_gen(
	int image_num,
	hls::stream< Word > & Output_1
) {
  #include "data.h"
  for(int i=0; i<image_num*1024; i++)
  {
	  Output_1.write(data_in[i]);
  }
}

void fp_conv_gen(
	hls::stream< Word > & Output_1
) {
#include "fp_conv_par.h"
	for(int kh_i=0; kh_i<192; kh_i++)
  {
	  Output_1.write(fp_conv_wt[kh_i]);
  }
}


// -----------------------------------------------------------------------
// Accelerator top module
// -----------------------------------------------------------------------
void top(
	hls::stream< Word > & Input_1,
    Word wt_i[WT_WORDS],
    Word kh_i[KH_WORDS],
    Word dmem_i[DMEM_WORDS],
    Word dmem_o[DMEM_O_WORDS],
    const Address    n_inputs,
    const Address    n_outputs,
    const Address    input_words,
    const Address    output_words,
    const ap_uint<3> layer_mode,  // [0]='new layer', [2:1]='conv1,conv,dense,last'
    const ap_uint<1> dmem_mode,   // 0 means dmem[0] is input
    const ap_uint<2> width_mode,  // 0=8'b, 1=16'b, 2=32'b
    const ap_uint<2> norm_mode    // 0='do nothing', 1='do norm', 2='do pool'
) {
	hls::stream< Word > fp_conv_in1("fp_conv_in1");
	hls::stream< Word > fp_conv_in2("fp_conv_in2");
	hls::stream< Word > fp_conv_out1("fp_conv_out1");
	hls::stream< Word > bin_conv_in1("bin_conv_in1");
	hls::stream< Word > bin_conv_in2("bin_conv_in2");
	hls::stream< Word > bin_conv_out1("bin_conv_out1");
	hls::stream< Word > bin_dense_in1("bin_dense_in1");
	hls::stream< Word > bin_dense_in2("bin_dense_in2");
	hls::stream< Word > bin_dense_out1("bin_dense_out1");

	static Word dmem[2][CONVOLVERS][C_DMEM_WORDS];


	static int bin_conv_cnt = 0;
	static int layer_cnt = 0;

	if(layer_cnt == 0) {


		fp_conv_gen(fp_conv_in1);

	    fp_conv(
	    	fp_conv_in1,//wt_mem,
			Input_1,
			fp_conv_out1
	    );

	    for(unsigned int dmem_i=0; dmem_i<2; dmem_i++)
	  	  for(unsigned int dmem_j=0; dmem_j<CONVOLVERS; dmem_j++)
	        for(unsigned int dmem_k=0; dmem_k<C_DMEM_WORDS; dmem_k++)
	          dmem[dmem_i][dmem_j][dmem_k] = fp_conv_out1.read();
	}



	if(layer_cnt >=1 && layer_cnt <= 16) {

		for(unsigned int dmem_i=0; dmem_i<2; dmem_i++)
		  for(unsigned int dmem_j=0; dmem_j<CONVOLVERS; dmem_j++)
			for(unsigned int dmem_k=0; dmem_k<C_DMEM_WORDS; dmem_k++)
				bin_conv_in2.write(dmem[dmem_i][dmem_j][dmem_k]);




		for(unsigned int wt_mem_i=0; wt_mem_i<CONVOLVERS; wt_mem_i++)
		  for(unsigned int wt_mem_j=0; wt_mem_j<C_WT_WORDS; wt_mem_j++)
		  {
			//bin_conv_in1.write(wt_mem[wt_mem_i][wt_mem_j]);
			bin_conv_in1.write(bin_conv_wt[bin_conv_cnt]);
			bin_conv_cnt++;
		  }

		for(unsigned int kh_i=0; kh_i<KH_WORDS; kh_i++)
		{
			//bin_conv_in1.write(kh_mem[kh_i]);
			bin_conv_in1.write(bin_conv_wt[bin_conv_cnt]);
			bin_conv_cnt++;
		}

		if(bin_conv_cnt == 75936) bin_conv_cnt = 0;
		bin_conv_wrapper(
			bin_conv_in1,
			bin_conv_in2,
			bin_conv_out1
		);

		for(unsigned int dmem_i=0; dmem_i<2; dmem_i++)
		  for(unsigned int dmem_j=0; dmem_j<CONVOLVERS; dmem_j++)
			for(unsigned int dmem_k=0; dmem_k<C_DMEM_WORDS; dmem_k++)
			  dmem[dmem_i][dmem_j][dmem_k] = bin_conv_out1.read();
	}

	if(layer_cnt >=17) {

		for(int i=0; i<2; i++)
		  for(int j=0; j<2; j++)
			for(int k=0; k<64; k++){
				bin_dense_in2.write(dmem[i][j][k]);
			}

		for(unsigned int wt_mem_i=0; wt_mem_i<CONVOLVERS; wt_mem_i++)
		  for(unsigned int wt_mem_j=0; wt_mem_j<C_WT_WORDS; wt_mem_j++)
		  {
			//bin_dense_in1.write(wt_mem[wt_mem_i][wt_mem_j]);
			bin_dense_in1.write(bin_dense_wt[bin_dense_cnt]);
			bin_dense_cnt++;

		  }

		for(unsigned int kh_i=0; kh_i<KH_WORDS; kh_i++)
		{
			//bin_dense_in1.write(kh_mem[kh_i]);
			bin_dense_in1.write(bin_dense_wt[bin_dense_cnt]);
			bin_dense_cnt++;
		}

		if (bin_dense_cnt == 175602) bin_dense_cnt = 0;
		//printf("bin_dense\n");
		  bin_dense_wrapper(
			bin_dense_in1,
			bin_dense_in2,
			bin_dense_out1
		);

		for(int i=0; i<2; i++)
		  for(int j=0; j<2; j++)
			for(int k=0; k<64; k++){
				dmem[i][j][k] = bin_dense_out1.read();
			}

	} // layer_type

	if(layer_cnt ==53) {
      dmem_o[0] = dmem[0][0][0];
	}
	layer_cnt++;
	if(layer_cnt == 54) layer_cnt = 0;

}
