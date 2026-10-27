/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-10-27T17:49:44+0700
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "network.h"
#include "network_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_network
 
#undef AI_NETWORK_MODEL_SIGNATURE
#define AI_NETWORK_MODEL_SIGNATURE     "0x402dab2e5f51bbfa2ba24fe36a0d09fc"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "2026-10-27T17:49:44+0700"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_NETWORK_N_BATCHES
#define AI_NETWORK_N_BATCHES         (1)

static ai_ptr g_network_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_network_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  serving_default_input0_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 128, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_39_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  gemm_45_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  nl_45_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  gemm_46_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  nl_47_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_29_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  gemm_35_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  nl_35_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  gemm_36_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  nl_37_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_19_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  gemm_25_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  nl_25_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  gemm_26_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  nl_27_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_9_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  gemm_15_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  nl_15_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  gemm_16_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  nl_17_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  gemm_6_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  nl_6_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  gemm_7_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  nl_8_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_18_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_28_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_38_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7, AI_STATIC)

/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_48_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7, AI_STATIC)

/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_49_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 7, AI_STATIC)

/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_39_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 144, AI_STATIC)

/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_39_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  gemm_45_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  gemm_45_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#35 */
AI_ARRAY_OBJ_DECLARE(
  gemm_46_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 224, AI_STATIC)

/* Array#36 */
AI_ARRAY_OBJ_DECLARE(
  gemm_46_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7, AI_STATIC)

/* Array#37 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_29_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 144, AI_STATIC)

/* Array#38 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_29_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#39 */
AI_ARRAY_OBJ_DECLARE(
  gemm_35_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#40 */
AI_ARRAY_OBJ_DECLARE(
  gemm_35_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#41 */
AI_ARRAY_OBJ_DECLARE(
  gemm_36_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 224, AI_STATIC)

/* Array#42 */
AI_ARRAY_OBJ_DECLARE(
  gemm_36_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7, AI_STATIC)

/* Array#43 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_19_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 144, AI_STATIC)

/* Array#44 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_19_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#45 */
AI_ARRAY_OBJ_DECLARE(
  gemm_25_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#46 */
AI_ARRAY_OBJ_DECLARE(
  gemm_25_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#47 */
AI_ARRAY_OBJ_DECLARE(
  gemm_26_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 224, AI_STATIC)

/* Array#48 */
AI_ARRAY_OBJ_DECLARE(
  gemm_26_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7, AI_STATIC)

/* Array#49 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_9_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 144, AI_STATIC)

/* Array#50 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_9_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#51 */
AI_ARRAY_OBJ_DECLARE(
  gemm_15_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#52 */
AI_ARRAY_OBJ_DECLARE(
  gemm_15_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#53 */
AI_ARRAY_OBJ_DECLARE(
  gemm_16_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 224, AI_STATIC)

/* Array#54 */
AI_ARRAY_OBJ_DECLARE(
  gemm_16_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7, AI_STATIC)

/* Array#55 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 144, AI_STATIC)

/* Array#56 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#57 */
AI_ARRAY_OBJ_DECLARE(
  gemm_6_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#58 */
AI_ARRAY_OBJ_DECLARE(
  gemm_6_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#59 */
AI_ARRAY_OBJ_DECLARE(
  gemm_7_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 224, AI_STATIC)

/* Array#60 */
AI_ARRAY_OBJ_DECLARE(
  gemm_7_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7, AI_STATIC)

/* Array#61 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_49_scale_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7, AI_STATIC)

/* Array#62 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_49_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7, AI_STATIC)

/* Array#63 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_39_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 18, AI_STATIC)

/* Array#64 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_39_scratch1_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#65 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_29_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 18, AI_STATIC)

/* Array#66 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_29_scratch1_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#67 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_19_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 18, AI_STATIC)

/* Array#68 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_19_scratch1_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#69 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_9_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 18, AI_STATIC)

/* Array#70 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_9_scratch1_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#71 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 18, AI_STATIC)

/* Array#72 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_0_scratch1_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_0_bias, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &conv2d_0_bias_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_0_output, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 4, 4), AI_STRIDE_INIT(4, 4, 4, 32, 128),
  1, &conv2d_0_output_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_0_output0, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv2d_0_output_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_0_scratch0, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 3, 3), AI_STRIDE_INIT(4, 4, 4, 8, 24),
  1, &conv2d_0_scratch0_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_0_scratch1, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 8, 2), AI_STRIDE_INIT(4, 4, 4, 32, 256),
  1, &conv2d_0_scratch1_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_0_weights, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 2, 3, 3, 8), AI_STRIDE_INIT(4, 4, 8, 64, 192),
  1, &conv2d_0_weights_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_19_bias, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &conv2d_19_bias_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_19_output, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 4, 4), AI_STRIDE_INIT(4, 4, 4, 32, 128),
  1, &conv2d_19_output_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_19_output0, AI_STATIC,
  8, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv2d_19_output_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_19_scratch0, AI_STATIC,
  9, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 3, 3), AI_STRIDE_INIT(4, 4, 4, 8, 24),
  1, &conv2d_19_scratch0_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_19_scratch1, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 8, 2), AI_STRIDE_INIT(4, 4, 4, 32, 256),
  1, &conv2d_19_scratch1_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_19_weights, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 2, 3, 3, 8), AI_STRIDE_INIT(4, 4, 8, 64, 192),
  1, &conv2d_19_weights_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_29_bias, AI_STATIC,
  12, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &conv2d_29_bias_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_29_output, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 4, 4), AI_STRIDE_INIT(4, 4, 4, 32, 128),
  1, &conv2d_29_output_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_29_output0, AI_STATIC,
  14, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv2d_29_output_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_29_scratch0, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 3, 3), AI_STRIDE_INIT(4, 4, 4, 8, 24),
  1, &conv2d_29_scratch0_array, NULL)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_29_scratch1, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 8, 2), AI_STRIDE_INIT(4, 4, 4, 32, 256),
  1, &conv2d_29_scratch1_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_29_weights, AI_STATIC,
  17, 0x0,
  AI_SHAPE_INIT(4, 2, 3, 3, 8), AI_STRIDE_INIT(4, 4, 8, 64, 192),
  1, &conv2d_29_weights_array, NULL)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_39_bias, AI_STATIC,
  18, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &conv2d_39_bias_array, NULL)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_39_output, AI_STATIC,
  19, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 4, 4), AI_STRIDE_INIT(4, 4, 4, 32, 128),
  1, &conv2d_39_output_array, NULL)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_39_output0, AI_STATIC,
  20, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv2d_39_output_array, NULL)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_39_scratch0, AI_STATIC,
  21, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 3, 3), AI_STRIDE_INIT(4, 4, 4, 8, 24),
  1, &conv2d_39_scratch0_array, NULL)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_39_scratch1, AI_STATIC,
  22, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 8, 2), AI_STRIDE_INIT(4, 4, 4, 32, 256),
  1, &conv2d_39_scratch1_array, NULL)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_39_weights, AI_STATIC,
  23, 0x0,
  AI_SHAPE_INIT(4, 2, 3, 3, 8), AI_STRIDE_INIT(4, 4, 8, 64, 192),
  1, &conv2d_39_weights_array, NULL)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_9_bias, AI_STATIC,
  24, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &conv2d_9_bias_array, NULL)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_9_output, AI_STATIC,
  25, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 4, 4), AI_STRIDE_INIT(4, 4, 4, 32, 128),
  1, &conv2d_9_output_array, NULL)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_9_output0, AI_STATIC,
  26, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv2d_9_output_array, NULL)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_9_scratch0, AI_STATIC,
  27, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 3, 3), AI_STRIDE_INIT(4, 4, 4, 8, 24),
  1, &conv2d_9_scratch0_array, NULL)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_9_scratch1, AI_STATIC,
  28, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 8, 2), AI_STRIDE_INIT(4, 4, 4, 32, 256),
  1, &conv2d_9_scratch1_array, NULL)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_9_weights, AI_STATIC,
  29, 0x0,
  AI_SHAPE_INIT(4, 2, 3, 3, 8), AI_STRIDE_INIT(4, 4, 8, 64, 192),
  1, &conv2d_9_weights_array, NULL)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_18_output, AI_STATIC,
  30, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &eltwise_18_output_array, NULL)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_28_output, AI_STATIC,
  31, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &eltwise_28_output_array, NULL)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_38_output, AI_STATIC,
  32, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &eltwise_38_output_array, NULL)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_48_output, AI_STATIC,
  33, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &eltwise_48_output_array, NULL)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_49_bias, AI_STATIC,
  34, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &eltwise_49_bias_array, NULL)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_49_output, AI_STATIC,
  35, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &eltwise_49_output_array, NULL)

/* Tensor #36 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_49_scale, AI_STATIC,
  36, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &eltwise_49_scale_array, NULL)

/* Tensor #37 */
AI_TENSOR_OBJ_DECLARE(
  gemm_15_bias, AI_STATIC,
  37, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &gemm_15_bias_array, NULL)

/* Tensor #38 */
AI_TENSOR_OBJ_DECLARE(
  gemm_15_output, AI_STATIC,
  38, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &gemm_15_output_array, NULL)

/* Tensor #39 */
AI_TENSOR_OBJ_DECLARE(
  gemm_15_weights, AI_STATIC,
  39, 0x0,
  AI_SHAPE_INIT(4, 128, 32, 1, 1), AI_STRIDE_INIT(4, 4, 512, 16384, 16384),
  1, &gemm_15_weights_array, NULL)

/* Tensor #40 */
AI_TENSOR_OBJ_DECLARE(
  gemm_16_bias, AI_STATIC,
  40, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &gemm_16_bias_array, NULL)

/* Tensor #41 */
AI_TENSOR_OBJ_DECLARE(
  gemm_16_output, AI_STATIC,
  41, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &gemm_16_output_array, NULL)

/* Tensor #42 */
AI_TENSOR_OBJ_DECLARE(
  gemm_16_weights, AI_STATIC,
  42, 0x0,
  AI_SHAPE_INIT(4, 32, 7, 1, 1), AI_STRIDE_INIT(4, 4, 128, 896, 896),
  1, &gemm_16_weights_array, NULL)

/* Tensor #43 */
AI_TENSOR_OBJ_DECLARE(
  gemm_25_bias, AI_STATIC,
  43, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &gemm_25_bias_array, NULL)

/* Tensor #44 */
AI_TENSOR_OBJ_DECLARE(
  gemm_25_output, AI_STATIC,
  44, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &gemm_25_output_array, NULL)

/* Tensor #45 */
AI_TENSOR_OBJ_DECLARE(
  gemm_25_weights, AI_STATIC,
  45, 0x0,
  AI_SHAPE_INIT(4, 128, 32, 1, 1), AI_STRIDE_INIT(4, 4, 512, 16384, 16384),
  1, &gemm_25_weights_array, NULL)

/* Tensor #46 */
AI_TENSOR_OBJ_DECLARE(
  gemm_26_bias, AI_STATIC,
  46, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &gemm_26_bias_array, NULL)

/* Tensor #47 */
AI_TENSOR_OBJ_DECLARE(
  gemm_26_output, AI_STATIC,
  47, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &gemm_26_output_array, NULL)

/* Tensor #48 */
AI_TENSOR_OBJ_DECLARE(
  gemm_26_weights, AI_STATIC,
  48, 0x0,
  AI_SHAPE_INIT(4, 32, 7, 1, 1), AI_STRIDE_INIT(4, 4, 128, 896, 896),
  1, &gemm_26_weights_array, NULL)

/* Tensor #49 */
AI_TENSOR_OBJ_DECLARE(
  gemm_35_bias, AI_STATIC,
  49, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &gemm_35_bias_array, NULL)

/* Tensor #50 */
AI_TENSOR_OBJ_DECLARE(
  gemm_35_output, AI_STATIC,
  50, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &gemm_35_output_array, NULL)

/* Tensor #51 */
AI_TENSOR_OBJ_DECLARE(
  gemm_35_weights, AI_STATIC,
  51, 0x0,
  AI_SHAPE_INIT(4, 128, 32, 1, 1), AI_STRIDE_INIT(4, 4, 512, 16384, 16384),
  1, &gemm_35_weights_array, NULL)

/* Tensor #52 */
AI_TENSOR_OBJ_DECLARE(
  gemm_36_bias, AI_STATIC,
  52, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &gemm_36_bias_array, NULL)

/* Tensor #53 */
AI_TENSOR_OBJ_DECLARE(
  gemm_36_output, AI_STATIC,
  53, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &gemm_36_output_array, NULL)

/* Tensor #54 */
AI_TENSOR_OBJ_DECLARE(
  gemm_36_weights, AI_STATIC,
  54, 0x0,
  AI_SHAPE_INIT(4, 32, 7, 1, 1), AI_STRIDE_INIT(4, 4, 128, 896, 896),
  1, &gemm_36_weights_array, NULL)

/* Tensor #55 */
AI_TENSOR_OBJ_DECLARE(
  gemm_45_bias, AI_STATIC,
  55, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &gemm_45_bias_array, NULL)

/* Tensor #56 */
AI_TENSOR_OBJ_DECLARE(
  gemm_45_output, AI_STATIC,
  56, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &gemm_45_output_array, NULL)

/* Tensor #57 */
AI_TENSOR_OBJ_DECLARE(
  gemm_45_weights, AI_STATIC,
  57, 0x0,
  AI_SHAPE_INIT(4, 128, 32, 1, 1), AI_STRIDE_INIT(4, 4, 512, 16384, 16384),
  1, &gemm_45_weights_array, NULL)

/* Tensor #58 */
AI_TENSOR_OBJ_DECLARE(
  gemm_46_bias, AI_STATIC,
  58, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &gemm_46_bias_array, NULL)

/* Tensor #59 */
AI_TENSOR_OBJ_DECLARE(
  gemm_46_output, AI_STATIC,
  59, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &gemm_46_output_array, NULL)

/* Tensor #60 */
AI_TENSOR_OBJ_DECLARE(
  gemm_46_weights, AI_STATIC,
  60, 0x0,
  AI_SHAPE_INIT(4, 32, 7, 1, 1), AI_STRIDE_INIT(4, 4, 128, 896, 896),
  1, &gemm_46_weights_array, NULL)

/* Tensor #61 */
AI_TENSOR_OBJ_DECLARE(
  gemm_6_bias, AI_STATIC,
  61, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &gemm_6_bias_array, NULL)

/* Tensor #62 */
AI_TENSOR_OBJ_DECLARE(
  gemm_6_output, AI_STATIC,
  62, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &gemm_6_output_array, NULL)

/* Tensor #63 */
AI_TENSOR_OBJ_DECLARE(
  gemm_6_weights, AI_STATIC,
  63, 0x0,
  AI_SHAPE_INIT(4, 128, 32, 1, 1), AI_STRIDE_INIT(4, 4, 512, 16384, 16384),
  1, &gemm_6_weights_array, NULL)

/* Tensor #64 */
AI_TENSOR_OBJ_DECLARE(
  gemm_7_bias, AI_STATIC,
  64, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &gemm_7_bias_array, NULL)

/* Tensor #65 */
AI_TENSOR_OBJ_DECLARE(
  gemm_7_output, AI_STATIC,
  65, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &gemm_7_output_array, NULL)

/* Tensor #66 */
AI_TENSOR_OBJ_DECLARE(
  gemm_7_weights, AI_STATIC,
  66, 0x0,
  AI_SHAPE_INIT(4, 32, 7, 1, 1), AI_STRIDE_INIT(4, 4, 128, 896, 896),
  1, &gemm_7_weights_array, NULL)

/* Tensor #67 */
AI_TENSOR_OBJ_DECLARE(
  nl_15_nl_output, AI_STATIC,
  67, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &nl_15_nl_output_array, NULL)

/* Tensor #68 */
AI_TENSOR_OBJ_DECLARE(
  nl_17_output, AI_STATIC,
  68, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &nl_17_output_array, NULL)

/* Tensor #69 */
AI_TENSOR_OBJ_DECLARE(
  nl_25_nl_output, AI_STATIC,
  69, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &nl_25_nl_output_array, NULL)

/* Tensor #70 */
AI_TENSOR_OBJ_DECLARE(
  nl_27_output, AI_STATIC,
  70, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &nl_27_output_array, NULL)

/* Tensor #71 */
AI_TENSOR_OBJ_DECLARE(
  nl_35_nl_output, AI_STATIC,
  71, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &nl_35_nl_output_array, NULL)

/* Tensor #72 */
AI_TENSOR_OBJ_DECLARE(
  nl_37_output, AI_STATIC,
  72, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &nl_37_output_array, NULL)

/* Tensor #73 */
AI_TENSOR_OBJ_DECLARE(
  nl_45_nl_output, AI_STATIC,
  73, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &nl_45_nl_output_array, NULL)

/* Tensor #74 */
AI_TENSOR_OBJ_DECLARE(
  nl_47_output, AI_STATIC,
  74, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &nl_47_output_array, NULL)

/* Tensor #75 */
AI_TENSOR_OBJ_DECLARE(
  nl_6_nl_output, AI_STATIC,
  75, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &nl_6_nl_output_array, NULL)

/* Tensor #76 */
AI_TENSOR_OBJ_DECLARE(
  nl_8_output, AI_STATIC,
  76, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &nl_8_output_array, NULL)

/* Tensor #77 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_input0_output, AI_STATIC,
  77, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 8, 8), AI_STRIDE_INIT(4, 4, 4, 8, 64),
  1, &serving_default_input0_output_array, NULL)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_49_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_48_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_49_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_49_scale, &eltwise_49_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_49_layer, 49,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &eltwise_49_chain,
  NULL, &eltwise_49_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_48_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_38_output, &nl_47_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_48_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_48_layer, 48,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_48_chain,
  NULL, &eltwise_49_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_38_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_28_output, &nl_37_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_38_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_38_layer, 38,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_38_chain,
  NULL, &eltwise_48_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_28_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_18_output, &nl_27_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_28_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_28_layer, 28,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_28_chain,
  NULL, &eltwise_38_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_18_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &nl_8_output, &nl_17_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_18_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_18_layer, 18,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_18_chain,
  NULL, &eltwise_28_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_8_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_7_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_8_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_8_layer, 8,
  SM_TYPE, 0x0, NULL,
  sm, forward_sm,
  &nl_8_chain,
  NULL, &eltwise_18_layer, AI_STATIC, 
  .nl_params = NULL, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_7_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_6_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_7_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_7_layer, 7,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_7_chain,
  NULL, &nl_8_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_6_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_6_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_6_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_6_nl_layer, 6,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_6_nl_chain,
  NULL, &gemm_7_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_6_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_6_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_6_layer, 6,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_6_chain,
  NULL, &nl_6_nl_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &serving_default_input0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_0_weights, &conv2d_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_0_scratch0, &conv2d_0_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_0_layer, 1,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool, forward_conv2d_if32of32wf32_nl_pool,
  &conv2d_0_chain,
  NULL, &gemm_6_layer, AI_STATIC, 
  .groups = 1, 
  .nl_params = NULL, 
  .nl_func = AI_HANDLE_PTR(forward_lite_nl_relu_if32of32), 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_f32), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_17_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_16_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_17_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_17_layer, 17,
  SM_TYPE, 0x0, NULL,
  sm, forward_sm,
  &nl_17_chain,
  NULL, &conv2d_0_layer, AI_STATIC, 
  .nl_params = NULL, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_16_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_15_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_16_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_16_weights, &gemm_16_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_16_layer, 16,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_16_chain,
  NULL, &nl_17_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_15_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_15_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_15_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_15_nl_layer, 15,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_15_nl_chain,
  NULL, &gemm_16_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_15_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_9_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_15_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_15_weights, &gemm_15_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_15_layer, 15,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_15_chain,
  NULL, &nl_15_nl_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_9_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &serving_default_input0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_9_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_9_weights, &conv2d_9_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_9_scratch0, &conv2d_9_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_9_layer, 10,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool, forward_conv2d_if32of32wf32_nl_pool,
  &conv2d_9_chain,
  NULL, &gemm_15_layer, AI_STATIC, 
  .groups = 1, 
  .nl_params = NULL, 
  .nl_func = AI_HANDLE_PTR(forward_lite_nl_relu_if32of32), 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_f32), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_27_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_26_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_27_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_27_layer, 27,
  SM_TYPE, 0x0, NULL,
  sm, forward_sm,
  &nl_27_chain,
  NULL, &conv2d_9_layer, AI_STATIC, 
  .nl_params = NULL, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_26_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_25_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_26_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_26_weights, &gemm_26_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_26_layer, 26,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_26_chain,
  NULL, &nl_27_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_25_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_25_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_25_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_25_nl_layer, 25,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_25_nl_chain,
  NULL, &gemm_26_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_25_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_19_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_25_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_25_weights, &gemm_25_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_25_layer, 25,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_25_chain,
  NULL, &nl_25_nl_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_19_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &serving_default_input0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_19_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_19_weights, &conv2d_19_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_19_scratch0, &conv2d_19_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_19_layer, 20,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool, forward_conv2d_if32of32wf32_nl_pool,
  &conv2d_19_chain,
  NULL, &gemm_25_layer, AI_STATIC, 
  .groups = 1, 
  .nl_params = NULL, 
  .nl_func = AI_HANDLE_PTR(forward_lite_nl_relu_if32of32), 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_f32), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_37_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_36_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_37_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_37_layer, 37,
  SM_TYPE, 0x0, NULL,
  sm, forward_sm,
  &nl_37_chain,
  NULL, &conv2d_19_layer, AI_STATIC, 
  .nl_params = NULL, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_36_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_35_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_36_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_36_weights, &gemm_36_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_36_layer, 36,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_36_chain,
  NULL, &nl_37_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_35_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_35_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_35_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_35_nl_layer, 35,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_35_nl_chain,
  NULL, &gemm_36_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_35_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_29_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_35_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_35_weights, &gemm_35_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_35_layer, 35,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_35_chain,
  NULL, &nl_35_nl_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_29_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &serving_default_input0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_29_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_29_weights, &conv2d_29_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_29_scratch0, &conv2d_29_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_29_layer, 30,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool, forward_conv2d_if32of32wf32_nl_pool,
  &conv2d_29_chain,
  NULL, &gemm_35_layer, AI_STATIC, 
  .groups = 1, 
  .nl_params = NULL, 
  .nl_func = AI_HANDLE_PTR(forward_lite_nl_relu_if32of32), 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_f32), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_47_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_46_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_47_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_47_layer, 47,
  SM_TYPE, 0x0, NULL,
  sm, forward_sm,
  &nl_47_chain,
  NULL, &conv2d_29_layer, AI_STATIC, 
  .nl_params = NULL, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_46_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_45_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_46_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_46_weights, &gemm_46_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_46_layer, 46,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_46_chain,
  NULL, &nl_47_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_45_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_45_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_45_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_45_nl_layer, 45,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_45_nl_chain,
  NULL, &gemm_46_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_45_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_39_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_45_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_45_weights, &gemm_45_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_45_layer, 45,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_45_chain,
  NULL, &nl_45_nl_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_39_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &serving_default_input0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_39_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_39_weights, &conv2d_39_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_39_scratch0, &conv2d_39_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_39_layer, 40,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool, forward_conv2d_if32of32wf32_nl_pool,
  &conv2d_39_chain,
  NULL, &gemm_45_layer, AI_STATIC, 
  .groups = 1, 
  .nl_params = NULL, 
  .nl_func = AI_HANDLE_PTR(forward_lite_nl_relu_if32of32), 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_f32), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 90276, 1, 1),
    90276, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 1736, 1, 1),
    1736, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &serving_default_input0_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &eltwise_49_output),
  &conv2d_39_layer, 0x3e5d1061, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 90276, 1, 1),
      90276, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 1736, 1, 1),
      1736, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &serving_default_input0_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &eltwise_49_output),
  &conv2d_39_layer, 0x3e5d1061, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_network_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    serving_default_input0_output_array.data = AI_PTR(g_network_activations_map[0] + 612);
    serving_default_input0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 612);
    conv2d_39_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 540);
    conv2d_39_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 540);
    conv2d_39_scratch1_array.data = AI_PTR(g_network_activations_map[0] + 28);
    conv2d_39_scratch1_array.data_start = AI_PTR(g_network_activations_map[0] + 28);
    conv2d_39_output_array.data = AI_PTR(g_network_activations_map[0] + 1124);
    conv2d_39_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1124);
    gemm_45_output_array.data = AI_PTR(g_network_activations_map[0] + 484);
    gemm_45_output_array.data_start = AI_PTR(g_network_activations_map[0] + 484);
    nl_45_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 484);
    nl_45_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 484);
    gemm_46_output_array.data = AI_PTR(g_network_activations_map[0] + 1608);
    gemm_46_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1608);
    nl_47_output_array.data = AI_PTR(g_network_activations_map[0] + 584);
    nl_47_output_array.data_start = AI_PTR(g_network_activations_map[0] + 584);
    conv2d_29_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 512);
    conv2d_29_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 512);
    conv2d_29_scratch1_array.data = AI_PTR(g_network_activations_map[0] + 1124);
    conv2d_29_scratch1_array.data_start = AI_PTR(g_network_activations_map[0] + 1124);
    conv2d_29_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_29_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_35_output_array.data = AI_PTR(g_network_activations_map[0] + 1124);
    gemm_35_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1124);
    nl_35_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 1124);
    nl_35_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1124);
    gemm_36_output_array.data = AI_PTR(g_network_activations_map[0] + 1252);
    gemm_36_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1252);
    nl_37_output_array.data = AI_PTR(g_network_activations_map[0] + 1124);
    nl_37_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1124);
    conv2d_19_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 512);
    conv2d_19_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 512);
    conv2d_19_scratch1_array.data = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_19_scratch1_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_19_output_array.data = AI_PTR(g_network_activations_map[0] + 1152);
    conv2d_19_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1152);
    gemm_25_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_25_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    nl_25_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 128);
    nl_25_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 128);
    gemm_26_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_26_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    nl_27_output_array.data = AI_PTR(g_network_activations_map[0] + 28);
    nl_27_output_array.data_start = AI_PTR(g_network_activations_map[0] + 28);
    conv2d_9_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 1152);
    conv2d_9_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 1152);
    conv2d_9_scratch1_array.data = AI_PTR(g_network_activations_map[0] + 72);
    conv2d_9_scratch1_array.data_start = AI_PTR(g_network_activations_map[0] + 72);
    conv2d_9_output_array.data = AI_PTR(g_network_activations_map[0] + 1224);
    conv2d_9_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1224);
    gemm_15_output_array.data = AI_PTR(g_network_activations_map[0] + 56);
    gemm_15_output_array.data_start = AI_PTR(g_network_activations_map[0] + 56);
    nl_15_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 184);
    nl_15_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 184);
    gemm_16_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_16_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    nl_17_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    nl_17_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 1152);
    conv2d_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 1152);
    conv2d_0_scratch1_array.data = AI_PTR(g_network_activations_map[0] + 56);
    conv2d_0_scratch1_array.data_start = AI_PTR(g_network_activations_map[0] + 56);
    conv2d_0_output_array.data = AI_PTR(g_network_activations_map[0] + 1224);
    conv2d_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1224);
    gemm_6_output_array.data = AI_PTR(g_network_activations_map[0] + 56);
    gemm_6_output_array.data_start = AI_PTR(g_network_activations_map[0] + 56);
    nl_6_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 184);
    nl_6_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 184);
    gemm_7_output_array.data = AI_PTR(g_network_activations_map[0] + 56);
    gemm_7_output_array.data_start = AI_PTR(g_network_activations_map[0] + 56);
    nl_8_output_array.data = AI_PTR(g_network_activations_map[0] + 84);
    nl_8_output_array.data_start = AI_PTR(g_network_activations_map[0] + 84);
    eltwise_18_output_array.data = AI_PTR(g_network_activations_map[0] + 56);
    eltwise_18_output_array.data_start = AI_PTR(g_network_activations_map[0] + 56);
    eltwise_28_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_28_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_38_output_array.data = AI_PTR(g_network_activations_map[0] + 28);
    eltwise_38_output_array.data_start = AI_PTR(g_network_activations_map[0] + 28);
    eltwise_48_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_48_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_49_output_array.data = AI_PTR(g_network_activations_map[0] + 28);
    eltwise_49_output_array.data_start = AI_PTR(g_network_activations_map[0] + 28);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}




/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_network_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    conv2d_39_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_39_weights_array.data = AI_PTR(g_network_weights_map[0] + 0);
    conv2d_39_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 0);
    conv2d_39_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_39_bias_array.data = AI_PTR(g_network_weights_map[0] + 576);
    conv2d_39_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 576);
    gemm_45_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_45_weights_array.data = AI_PTR(g_network_weights_map[0] + 608);
    gemm_45_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 608);
    gemm_45_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_45_bias_array.data = AI_PTR(g_network_weights_map[0] + 16992);
    gemm_45_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 16992);
    gemm_46_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_46_weights_array.data = AI_PTR(g_network_weights_map[0] + 17120);
    gemm_46_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 17120);
    gemm_46_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_46_bias_array.data = AI_PTR(g_network_weights_map[0] + 18016);
    gemm_46_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 18016);
    conv2d_29_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_29_weights_array.data = AI_PTR(g_network_weights_map[0] + 18044);
    conv2d_29_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 18044);
    conv2d_29_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_29_bias_array.data = AI_PTR(g_network_weights_map[0] + 18620);
    conv2d_29_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 18620);
    gemm_35_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_35_weights_array.data = AI_PTR(g_network_weights_map[0] + 18652);
    gemm_35_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 18652);
    gemm_35_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_35_bias_array.data = AI_PTR(g_network_weights_map[0] + 35036);
    gemm_35_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 35036);
    gemm_36_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_36_weights_array.data = AI_PTR(g_network_weights_map[0] + 35164);
    gemm_36_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 35164);
    gemm_36_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_36_bias_array.data = AI_PTR(g_network_weights_map[0] + 36060);
    gemm_36_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 36060);
    conv2d_19_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_19_weights_array.data = AI_PTR(g_network_weights_map[0] + 36088);
    conv2d_19_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 36088);
    conv2d_19_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_19_bias_array.data = AI_PTR(g_network_weights_map[0] + 36664);
    conv2d_19_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 36664);
    gemm_25_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_25_weights_array.data = AI_PTR(g_network_weights_map[0] + 36696);
    gemm_25_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 36696);
    gemm_25_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_25_bias_array.data = AI_PTR(g_network_weights_map[0] + 53080);
    gemm_25_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 53080);
    gemm_26_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_26_weights_array.data = AI_PTR(g_network_weights_map[0] + 53208);
    gemm_26_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 53208);
    gemm_26_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_26_bias_array.data = AI_PTR(g_network_weights_map[0] + 54104);
    gemm_26_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 54104);
    conv2d_9_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_9_weights_array.data = AI_PTR(g_network_weights_map[0] + 54132);
    conv2d_9_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 54132);
    conv2d_9_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_9_bias_array.data = AI_PTR(g_network_weights_map[0] + 54708);
    conv2d_9_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 54708);
    gemm_15_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_15_weights_array.data = AI_PTR(g_network_weights_map[0] + 54740);
    gemm_15_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 54740);
    gemm_15_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_15_bias_array.data = AI_PTR(g_network_weights_map[0] + 71124);
    gemm_15_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 71124);
    gemm_16_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_16_weights_array.data = AI_PTR(g_network_weights_map[0] + 71252);
    gemm_16_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 71252);
    gemm_16_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_16_bias_array.data = AI_PTR(g_network_weights_map[0] + 72148);
    gemm_16_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 72148);
    conv2d_0_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 72176);
    conv2d_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 72176);
    conv2d_0_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 72752);
    conv2d_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 72752);
    gemm_6_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_6_weights_array.data = AI_PTR(g_network_weights_map[0] + 72784);
    gemm_6_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 72784);
    gemm_6_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_6_bias_array.data = AI_PTR(g_network_weights_map[0] + 89168);
    gemm_6_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 89168);
    gemm_7_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_7_weights_array.data = AI_PTR(g_network_weights_map[0] + 89296);
    gemm_7_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 89296);
    gemm_7_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_7_bias_array.data = AI_PTR(g_network_weights_map[0] + 90192);
    gemm_7_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 90192);
    eltwise_49_scale_array.format |= AI_FMT_FLAG_CONST;
    eltwise_49_scale_array.data = AI_PTR(g_network_weights_map[0] + 90220);
    eltwise_49_scale_array.data_start = AI_PTR(g_network_weights_map[0] + 90220);
    eltwise_49_bias_array.format |= AI_FMT_FLAG_CONST;
    eltwise_49_bias_array.data = AI_PTR(g_network_weights_map[0] + 90248);
    eltwise_49_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 90248);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/



AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_network_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 73762,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x3e5d1061,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}



AI_API_ENTRY
ai_bool ai_network_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 73762,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x3e5d1061,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_error ai_network_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}


AI_API_ENTRY
ai_error ai_network_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    AI_CONTEXT_OBJ(&AI_NET_OBJ_INSTANCE),
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}


AI_API_ENTRY
ai_error ai_network_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
  ai_error err;
  ai_network_params params;

  err = ai_network_create(network, AI_NETWORK_DATA_CONFIG);
  if (err.type != AI_ERROR_NONE) {
    return err;
  }
  
  if (ai_network_data_params_get(&params) != true) {
    err = ai_network_get_error(*network);
    return err;
  }
#if defined(AI_NETWORK_DATA_ACTIVATIONS_COUNT)
  /* set the addresses of the activations buffers */
  for (ai_u16 idx=0; activations && idx<params.map_activations.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
  }
#endif
#if defined(AI_NETWORK_DATA_WEIGHTS_COUNT)
  /* set the addresses of the weight buffers */
  for (ai_u16 idx=0; weights && idx<params.map_weights.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
  }
#endif
  if (ai_network_init(*network, &params) != true) {
    err = ai_network_get_error(*network);
  }
  return err;
}


AI_API_ENTRY
ai_buffer* ai_network_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_buffer* ai_network_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_handle ai_network_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}


AI_API_ENTRY
ai_bool ai_network_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = AI_NETWORK_OBJ(ai_platform_network_init(network, params));
  ai_bool ok = true;

  if (!net_ctx) return false;
  ok &= network_configure_weights(net_ctx, params);
  ok &= network_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_network_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}


AI_API_ENTRY
ai_i32 ai_network_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_NETWORK_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

