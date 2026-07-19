/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Template for a pipelined GEMM kernel. Does not compute batching or support split-K.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Mma_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Epilogue_,             ///! Epilogue
  typename ThreadblockSwizzle_    ///! Threadblock swizzling function
>
struct GemmBatchedStdABFT {

  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using OutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  /// Parameters structure
  struct Params {
    cutlass::gemm::GemmCoord problem_size{};
    cutlass::gemm::GemmCoord grid_tiled_shape{};
    int swizzle_log_tile{0};
    typename Mma::IteratorA::Params params_A{};
    typename Mma::IteratorA::TensorRef ref_A{};
    int64_t stride_A{0};
    typename Mma::IteratorB::Params params_B{};
    typename Mma::IteratorB::TensorRef ref_B{};
    int64_t stride_B{0};
    typename Epilogue::OutputTileIterator::Params params_C{};
    typename Epilogue::OutputTileIterator::TensorRef ref_C{};
    int64_t stride_C{0};
    typename Epilogue::OutputTileIterator::Params params_D{};
    typename Epilogue::OutputTileIterator::TensorRef ref_D{};
    int64_t stride_D{0};
    typename OutputOp::Params epilogue{};
    int batch_count{1};
    int gemm_k_iterations{0};

    //
    // Methods
    //
    Params() = default;

    CUTLASS_HOST_DEVICE
    Params(
      cutlass::gemm::GemmCoord const & problem_size_,
      cutlass::gemm::GemmCoord const & grid_tiled_shape_,
      typename Mma::IteratorA::TensorRef ref_A_,
      int64_t stride_A_,
      typename Mma::IteratorB::TensorRef ref_B_,
      int64_t stride_B_,
      typename Epilogue::OutputTileIterator::TensorRef ref_C_,
      int64_t stride_C_,
      typename Epilogue::OutputTileIterator::TensorRef ref_D_,
      int64_t stride_D_,
      typename OutputOp::Params epilogue_,
      int batch_count_
    ):
      problem_size(problem_size_),
      grid_tiled_shape(grid_tiled_shape_),
      swizzle_log_tile(ThreadblockSwizzle().get_log_tile(grid_tiled_shape)),
      params_A(ref_A_.layout()),
      ref_A(ref_A_),
      stride_A(stride_A_),
      params_B(ref_B_.layout()),
      ref_B(ref_B_),
      stride_B(stride_B_),
      params_C(ref_C_.layout()),
      ref_C(ref_C_),
      stride_C(stride_C_),
      params_D(ref_D_.layout()),
      ref_D(ref_D_),
      stride_D(stride_D_),
      epilogue(epilogue_),
      batch_count(batch_count_),
      gemm_k_iterations((problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK) {}
  };

  /// Shared memory storage structure
  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

  template <typename T>
  __device__ void force_bit_one_bf16(T *dA, int bit){ 
    // 30 or 29
    float orgValue = static_cast<float>(*dA);
    float tmp = orgValue;
    // printf("%.4f ", orgValue);
    
    // uint32_t* intValue = reinterpret_cast<uint32_t*>(&orgValue);
    uint32_t intValue = *reinterpret_cast<uint32_t*>(&orgValue);
    uint16_t bf16_bits = static_cast<uint16_t>(intValue >> 16);
    bf16_bits |= (1u << bit);
    // *intValue &= ~ ((1u << bit));

    uint32_t new_int_value = (static_cast<uint32_t>(bf16_bits) << 16);
    float new_float = *reinterpret_cast<float*>(&new_int_value);
    *dA = static_cast<cutlass::bfloat16_t>(new_float);

    // if(tmp != new_float){
    //   // printf("%.4f %.4f ", tmp, new_float);
    //   // int idx = (*count) * 2;
    //   int idx = *count;
    //   *(buf + idx) = tmp;
    //   *(buf + (idx + 1)) = new_float;
    //   (*count) += 2;
    // }
    // printf("%.4f ", *(dA));
  }

  //
  // Methods
  //
  GemmBatchedStdABFT() = default;

  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage, uint8_t *Signature_Array,
                  int faulty_smid, int *faulty_MMAs, int *faulty_elements, int faulty_bit) {
    unsigned int real_smid;
    asm volatile("mov.u32 %0, %smid;" : "=r"(real_smid));
    int thread_idx = threadIdx.x;

    int num_block = params.grid_tiled_shape.m() * params.grid_tiled_shape.n();

    // Compute threadblock location
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    // Early exit if CTA is out of range
    if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
      params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {

      return;
    }


    // Each CTA handles multiple batch indices to accommodate limited range of CUDA grid's Z dimension
    for (int batch_idx = threadblock_swizzle.get_batch_idx(); 
      batch_idx < params.batch_count; 
      batch_idx += gridDim.z) {

      // Compute initial location in logical coordinates
      cutlass::MatrixCoord tb_offset_A{
        threadblock_tile_offset.m() * Mma::Shape::kM,
        0
      };

      cutlass::MatrixCoord tb_offset_B{
        0,
        threadblock_tile_offset.n() * Mma::Shape::kN
      };

      // Compute position within threadblock
      int thread_idx = threadIdx.x;

      // Construct iterators to A and B operands
      typename Mma::IteratorA iterator_A(
        params.params_A,
        params.ref_A.data(),
        params.problem_size.mk(),
        thread_idx,
        tb_offset_A);

      iterator_A.add_pointer_offset(params.stride_A * batch_idx);

      typename Mma::IteratorB iterator_B(
        params.params_B,
        params.ref_B.data(),
        params.problem_size.kn(),
        thread_idx,
        tb_offset_B);

      iterator_B.add_pointer_offset(params.stride_B * batch_idx);


      //
      // Main loop
      //

      // Broadcast the warp_id computed by lane 0 to ensure dependent code
      // is compiled as warp-uniform.
      int warp_idx = canonical_warp_idx_sync();

      int lane_idx = threadIdx.x % 32;
      
      Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

      typename Mma::FragmentC accumulators;

      accumulators.clear();


      // Compute threadblock-scoped matrix multiply-add
      mma(params.gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);

      //
      // Epilogue
      //

      OutputOp output_op(params.epilogue);

      //
      // Masked tile iterators constructed from members
      //

      threadblock_tile_offset =
          threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

      //assume identity swizzle
      MatrixCoord threadblock_offset(
        threadblock_tile_offset.m() * Mma::Shape::kM,
        threadblock_tile_offset.n() * Mma::Shape::kN
      );

      int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * params.grid_tiled_shape.m();
      if(thread_idx == 0) *(Signature_Array + batch_idx * num_block + block_idx) = real_smid;

      // Tile iterator writing to output tile
      typename Epilogue::OutputTileIterator iterator_C(
        params.params_C,
        params.ref_C.data(),
        params.problem_size.mn(),
        thread_idx,
        threadblock_offset
      );

      iterator_C.add_pointer_offset(params.stride_C * batch_idx);

      // Tile iterator writing to output tile
      typename Epilogue::OutputTileIterator iterator_D(
        params.params_D,
        params.ref_D.data(),
        params.problem_size.mn(),
        thread_idx,
        threadblock_offset
      );

      iterator_D.add_pointer_offset(params.stride_D * batch_idx);

      Epilogue epilogue(
        shared_storage.epilogue, 
        thread_idx, 
        warp_idx, 
        lane_idx);

      // run efficient epilogue
      epilogue(output_op, iterator_D, accumulators, iterator_C);

      // Fault Injection
      if(real_smid == faulty_smid && thread_idx == 0){
        // int mma_grid_m = params.problem_size.m() / 16;
        // int mma_grid_n = params.problem_size.n() / 8;
        int N = params.problem_size.n();
        int c = 0;
        for(int i = 0; i < 64; i++){
          int mma_m = (threadblock_tile_offset.m() * 128) + (faulty_MMAs[i] % 8) * 16;
          int mma_n = (threadblock_tile_offset.n() * 256) + (faulty_MMAs[i] / 8) * 8;

          // index of 1st faulty element
          int fault_m = faulty_elements[i] % 8;
          int fault_n = faulty_elements[i] / 8;
          if((mma_n + fault_n) < params.problem_size.n() ){
            int idx = (mma_m + fault_m) * N + (mma_n + fault_n);
            force_bit_one_bf16((params.ref_D.data() + idx + batch_idx * params.stride_D), faulty_bit);

            // index of 2nd faulty element (gap is 64)
            fault_m += 8;
            idx = (mma_m + fault_m) * N + (mma_n + fault_n);
            force_bit_one_bf16((params.ref_D.data() + idx + batch_idx * params.stride_D), faulty_bit);
          }
        }
      }
      // if(batch_idx == 0 && block_idx == 0 && thread_idx == 0){
      //   *(params.ref_D.data() + 1) = static_cast<cutlass::bfloat16_t>(1e6);
      //   printf("ground true faulty SM: %d\n", real_smid);
      // }
    }

    // 
    // if(real_smid == 0 && thread_idx == 0){
    //   *(params.ref_D.data() + 1) = static_cast<cutlass::bfloat16_t>(1e6);
    // }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass