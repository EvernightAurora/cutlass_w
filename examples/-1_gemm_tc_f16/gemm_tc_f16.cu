

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
// #include "gemm.h"
//#include "include/pch_common.h"
#include <iostream>
// #include <spgemm/gemm.h>
// #include <spgemm/refgemm.h>

constexpr bool kSplitKSerial = false;
// Mma::FragmentC cutlass::Array<float, 64, true>
// Mma::FragmentA cutlass::Array<float, 4, true>
// Mma::FragmentB cutlass::Array<float, 4, true>
constexpr bool TransA = false;
constexpr bool TransB = false;
constexpr bool TransC = false;

using TAcc = cutlass::half_t;
using TA = cutlass::half_t;
using TB = cutlass::half_t;
using TC = cutlass::half_t;
using ColumnMajor = cutlass::layout::ColumnMajor;
using RowMajor = cutlass::layout::RowMajor;
using LayoutA = std::conditional_t<TransA, ColumnMajor, RowMajor>;
using LayoutB = std::conditional_t<TransB, ColumnMajor, RowMajor>;
using LayoutC = std::conditional_t<TransC, ColumnMajor, RowMajor>;

// The code section below describes datatype for input, output matrices and
// computation between elements in input matrices.
using ElementAccumulator = TAcc; // <- data type of accumulator
using ElementComputeEpilogue =
    ElementAccumulator;   // <- data type of epilogue operations
using ElementInputA = TA; // <- data type of elements in input matrix A
using ElementInputB = TB; // <- data type of elements in input matrix B
using ElementOutput = TC; // <- data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices.
// Column Major for Matrix A, Row Major for Matrix B and Row Major for Matrix C
using LayoutInputA = LayoutA;
using LayoutInputB = LayoutB;  
using LayoutOutput = LayoutC;  

// This code section describes whether you want to use tensor cores or regular
// SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp; 

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm70;  

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<128, 128, 32>; // <- threadblock tile M = 128, N =
                                            // 128, K = 32
// This code section describes tile size a warp will compute
using ShapeMMAWarp =
    cutlass::gemm::GemmShape<64, 32, 32>; // <- warp tile M = 64, N = 64, K = 32
// This code section describes the size of MMA op
using ShapeMMAOp =
    cutlass::gemm::GemmShape<8, 8, 4>; // <- MMA Op tile M = 8, N = 8, K = 4

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock =
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // <- ??

// This code section describes ?
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput, // <- data type of output matrix
    128 / cutlass::sizeof_bits<
              ElementOutput>::value, // <- this is the number of elements per
                                     // vectorized memory access. For half
                                     // precision, it's 8 elements. This becomes
                                     // the vector width of math instructions in
                                     // epilogue too
    ElementAccumulator,              // <- data type of accumulator
    ElementComputeEpilogue>;         // <- data type for alpha/beta in linear
                                     // combination function

// Number of pipelines you want to use
constexpr int NumStages = 2;

using CutlassGemm = cutlass::gemm::device::Gemm< 
    ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput,
    LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock,
    ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
using G = CutlassGemm;
using TBS = CutlassGemm::ThreadblockShape;

using CGK = CutlassGemm::GemmKernel;
using DG = CutlassGemm::DefaultType;  
using DGOP = CutlassGemm::DefaultOperator; 
using DMma = CutlassGemm::DefaultType::MmaType; 
using DMmaCore = DMma::MmaCore;

using Arguments = CutlassGemm::Arguments;
