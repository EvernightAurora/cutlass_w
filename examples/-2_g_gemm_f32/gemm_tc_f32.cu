#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
// #include "gemm.h"
//#include "include/pch_common.h"
#include <iostream>
// #include <spgemm/gemm.h>
// #include <spgemm/refgemm.h>
#include"SimtKernel.h"
#include"stdlib.h"

template<typename T1, typename T2, int LINE>
struct ASSERT_TYPE_{
    ASSERT_TYPE_(){
        std::cout<<"Assertion Fault at line "<<LINE<<" that  ";
        std::cout<<typeid(T1).name()<<" is not "<<typeid(T2).name();
        std::cout<<"   at "<<__ASSERT_FUNCTION;
        std::cout<<std::endl;
        std::cout.flush();
    }
};

template<typename T1, int LINE>
struct ASSERT_TYPE_<T1, T1, LINE>{
};

#define ASSERT_TYPE(a1, a2)  {auto _A_T = (void*) new ASSERT_TYPE_<a1, a2, __LINE__>();}

template<typename T, int LINE>
struct S_T_{
    S_T_(){
        char Buf[1100];
        strcpy(Buf, __ASSERT_FUNCTION + 32);
        int l = strlen(Buf);
        while(Buf[l-1] != ';')
            --l;
        Buf[l] = '\0';
        std::cout<<"This Type at line " << LINE<< " ";
        std::cout<<" is  "<<Buf<<std::endl;
    };
};

#define SHOW_TYPE(t)  {auto S_T_##__LINE__ = S_T_<t, __LINE__>();}


constexpr bool kSplitKSerial = false;
// Mma::FragmentC cutlass::Array<float, 64, true>
// Mma::FragmentA cutlass::Array<float, 4, true>
// Mma::FragmentB cutlass::Array<float, 4, true>
constexpr bool TransA = false;
constexpr bool TransB = false;
constexpr bool TransC = false;

using TAcc = float;
using TA = float;
using TB = float;
using TC = float;
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
using MMAOp = cutlass::arch::OpClassSimt; 

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm70;  

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<128, 128, 8>; // <- threadblock tile 
// This code section describes tile size a warp will compute
using ShapeMMAWarp =
    cutlass::gemm::GemmShape<32, 64, 8>; // <- warp tile 
// This code section describes the size of MMA op
using ShapeMMAOp =
    cutlass::gemm::GemmShape<1, 1, 1>; // <- MMA Op tile

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock =
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // <- ??

// This code section describes ?
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput, // <- data type of output matrix
    1,
    float,
    float>;

// Number of pipelines you want to use
constexpr int NumStages = 2;

using CutlassGemm = cutlass::gemm::device::Gemm< 
    ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput,
    LayoutOutput, ElementAccumulator
    , MMAOp, SmArch, ShapeMMAThreadBlock,
    ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;



using GUESS_GEMM_CONFIG = typename cutlass::gemm::device::DefaultGemmConfiguration<
    MMAOp, SmArch,
    ElementInputA, ElementInputB, ElementOutput,
    ElementAccumulator>;

using GUESS_GEMM_OPERATOR = GUESS_GEMM_CONFIG::Operator;

constexpr auto ALIGN_A = GUESS_GEMM_CONFIG::kAlignmentA;
constexpr auto ALIGN_B = GUESS_GEMM_CONFIG::kAlignmentB;



using GUESS_KERNEL_SHELL = typename cutlass::gemm::kernel::DefaultGemm<
    ElementInputA, 
    LayoutInputA, 
    ALIGN_A, ElementInputB, LayoutInputB, ALIGN_B,
    ElementOutput, LayoutOutput, ElementAccumulator, 
    MMAOp, SmArch, 
    ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, 
    EpilogueOp, SwizzleThreadBlock, NumStages,
    false,
    GUESS_GEMM_OPERATOR,
    cutlass::gemm::SharedMemoryClearOption::kNone,
    false, false, false
  >;

using GUESS_KERNEL_GEMM_MMA_TYPE = typename cutlass::gemm::threadblock::DefaultMma<
    ElementInputA, 
    LayoutInputA, 
    ALIGN_A, ElementInputB, LayoutInputB, ALIGN_B,
    ElementAccumulator,  LayoutOutput,
    MMAOp, cutlass::arch::Sm50, 
    ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, 
    2,
    GUESS_GEMM_OPERATOR,
    false,
    cutlass::gemm::SharedMemoryClearOption::kNone,
    false,
    false>;


// get MMA related

static int const kWarpSize = cutlass::gemm::warp::WarpSize<cutlass::arch::OpClassSimt>::value;
static const int WarpNumThreadsM = cutlass::gemm::threadblock::detail::simt_get_warp_threads_m<ShapeMMAWarp>();
static const int WarpNumThreadsN = kWarpSize / WarpNumThreadsM;
static const int ThreadTileM = ShapeMMAWarp::kM / WarpNumThreadsM;
static const int ThreadTileN = ShapeMMAWarp::kN / WarpNumThreadsN;
static const int numElementsA = 128 / cutlass::sizeof_bits<ElementInputA>::value;
static const int numElementsB = 128 / cutlass::sizeof_bits<ElementInputB>::value;
static const int LaneM = cutlass::const_min(numElementsA, ThreadTileM);
static const int LaneN = cutlass::const_min(numElementsB, ThreadTileN);
using LaneMmaShape = cutlass::gemm::GemmShape<
      LaneM,
      LaneN,
      1>;
static int const PartitionsK = ShapeMMAThreadBlock::kK / ShapeMMAWarp::kK;
using WarpCount = cutlass::gemm::GemmShape<
    ShapeMMAThreadBlock::kM / ShapeMMAWarp::kM,
    ShapeMMAThreadBlock::kN / ShapeMMAWarp::kN,
    PartitionsK
  >;
static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
using Policy = cutlass::gemm::warp::MmaSimtPolicy<
    cutlass::MatrixShape<WarpNumThreadsM, WarpNumThreadsN>,   // WarpShape
    cutlass::layout::RowMajorInterleaved<LaneLayout>,         // LaneLayout
    LaneMmaShape
>;

using MmaWarpSimt = cutlass::gemm::warp::MmaSimt<
    ShapeMMAWarp,    /// Size of the Gemm problem - concept: gemm::GemmShape<> 128, 128, 8
    TA,     /// Data type of A elements
    ColumnMajor,  /// Layout of A matrix (concept: MatrixLayout)
    TB,     /// Data type of B elements
    RowMajor,  /// Layout of B matrix (concept: MatrixLayout)
    TC,     /// Element type of C matrix
    LayoutC,      /// Layout of C matrix (concept: MatrixLayout)
    Policy        /// Policy describing warp-level MmaSimtOp (concept: MmaSimtOp policy)
  >;
static int const kPaddingM = cutlass::gemm::threadblock::detail::simt_transpose_padding(kWarpSize, ShapeMMAThreadBlock::kK, cutlass::sizeof_bits<TA>::value);
using MmaPolicy = cutlass::gemm::threadblock::MmaPolicy<
    MmaWarpSimt,
    cutlass::MatrixShape<kPaddingM, 0>,    // skew for A matrix to avoid SMEM bank conflicts
    cutlass::MatrixShape<0, 0>, 
    WarpCount::kK
>;

using MMA_SHARE_STORAGE = cutlass::gemm::threadblock::MmaBase<
    ShapeMMAThreadBlock, MmaPolicy, 2>::SharedStorage;


using GUESS_MMA_CORE = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, TA, LayoutA,
      TB, LayoutB, ElementAccumulator, LayoutC,
      MMAOp, NumStages, GUESS_GEMM_OPERATOR>;

using GUESS_MMA_ITERATOR_A = cutlass::transform::threadblock::PredicatedTileIterator<
          cutlass::MatrixShape<GUESS_MMA_CORE::Shape::kM, GUESS_MMA_CORE::Shape::kK>,               //128   8
          TA, LayoutA, 1, typename GUESS_MMA_CORE::IteratorThreadMapA, ALIGN_A, false>;             //float, RowMajor,  1   ~ 1, false

using GUESS_MMA_ITERATOR_B = cutlass::transform::threadblock::PredicatedTileIterator<
          cutlass::MatrixShape<GUESS_MMA_CORE::Shape::kK, GUESS_MMA_CORE::Shape::kN>,               //128   8
          TB, LayoutB, 0, typename GUESS_MMA_CORE::IteratorThreadMapB, ALIGN_B, false>;             //float, RowMajor,  1   ~ 1, false

using SmemLayoutA = cutlass::layout::ColumnMajor;
using SmemLayoutB = cutlass::layout::RowMajor;

static int const kThreads = WarpCount::kCount * kWarpSize;
static int const kElementsPerAccess = 1;
using IteratorThreadMapA = cutlass::transform::PitchLinearStripminedThreadMap<
    cutlass::layout::PitchLinearShape<ShapeMMAThreadBlock::kK, ShapeMMAThreadBlock::kM>,                         //8 128,   256, 1
    kThreads,
    kElementsPerAccess
    >;
using SmemThreadMapA = cutlass::transform::TransposePitchLinearThreadMapSimt<IteratorThreadMapA>;
using IteratorThreadMapB = cutlass::transform::PitchLinearStripminedThreadMap<
    cutlass::layout::PitchLinearShape<ShapeMMAThreadBlock::kN, ShapeMMAThreadBlock::kK>,
    kThreads,
    kElementsPerAccess
  >;

using SmemIteratorA = cutlass::transform::threadblock::RegularTileIterator<
    cutlass::MatrixShape<ShapeMMAThreadBlock::kM, ShapeMMAThreadBlock::kK>, 
    TA, 
    SmemLayoutA,
    1,
    SmemThreadMapA
>;
using SmemIteratorB = cutlass::transform::threadblock::RegularTileIterator<
    cutlass::MatrixShape<ShapeMMAThreadBlock::kK, ShapeMMAThreadBlock::kN>, 
    TB, 
    SmemLayoutB,
    0,
    IteratorThreadMapB
>;

// WarpMMA:  Policy::Operator



void TYPE_ASSERT(){
    ASSERT_TYPE(CutlassGemm::ThreadblockShape, ShapeMMAThreadBlock);
            //GemmShape<128, 256, 64>
    ASSERT_TYPE(CutlassGemm::WarpShape, ShapeMMAWarp);
            //GemmShape<64, 64, 32>
    ASSERT_TYPE(CutlassGemm::InstructionShape, ShapeMMAOp);
            //GemmShape<1, 1, 1>
    ASSERT_TYPE(CutlassGemm::Operator, GUESS_GEMM_OPERATOR);
    ASSERT_TYPE(CutlassGemm::DefaultType, GUESS_KERNEL_SHELL);
    ASSERT_TYPE(CutlassGemm::DefaultType::MmaType, GUESS_KERNEL_GEMM_MMA_TYPE);
    
    ASSERT_TYPE(CutlassGemm::DefaultType::MmaType::ThreadblockMma::Policy, MmaPolicy);        
    ASSERT_TYPE(CutlassGemm::GemmKernel::Mma::SharedStorage, MMA_SHARE_STORAGE);

    ASSERT_TYPE(CutlassGemm::GemmKernel::Mma::IteratorA, GUESS_MMA_ITERATOR_A);
    ASSERT_TYPE(CutlassGemm::GemmKernel::Mma::IteratorB, GUESS_MMA_ITERATOR_B);

    ASSERT_TYPE(CutlassGemm::GemmKernel::Mma::SmemIteratorA, SmemIteratorA);
    ASSERT_TYPE(CutlassGemm::GemmKernel::Mma::SmemIteratorB, SmemIteratorB);
    
    

    SHOW_TYPE(CutlassGemm::GemmKernel::Mma::FragmentC);
    std::cout<<"MMA_TYPE: "<<GUESS_KERNEL_GEMM_MMA_TYPE::SIGN_LINE<<std::endl;
    std::cout<<"THREAD_BLOCK_MMA: "<<GUESS_KERNEL_GEMM_MMA_TYPE::ThreadblockMma::SIGN_LINE<<std::endl;
    std::cout<<"RUAAAAAA"<<GUESS_KERNEL_GEMM_MMA_TYPE::ThreadblockMma::kWarpGemmIterations<<std::endl;
}

const int M=128, N=300, K=24, G=3;
TA* pA, *dpA;
TB *pB, *dpB;
TC *pC, *dpC;
ElementOutput *pD, *dpD;


using MY_GEMM = MY_KERNEL::Gemm<
    TA, LayoutA, TB, LayoutB,
    TC, LayoutC, TC, G>;

template<typename T, typename MAJOR>
struct MakeMatrix{
    static cudaError_t alloc(T*&pm, T*&dpm, int a, int b, MAJOR mj, int filltype=1){
        size_t item_size = sizeof(T);
        pm = new T[a * b];
        auto err_t = cudaMalloc((void**)&dpm, a*b*item_size);
        int wi;
        memset(pm, 0, a*b*item_size);
        if(filltype == 1)
            for(wi=0;wi<a && wi<b; ++wi)
                pm[mj((cutlass::MatrixCoord){wi, wi})] = 1;
        else if (filltype == 2)
        {
            int wia;
            for(wi=0; wi<a; ++wi)
                for(wia=0; wia<b; ++wia)
                    pm[mj((cutlass::MatrixCoord){wi, wia})] = rand() * 1.0 / RAND_MAX;
        }else if (filltype == 3)
        {
            int wia;
            for(wi=0; wi<a; ++wi)
                for(wia=0; wia<b; ++wia)
                    pm[mj((cutlass::MatrixCoord){wi, wia})] = 1. / (1. + wi + wia);
        }
        
        
        cudaMemcpy(dpm, pm, a*b*item_size, cudaMemcpyHostToDevice);
        return err_t;
    }
    static void print(T*pm, T*dpm, int a, int b, MAJOR mj){
        cudaMemcpy(pm, dpm, a*b*sizeof(T), cudaMemcpyDeviceToHost);
        int wi, wia;
        for(wi=0; wi<a; ++wi){
            std::cout<<"| ";
            for(wia=0; wia<b; ++wia)
                std::cout<<pm[mj({wi, wia})]<<' ';
            std::cout<<std::endl;
        }
    }
    static bool compare(T* pa, T* dpa, T* pb, T* dpb, int a, int b, MAJOR mj, T maxeps_rate = 1e-6){
        cudaMemcpy(pa, dpa, a*b*sizeof(T), cudaMemcpyDeviceToHost);
        cudaMemcpy(pb, dpb, a*b*sizeof(T), cudaMemcpyDeviceToHost);
        int wi,wia;
        for(wi=0; wi<a; ++wi)
            for(wia=0; wia<b; ++wia){
                T va = pa[mj({wi, wia})], vb = pb[mj({wi, wia})];
                T max_eps = abs(max(va, vb)) * maxeps_rate;
                if (abs(va-vb) > max_eps){
                    std::cout<<"Not Match!"<<std::endl;
                    return false;
                } 
            }
        std::cout<<"Match!"<<std::endl;
        return true;
    }
};

void Init_MM(){
    MakeMatrix<TA, LayoutA>::alloc(pA, dpA, M, K, LayoutA(K), 3);
    MakeMatrix<TB, LayoutB>::alloc(pB, dpB, K, N/G, LayoutB(N/G), 3);
    MakeMatrix<TC, LayoutC>::alloc(pC, dpC, M, N, LayoutC(N), 0);
    MakeMatrix<ElementOutput, LayoutOutput>::alloc(pD, dpD, M, N, LayoutOutput(N), 0);        
}

void Process_MM(){
    
    std::cout<<"A"<<std::endl;

    MakeMatrix<TA, LayoutA>::print(pA, dpA, M, K, LayoutOutput(K));    
    std::cout<<"B"<<std::endl;

    MakeMatrix<TB, LayoutB>::print(pB, dpB, K, N/G, LayoutOutput(N/G)); 
    std::cout<<"ABEND"<<std::endl;

/*
    CutlassGemm::Arguments Args({
        {M, N, K},
        {dpA, K},
        {dpB, N},
        {dpC, N},
        {dpC, N},
        {1., 1.},
        G
    });
    auto Processor = CutlassGemm();

    auto Status = Processor(Args);
    cudaDeviceSynchronize();


    if(Status != cutlass::Status::kSuccess)
        std::cout<<"Failed"<<cudaGetErrorString(cudaGetLastError())<<std::endl;
    //MakeMatrix<TC, LayoutC>::print(pC, dpC, M, N, LayoutOutput(N));    
    */
    MY_GEMM::Arguments my_Args({
        {M, N, K},
        {dpA, K},
        {dpB, N/G},
        {dpD, N},
        {dpD, N},
        {1., 1.}
    });
    auto myProcessor = MY_GEMM();
    auto my_status = myProcessor(my_Args);
    
    cudaDeviceSynchronize();
    if(my_status != cudaSuccess)
        std::cout<<"Failed"<<cudaGetErrorString(cudaGetLastError())<<std::endl;
    cudaMemcpy(pD, dpD, N*M*sizeof(float), cudaMemcpyDeviceToHost);
    MakeMatrix<ElementOutput, LayoutOutput>::print(pD, dpD, M, 1, LayoutOutput(N));  
    //MakeMatrix<TC, LayoutC>::compare(pC, dpC, pD, dpD, M, N, LayoutOutput(N));
    
}



using Arguments = CutlassGemm::Arguments;



#include<time.h>

int main(){
    TYPE_ASSERT();


    Init_MM();
    Process_MM();

    char Buf[1100];
    time_t rtime;
    time(&rtime);
    strftime(Buf, sizeof(Buf), "%T\n", localtime(&rtime));
    std::cout<<Buf;
    std::cout<<"Finished"<<std::endl<<std::endl<<std::endl;
    std::cout.flush();
    std::cout<<MY_GEMM::IteratorThreadMapA::Iterations::kCount<<std::endl;
    return 0;
}