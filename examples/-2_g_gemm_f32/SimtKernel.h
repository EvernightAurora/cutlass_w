#include"PipelineMma.h"
#include "cutlass/arch/arch.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/gemm/warp/mma_simt_policy.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_pitch_linear.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_pitch_linear_2dthreadtile.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/matrix_coord.h"
#include <algorithm>
#include <iostream>

namespace MY_KERNEL{
    template<typename Operator>
    __global__ void Kernel(typename Operator::KernelParams kp){
        extern __shared__ unsigned char ShareDest[];
        Operator::ActivateKernel(kp, ShareDest);
    }

template<
    typename TypeA_,
    typename LayoutA_,
    typename TypeB_,
    typename LayoutB_,
    typename TypeC_,
    typename LayoutC_,
    typename TypeAcc_ = TypeC_,
    int Groups_ = 1,
    typename OpClass_ = cutlass::arch::OpClassSimt,
    typename ArchTag_ = cutlass::arch::Sm70,
    typename TBShape_ = cutlass::gemm::GemmShape<128, 128, 8>,
    typename WShape_ = cutlass::gemm::GemmShape<32, 64, 8>,
    typename IShape_ = cutlass::gemm::GemmShape<1, 1, 1>,
    typename EpilogOp_ = cutlass::epilogue::thread::LinearCombination<
            TypeC_, 1, float, float>,
    typename ThreadblockSwizzle_ =
        typename cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    int Stages = 2,
    int Align_A = 1,
    int Align_B = 1,
    bool SplitKSerial = false,
    typename Operator_ = cutlass::arch::OpMultiplyAdd,
    bool ScatterA = false,
    bool ScatterB = false,
    bool ScatterC = false
    >
struct Gemm;


template<
    typename TypeA_,
    typename TypeB_,
    typename TypeC_,
    typename LayoutC_,
    typename TypeAcc_,
    int Groups_,
    typename TBShape_,
    typename WShape_ ,
    typename EpilogOp_,
    typename ThreadblockSwizzle_
    >
struct Gemm<TypeA_, cutlass::layout::RowMajor, TypeB_, cutlass::layout::RowMajor, TypeC_, LayoutC_, TypeAcc_, Groups_,
            cutlass::arch::OpClassSimt, cutlass::arch::Sm70,
            TBShape_, WShape_, cutlass::gemm::GemmShape<1, 1, 1>, EpilogOp_, ThreadblockSwizzle_,
            2, 1, 1, false, cutlass::arch::OpMultiplyAdd, false, false, false>{
    using TypeA = TypeA_;
    using TypeB = TypeB_;
    using TypeC = TypeC_;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = LayoutC_;
    using TypeAcc = TypeAcc_;
    using OpClass = cutlass::arch::OpClassSimt;
    using ArchTag = cutlass::arch::Sm70;
    using TBShape = TBShape_;
    using WShape = WShape_;
    using IShape = cutlass::gemm::GemmShape<1, 1, 1>;
    using EpilogOp = EpilogOp_;
    using ThreadblockSwizzle = ThreadblockSwizzle_;

    static int const Groups = Groups_;
    
    using SmemLayoutA = cutlass::layout::ColumnMajor;
    using SmemLayoutB = cutlass::layout::RowMajor;
    static int const PartitionsK = TBShape::kK / WShape::kK;    //  1
    using WarpCount = cutlass::gemm::GemmShape<
        TBShape::kM / WShape::kM,               //4
        TBShape::kN / WShape::kN,               //2
        PartitionsK                             //1
    >;
    static int const kWarpSize = cutlass::gemm::warp::WarpSize<cutlass::arch::OpClassSimt>::value;          // 32
    using IteratorThreadMapA = cutlass::transform::PitchLinearStripminedThreadMap<
        cutlass::layout::PitchLinearShape<TBShape::kK, TBShape::kM>,                         //8 128,   256, 1
        WarpCount::kCount * kWarpSize,                  //32
        1
    >;
    using SmemThreadMapA = cutlass::transform::TransposePitchLinearThreadMapSimt<IteratorThreadMapA>;
    using IteratorThreadMapB = cutlass::transform::PitchLinearStripminedThreadMap<
        cutlass::layout::PitchLinearShape<TBShape::kN, TBShape::kK>,
        WarpCount::kCount * kWarpSize,
        1
    >;
    using SmemIteratorA = cutlass::transform::threadblock::RegularTileIterator<
        cutlass::MatrixShape<TBShape::kM, TBShape::kK>, 
        TypeA, 
        SmemLayoutA,                                                //Column Major
        1,
        SmemThreadMapA
    >;
    using SmemIteratorB = cutlass::transform::threadblock::RegularTileIterator<
        cutlass::MatrixShape<TBShape::kK, TBShape::kN>, 
        TypeB, 
        SmemLayoutB,                                                // Row Major
        0,
        IteratorThreadMapB
    >;
    static const int WarpNumThreadsM = 4;//cutlass::gemm::threadblock::detail::simt_get_warp_threads_m<WShape>();
    static const int WarpNumThreadsN = kWarpSize / WarpNumThreadsM;
    static const int ThreadTileM = WShape::kM / WarpNumThreadsM;
    static const int ThreadTileN = WShape::kN / WarpNumThreadsN;
    static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
    static const int numElementsA = 128 / cutlass::sizeof_bits<TypeA>::value;
    static const int numElementsB = 128 / cutlass::sizeof_bits<TypeB>::value;
    static const int LaneM = cutlass::const_min(numElementsA, ThreadTileM);
    static const int LaneN = cutlass::const_min(numElementsB, ThreadTileN);
    using LaneMmaShape = cutlass::gemm::GemmShape<
        LaneM,
        LaneN,
        1>;
    using Policy = cutlass::gemm::warp::MmaSimtPolicy<
        cutlass::MatrixShape<WarpNumThreadsM, WarpNumThreadsN>,   // WarpShape
        cutlass::layout::RowMajorInterleaved<LaneLayout>,         // LaneLayout
        LaneMmaShape
    >;

    using MmaWarpSimt = cutlass::gemm::warp::MmaSimt<
        WShape,    /// Size of the Gemm problem - concept: gemm::GemmShape<> 128, 128, 8
        TypeA,     /// Data type of A elements
        cutlass::layout::ColumnMajor,  /// Layout of A matrix (concept: MatrixLayout)
        TypeB,     /// Data type of B elements
        cutlass::layout::RowMajor,  /// Layout of B matrix (concept: MatrixLayout)
        TypeC,     /// Element type of C matrix
        LayoutC,      /// Layout of C matrix (concept: MatrixLayout)
        Policy        /// Policy describing warp-level MmaSimtOp (concept: MmaSimtOp policy)
    >;
    static int const kPaddingM = cutlass::gemm::threadblock::detail::simt_transpose_padding(kWarpSize, TBShape::kK, cutlass::sizeof_bits<TypeA>::value);
    using MmaPolicy = cutlass::gemm::threadblock::MmaPolicy<
        MmaWarpSimt,
        cutlass::MatrixShape<kPaddingM, 0>,    // skew for A matrix to avoid SMEM bank conflicts
        cutlass::MatrixShape<0, 0>, 
        WarpCount::kK
    >;

    using IteratorA =
        cutlass::transform::threadblock::PredicatedTileIterator<
        cutlass::MatrixShape<TBShape::kM, TBShape::kK>,
        TypeA, LayoutA, 1, IteratorThreadMapA, 1, false>;

    using IteratorB =
        cutlass::transform::threadblock::PredicatedTileIterator<
        cutlass::MatrixShape<TBShape::kK, TBShape::kN>,
        TypeB, LayoutB, 0, IteratorThreadMapB, 1, false>;
    
    using BACK_MMA = MY_MMA::PipelinedMma<TBShape, 
        IteratorA, SmemIteratorA,
        IteratorB, SmemIteratorB,
        TypeAcc, LayoutC, MmaPolicy>;
    

    static int const kThreadCount = 32 * BACK_MMA::WarpCount::kCount;

    using RegularEpilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueSimt<
        TBShape,
        typename BACK_MMA::Operator,
        EpilogOp,
        EpilogOp::kCount,
        false
        >::Epilogue;

    using Affine2Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueSimtAffineRankN<
        2,
        TBShape,
        typename BACK_MMA::Operator,
        EpilogOp,
        EpilogOp::kCount
        >::Epilogue;

    using Epilogue = typename cutlass::platform::conditional<cutlass::platform::is_same<LayoutC, cutlass::layout::RowMajor>::value,
                                                            RegularEpilogue,
                                                            Affine2Epilogue>::type;
        
    struct KernelParams{
        cutlass::gemm::GemmCoord problem_size, grid_tiled_shape;
        int swizzle_log_tile;
        typename BACK_MMA::IteratorA::Params params_A;
        typename BACK_MMA::IteratorA::TensorRef ref_A;
        typename BACK_MMA::IteratorB::Params params_B;
        typename BACK_MMA::IteratorB::TensorRef ref_B;
        typename Epilogue::OutputTileIterator::Params params_C;
        typename Epilogue::OutputTileIterator::TensorRef ref_C;
        typename Epilogue::OutputTileIterator::Params params_D;
        typename Epilogue::OutputTileIterator::TensorRef ref_D;
        typename Epilogue::OutputOp::Params output_op;
        int* semaphore;
        int gemm_k_size;
        CUTLASS_HOST_DEVICE
        KernelParams(){};
        CUTLASS_HOST_DEVICE
        KernelParams(
            cutlass::gemm::GemmCoord const & problem_size,
            cutlass::gemm::GemmCoord const & grid_tiled_shape,
            typename BACK_MMA::IteratorA::TensorRef ref_A,
            typename BACK_MMA::IteratorB::TensorRef ref_B,
            typename Epilogue::OutputTileIterator::TensorRef ref_C,
            typename Epilogue::OutputTileIterator::TensorRef ref_D,
            typename Epilogue::OutputOp::Params output_op = typename Epilogue::OutputOp::Params(),
            int *workspace = nullptr
        ):
            problem_size(problem_size),
            grid_tiled_shape(grid_tiled_shape),
            swizzle_log_tile(ThreadblockSwizzle().get_log_tile(grid_tiled_shape)),
            params_A(ref_A.layout()),
            ref_A(ref_A),
            params_B(ref_B.layout()),
            ref_B(ref_B),
            params_C(ref_C.layout()),
            ref_C(ref_C),
            params_D(ref_D.layout()),
            ref_D(ref_D),
            output_op(output_op),
            semaphore(workspace){
                int total_gemm_k_iterations = (problem_size.k() + BACK_MMA::TBShape::kK - 1) / BACK_MMA::TBShape::kK;
                int gemm_k_iterations = (total_gemm_k_iterations + grid_tiled_shape.k() - 1) / grid_tiled_shape.k();

                gemm_k_size = gemm_k_iterations * BACK_MMA::TBShape::kK;
            }
    };

    struct Arguments{
        cutlass::gemm::GemmCoord problem_size;
        cutlass::TensorRef<TypeA const, LayoutA> ref_A;
        cutlass::TensorRef<TypeB const, LayoutB> ref_B;
        cutlass::TensorRef<TypeC const, LayoutC> ref_C;
        cutlass::TensorRef<TypeC, LayoutC> ref_D;
        typename EpilogOp::Params epilogue;
        CUTLASS_HOST_DEVICE
        Arguments(): problem_size(0, 0, 0){}
        CUTLASS_HOST_DEVICE
        Arguments(
            cutlass::gemm::GemmCoord problem_size_,
            cutlass::TensorRef<TypeA const, LayoutA> ref_A_,
            cutlass::TensorRef<TypeB const, LayoutB> ref_B_,
            cutlass::TensorRef<TypeC const, LayoutC> ref_C_,
            cutlass::TensorRef<TypeC, LayoutC> ref_D_,
            typename EpilogOp::Params epilogue_ = 
                typename EpilogOp::Params()
        ): problem_size(problem_size_), ref_A(ref_A_), ref_B(ref_B_), ref_C(ref_C_),ref_D(ref_D_),epilogue(epilogue_){}
    };
    KernelParams params;

    union SharedStorage{
        typename BACK_MMA::SharedStorage main_loop;
        typename Epilogue::SharedStorage epilogue;
    };

    CUTLASS_DEVICE
    static void ActivateKernel(const KernelParams&params, unsigned char SharedDest[]){
        //extern __shared__  unsigned char SharedDest[];
        //printf("%x\n", SharedDest);
        
        //return;

        SharedStorage& shared_storage = *(reinterpret_cast<SharedStorage*> (SharedDest));
        ThreadblockSwizzle threadblock_swizzle;
        cutlass::gemm::GemmCoord threadblock_tile_offset = 
            threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

        if(params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
            params.grid_tiled_shape.n() <= threadblock_tile_offset.n()){
                return;
            }
        cutlass::MatrixCoord tb_offset_A{
            threadblock_tile_offset.m() * BACK_MMA::TBShape::kM,
            threadblock_tile_offset.k() * params.gemm_k_size 
        };
        cutlass::MatrixCoord tb_offset_B{
            threadblock_tile_offset.k() * params.gemm_k_size,
            threadblock_tile_offset.n() * BACK_MMA::TBShape::kN
        };
        int problem_size_k = min(
            params.problem_size.k(), (threadblock_tile_offset.k() + 1) * params.gemm_k_size
        );
        int gemm_k_iterations = (problem_size_k - 
            tb_offset_A.column() + BACK_MMA::TBShape::kK - 1) / BACK_MMA::TBShape::kK;
        int thread_idx = threadIdx.x;
        //printf("(%d# %d -> %d\n", threadIdx.x, gemm_k_iterations.k(), k_group_idx);

        typename BACK_MMA::IteratorA iter_A(
            params.params_A, params.ref_A.data(),
            {params.problem_size.m(), problem_size_k},
            thread_idx, tb_offset_A, nullptr);
        typename BACK_MMA::IteratorB iter_B(
            params.params_B, params.ref_B.data(),
            {problem_size_k, params.problem_size.n()},
            thread_idx, tb_offset_B, nullptr);
        
        int warp_idx = __shfl_sync(0xffffffff, threadIdx.x/32, 0);
        int lane_idx = threadIdx.x % 32;
        // threadid to warp may not aligned

        BACK_MMA Mma(&(shared_storage.main_loop), thread_idx, warp_idx, lane_idx);
        typename BACK_MMA::FragmentC acc[Groups];
        for(int i=0; i<Groups; ++i)
            acc[i].clear();
        //auto _acc = typename BACK_MMA::FragmentC();
        //_acc.clear();
        if(true)
            Mma.process_2(gemm_k_iterations, acc, iter_A, iter_B, acc, Groups);
        typename Epilogue::OutputOp output_op(params.output_op);
        threadblock_tile_offset = threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);
        for(int k_group_idx=0; k_group_idx<Groups; ++k_group_idx){
            cutlass::MatrixCoord threadblock_offset(
                threadblock_tile_offset.m() * BACK_MMA::TBShape::kM,
                threadblock_tile_offset.n() * BACK_MMA::TBShape::kN + k_group_idx * (params.problem_size.n())
            );

            auto p_mn = params.problem_size.mn();
            p_mn.at(1) *= Groups;
            typename Epilogue::OutputTileIterator iterator_C{
                params.params_C, params.ref_C.data(),
                p_mn, thread_idx,
                threadblock_offset, nullptr};
            typename Epilogue::OutputTileIterator iterator_D{
                params.params_D, params.ref_D.data(),
                p_mn, thread_idx,
                threadblock_offset, nullptr};

            Epilogue epilogue(
                shared_storage.epilogue,
                thread_idx,
                warp_idx,
                lane_idx
            );
            epilogue(output_op, iterator_D, acc[k_group_idx], iterator_C);
        }
        //delete[] acc;
    }

    void settingParams(const Arguments& args, void *workspace=nullptr){
        ThreadblockSwizzle threadblock_swizzle;
        auto new_problem_size = args.problem_size;
        new_problem_size.n() /= Groups;
        cutlass::gemm::GemmCoord grid_shape = threadblock_swizzle.get_tiled_shape(
            new_problem_size,
            {TBShape::kM, TBShape::kN, TBShape::kK}, 1);
        params = KernelParams(new_problem_size,
            grid_shape,
            args.ref_A.non_const_ref(),
            args.ref_B.non_const_ref(),
            args.ref_C.non_const_ref(),
            args.ref_D,
            args.epilogue,
            static_cast<int*> (workspace));
    }
    
    cudaError_t operator ()(const Arguments& arg, void*workspace=nullptr, cudaStream_t stream=nullptr){
        if(arg.problem_size.n() % Groups || arg.problem_size.k() % Groups)
            return cudaErrorInvalidValue;
        if(arg.problem_size.k() % (Groups * TBShape::kK))
            return cudaErrorNotYetImplemented;
        //if(arg.priblem_size.n() % arg.groups) return cudaErrorInvalidValue;
        settingParams(arg, workspace);
        ThreadblockSwizzle threadblock_swizzle;
        dim3 grid = threadblock_swizzle.get_grid_shape(params.grid_tiled_shape);
        dim3 block(kThreadCount, 1, 1);
        constexpr const int smem_size = sizeof(SharedStorage);
        static_assert(smem_size < (48<<10));
        Kernel<Gemm> <<<grid, block, smem_size, stream>>>(params);

        return cudaGetLastError();
    }
    Gemm(){}
};


}