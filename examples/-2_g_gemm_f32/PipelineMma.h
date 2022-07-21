#include"cutlass/cutlass.h"
#include"cutlass/gemm/gemm.h"
#include"cutlass/matrix_shape.h"
#include"cutlass/array.h"
#include"cutlass/numeric_conversion.h"

namespace MY_MMA{


template<
    typename TBShape_,
    typename IteratorA_,
    typename SmemIteratorA_,
    typename IteratorB_,
    typename SmemIteratorB_,
    typename TypeC_,
    typename LayoutC_,
    typename Policy_>
class PipelinedMma{
    public:
    using Policy = Policy_;
    using TBShape = TBShape_;
    using IteratorA = IteratorA_;
    using SmemIteratorA = SmemIteratorA_;
    using IteratorB = IteratorB_;
    using SmemIteratorB = SmemIteratorB_;
    using TypeC = TypeC_;
    using LayoutC = LayoutC_;

    using WarpCount = cutlass::gemm::GemmShape<TBShape::kM / Policy::Operator::Shape::kM,
                                TBShape::kN / Policy::Operator::Shape::kN,
                                TBShape::kK / Policy::Operator::Shape::kK>;

    using Operator = typename Policy::Operator;

    using TensorRefA = cutlass::TensorRef<typename Operator::ElementA, typename Operator::LayoutA>;
    using TensorRefB = cutlass::TensorRef<typename Operator::ElementB, typename Operator::LayoutB>;

    static auto const kStages = 2;
    class SharedStorage{
        public:
        using ShapeA = cutlass::MatrixShape<
                TBShape::kM + Policy::SmemPaddingA::kRow,
                TBShape::kK * kStages + Policy::SmemPaddingA::kColumn>;
        using ShapeB = cutlass::MatrixShape<
                TBShape::kK * kStages + Policy::SmemPaddingB::kRow,
                TBShape::kN + Policy::SmemPaddingB::kColumn>;
        template<typename Type, int N>
        class Buffer{
            alignas(16) Type storage[ShapeA::kCount];
            static auto const kCount=N;
            public:
            CUTLASS_HOST_DEVICE 
            Type* data(){return storage;}
            CUTLASS_HOST_DEVICE
            const Type* data()const{return storage;}
            CUTLASS_HOST_DEVICE
            constexpr auto size()const{return kCount;}
        };
        Buffer<typename Operator::ElementA, ShapeA::kCount> operand_A;
        Buffer<typename Operator::ElementB, ShapeB::kCount> operand_B;
        CUTLASS_DEVICE
        static typename Operator::LayoutA LayoutA() {return Operator::LayoutA::packed({ShapeA::kRow, ShapeA::kColumn});}
        CUTLASS_HOST_DEVICE
        static typename Operator::LayoutB LayoutB() {return Operator::LayoutB::packed({ShapeB::kRow, ShapeB::kColumn});}
        CUTLASS_HOST_DEVICE
        TensorRefA operand_A_ref(){return TensorRefA(operand_A.data(), LayoutA());}
        CUTLASS_HOST_DEVICE
        TensorRefB operand_B_ref(){return TensorRefB(operand_B.data(), LayoutB());}
        
    };

    typename Operator::IteratorA warp_tile_iterator_A_;
    typename Operator::IteratorB warp_tile_iterator_B_;

    SmemIteratorA smem_iterator_A_;
    SmemIteratorB smem_iterator_B_;
    using FragmentA = typename IteratorA::Fragment;
    using FragmentB = typename IteratorB::Fragment;
    using FragmentC = typename Policy::Operator::FragmentC;
    using ArchTag = typename Policy::Operator::ArchTag;

    using WarpFragmentA = typename Operator::FragmentA;
    using WarpFragmentB = typename Operator::FragmentB;

    static int const kWarpGemmIterations =  (Policy::Operator::Shape::kK / Operator::Policy::MmaShape::kK);
    public:
    CUTLASS_DEVICE
    PipelinedMma(
        void* pshared_storage,
        int thread_idx,
        int warp_idx,
        int lane_idx
    ){
        SharedStorage &shared_storage = 
                *(reinterpret_cast<SharedStorage*>
                            (pshared_storage));
        warp_tile_iterator_A_ = typename Operator::IteratorA(
            shared_storage.operand_A_ref(), lane_idx);
        warp_tile_iterator_B_ = typename Operator::IteratorB(
            shared_storage.operand_B_ref(), lane_idx);
        smem_iterator_A_ = SmemIteratorA(
            shared_storage.operand_A_ref(), thread_idx);
        smem_iterator_B_ = SmemIteratorB(
            shared_storage.operand_B_ref(), thread_idx);
        int N = TBShape::kN / Policy::Operator::Shape::kN, M = TBShape::kM / Policy::Operator::Shape::kM;
        int NM = N * M;
        int warp_idx_nm = warp_idx % NM, warp_idx_k = warp_idx / NM;
        int warp_idx_n = warp_idx_nm / M;
        int warp_idx_m = warp_idx_nm % M;
        warp_tile_iterator_A_.add_tile_offset({warp_idx_m, warp_idx_k * kWarpGemmIterations});
        warp_tile_iterator_B_.add_tile_offset({warp_idx_k * kWarpGemmIterations, warp_idx_n});
    }

    CUTLASS_DEVICE
    void operator ()(
        int gemm_k_iterations,
        FragmentC& Accum,
        IteratorA iterator_A,
        IteratorB iterator_B,
        FragmentC const& src_Accum
    ){
        Accum = src_Accum;
        FragmentA buf_a;
        FragmentB buf_b;
        buf_a.clear();
        buf_b.clear();
        iterator_A.load(buf_a);
        iterator_B.load(buf_b);
        ++iterator_A;
        ++iterator_B;
        smem_iterator_A_.store(buf_a);
        smem_iterator_B_.store(buf_b);
        ++smem_iterator_A_;
        ++smem_iterator_B_;

        WarpFragmentA wbuf_a[2];
        WarpFragmentB wbuf_b[2];
        Operator warp_mma;

        warp_tile_iterator_A_.set_kgroup_index(0);
        warp_tile_iterator_B_.set_kgroup_index(0);
        warp_tile_iterator_A_.load(wbuf_a[0]);
        warp_tile_iterator_B_.load(wbuf_b[0]);
        ++warp_tile_iterator_A_;
        ++warp_tile_iterator_B_;


        bool smem_write_stage_idx = 1;
        iterator_A.clear_mask(gemm_k_iterations<=1);
        iterator_B.clear_mask(gemm_k_iterations<=1);

        for(; gemm_k_iterations>0; --gemm_k_iterations){
            int warp_mma_k;
            for(warp_mma_k=0; warp_mma_k<kWarpGemmIterations; ++warp_mma_k){
                if(warp_mma_k == kWarpGemmIterations - 1){
                    smem_iterator_A_.store(buf_a);
                    smem_iterator_B_.store(buf_b);

                    __syncthreads();

                    ++smem_iterator_A_;
                    ++smem_iterator_B_;

                    if(smem_write_stage_idx){
                        smem_iterator_A_.add_tile_offset({0, -kStages});
                        smem_iterator_B_.add_tile_offset({-kStages, 0});
                    }
                    else{
                        warp_tile_iterator_A_.add_tile_offset(
                            {0, -kStages * Policy::kPartitionsK * kWarpGemmIterations});
                        warp_tile_iterator_B_.add_tile_offset(
                            {-kStages * Policy::kPartitionsK * kWarpGemmIterations, 0});
                    }
                    smem_write_stage_idx ^= 1;
                }
                warp_tile_iterator_A_.set_kgroup_index((warp_mma_k+1)%kWarpGemmIterations);
                warp_tile_iterator_B_.set_kgroup_index((warp_mma_k+1)%kWarpGemmIterations);
                
                warp_tile_iterator_A_.load(wbuf_a[!(warp_mma_k & 1)]);
                warp_tile_iterator_B_.load(wbuf_b[!(warp_mma_k & 1)]);
                ++warp_tile_iterator_A_;
                ++warp_tile_iterator_B_;

                if(warp_mma_k == 0){
                    iterator_A.load(buf_a);
                    iterator_B.load(buf_b);
                    ++iterator_A;
                    ++iterator_B;
                    iterator_A.clear_mask(gemm_k_iterations <= 2);
                    iterator_B.clear_mask(gemm_k_iterations <= 2);
                }
                warp_mma(Accum, wbuf_a[warp_mma_k & 1], wbuf_b[warp_mma_k & 1], Accum);
            }
        }

    }

    CUTLASS_DEVICE
    void process_2(
        int gemm_k_iterations,
        FragmentC *Accum,
        IteratorA iterator_A,
        IteratorB iterator_B,
        FragmentC const* src_Accum,
        int groups=1
    ){
        for(int i = 0; i<groups; ++i)
            Accum[i] = src_Accum[i];
        FragmentA buf_a;
        FragmentB buf_b;
        buf_a.clear();
        buf_b.clear();
        iterator_A.load(buf_a);
        iterator_B.load(buf_b);
        ++iterator_A;
        ++iterator_B;
        smem_iterator_A_.store(buf_a);
        smem_iterator_B_.store(buf_b);
        ++smem_iterator_A_;
        ++smem_iterator_B_;

        WarpFragmentA wbuf_a[2];
        WarpFragmentB wbuf_b[2];
        Operator warp_mma;

        warp_tile_iterator_A_.set_kgroup_index(0);
        warp_tile_iterator_B_.set_kgroup_index(0);
        warp_tile_iterator_A_.load(wbuf_a[0]);
        warp_tile_iterator_B_.load(wbuf_b[0]);
        ++warp_tile_iterator_A_;
        ++warp_tile_iterator_B_;


        bool smem_write_stage_idx = 1;
        iterator_A.clear_mask(gemm_k_iterations<=1);
        iterator_B.clear_mask(gemm_k_iterations<=1);
        int group_per = gemm_k_iterations / groups;
        int now_ite = 0;
        for(; gemm_k_iterations>0; --gemm_k_iterations){
            int warp_mma_k;
            for(warp_mma_k=0; warp_mma_k<kWarpGemmIterations; ++warp_mma_k){
                if(warp_mma_k == kWarpGemmIterations - 1){
                    smem_iterator_A_.store(buf_a);
                    smem_iterator_B_.store(buf_b);

                    __syncthreads();

                    ++smem_iterator_A_;
                    ++smem_iterator_B_;

                    if(smem_write_stage_idx){
                        smem_iterator_A_.add_tile_offset({0, -kStages});
                        smem_iterator_B_.add_tile_offset({-kStages, 0});
                    }
                    else{
                        warp_tile_iterator_A_.add_tile_offset(
                            {0, -kStages * Policy::kPartitionsK * kWarpGemmIterations});
                        warp_tile_iterator_B_.add_tile_offset(
                            {-kStages * Policy::kPartitionsK * kWarpGemmIterations, 0});
                    }
                    smem_write_stage_idx ^= 1;
                }
                warp_tile_iterator_A_.set_kgroup_index((warp_mma_k+1)%kWarpGemmIterations);
                warp_tile_iterator_B_.set_kgroup_index((warp_mma_k+1)%kWarpGemmIterations);
                
                warp_tile_iterator_A_.load(wbuf_a[!(warp_mma_k & 1)]);
                warp_tile_iterator_B_.load(wbuf_b[!(warp_mma_k & 1)]);
                ++warp_tile_iterator_A_;
                ++warp_tile_iterator_B_;

                if(warp_mma_k == 0){
                    iterator_A.load(buf_a);
                    iterator_B.load(buf_b);
                    ++iterator_A;
                    ++iterator_B;
                    iterator_A.clear_mask(gemm_k_iterations <= 2);
                    iterator_B.clear_mask(gemm_k_iterations <= 2);
                }
                warp_mma(Accum[now_ite / group_per], wbuf_a[warp_mma_k & 1], wbuf_b[warp_mma_k & 1], Accum[now_ite / group_per]);
            }
            ++now_ite;
        }

    }

};


}
