#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/warp/mma_simt_policy.h"


namespace WARP_ITERATOR{

template<typename wpShape, cutlass::gemm::Operand Op, typename Type, typename Major, typename Policy, int _=0, int __=0>
struct WarpIterator;


template<typename wpShape_, typename Type_, typename Policy_, int _, int __>
struct WarpIterator<wpShape_, cutlass::gemm::Operand::kA, Type_, cutlass::layout::ColumnMajor, Policy_, _, __>{
    using wpShape = wpShape_;
    using Policy = Policy_;
    using Type = Type_;

    using Thread_Flow_Size = cutlass::MatrixShape< wpShape::kRow / Policy::WarpShape::kRow, wpShape::kColumn>;
    using Access_Count = cutlass::MatrixShape< Thread_Flow_Size::kRow / Policy::LaneMmaShape::kM, wpShape::kColumn>;

    using Fragment = cutlass::Array<Type, Thread_Flow_Size::kCount>;
    using AccessType = cutlass::Array<Type, Policy::LaneMmaShape::kM>;

    cutlass::TensorRef<AccessType, cutlass::layout::ColumnMajor> ref_;
    CUTLASS_HOST_DEVICE
    WarpIterator(){};

    CUTLASS_HOST_DEVICE
    WarpIterator(cutlass::TensorRef<Type, cutlass::layout::ColumnMajor> ref, int lid){
        #define offset_m  (lid / Policy::WarpShape::kColumn)
        #define offset_n  (lid % Policy::WarpShape::kColumn)
        ref.add_coord_offset({offset_m * Policy::LaneMmaShape::kM, 0});
        ref_.reset(reinterpret_cast<AccessType*>(ref.data()), ref.stride(0) / Policy::LaneMmaShape::kM);
        #undef offset_n
        #undef offset_m
    }

    CUTLASS_HOST_DEVICE
    void operator ++(){
        ref_.add_coord_offset({0, wpShape::kColumn});
    }

    CUTLASS_HOST_DEVICE
    void add_tile_offset(typename cutlass::TensorRef<Type, cutlass::layout::ColumnMajor>::TensorCoord coord){
        ref_.add_coord_offset({coord.row() * wpShape::kRow / Policy::LaneMmaShape::kM, coord.column() * wpShape::kColumn});
    }

    CUTLASS_HOST_DEVICE
    void set_kgroup_index(int){};

    CUTLASS_HOST_DEVICE
    void load(Fragment&dest_){
        static_assert(Access_Count::kColumn == 1);
        AccessType* dest = reinterpret_cast<AccessType*>(&dest_);
        for(int r=0; r<Access_Count::kRow; ++r)
            for(int c=0; c<Access_Count::kColumn; ++c)
                dest[r + Access_Count::kRow * c] = *(ref_.data() + ref_.offset({r*Policy::WarpShape::kRow, c * wpShape::kColumn}));
    }
};


template<typename wpShape_, typename Type_, typename Policy_, int _, int __>
struct WarpIterator<wpShape_, cutlass::gemm::Operand::kB, Type_, cutlass::layout::RowMajor, Policy_, _, __>{
    using wpShape = wpShape_;
    using Policy = Policy_;
    using Type = Type_;

    using Thread_Flow_Size = cutlass::MatrixShape< wpShape::kRow, wpShape::kColumn / Policy::WarpShape::kColumn>;
    using Access_Count = cutlass::MatrixShape< wpShape::kRow, Thread_Flow_Size::kColumn / Policy::LaneMmaShape::kN>;

    using Fragment = cutlass::Array<Type, Thread_Flow_Size::kCount>;
    using AccessType = cutlass::Array<Type, Policy::LaneMmaShape::kN>;

    cutlass::TensorRef<AccessType, cutlass::layout::RowMajor> ref_;
    CUTLASS_HOST_DEVICE
    WarpIterator(){};

    CUTLASS_HOST_DEVICE
    WarpIterator(cutlass::TensorRef<Type, cutlass::layout::RowMajor> ref, int lid){
        #define offset_m  (lid / Policy::WarpShape::kColumn)
        #define offset_n  (lid % Policy::WarpShape::kColumn)
        ref.add_coord_offset({0, offset_n * Policy::LaneMmaShape::kN});
        ref_.reset(reinterpret_cast<AccessType*>(ref.data()), ref.stride(0) / Policy::LaneMmaShape::kN);
        #undef offset_n
        #undef offset_m
    }

    CUTLASS_HOST_DEVICE
    void operator ++(){
        ref_.add_coord_offset({wpShape::kRow, 0});
    }

    CUTLASS_HOST_DEVICE
    void add_tile_offset(cutlass::layout::RowMajor::TensorCoord coord){
        ref_.add_coord_offset({coord.row() * wpShape::kRow, coord.column() * wpShape::kColumn / Policy::LaneMmaShape::kN});
    }

    CUTLASS_HOST_DEVICE
    void set_kgroup_index(int){};

    CUTLASS_HOST_DEVICE
    void load(Fragment&dest_){
        AccessType* dest = reinterpret_cast<AccessType*>(&dest_);
        static_assert(Access_Count::kRow == 1);
        for(int r=0; r<Access_Count::kRow; ++r)
            for(int c=0; c<Access_Count::kColumn; ++c)
                dest[r*Access_Count::kColumn + c] = *(ref_.data() + ref_.offset({r * wpShape::kRow, c*Policy::WarpShape::kColumn}));
    }
};


template<typename wpShape_, typename Type_, typename Policy_, int _, int __>
struct WarpIterator<wpShape_, cutlass::gemm::Operand::kC, Type_, cutlass::layout::RowMajor, Policy_, _, __>{
    using wpShape = wpShape_;
    using Policy = Policy_;
    using Type = Type_;
    
    using Thread_Flow_Size = cutlass::MatrixShape< wpShape::kRow / Policy::WarpShape::kRow, wpShape::kColumn / Policy::WarpShape::kColumn>;
    using Iteration_Cnt = cutlass::MatrixShape< Thread_Flow_Size::kRow / Policy::LaneMmaShape::kM, Thread_Flow_Size::kRow / Policy::LaneMmaShape::kN>;

    using Delta = cutlass::MatrixShape<Thread_Flow_Size::kRow / Iteration_Cnt::kRow, Thread_Flow_Size::kColumn / Iteration_Cnt::kColumn>;

    using Fragment = cutlass::Array<Type, Thread_Flow_Size::kCount>;

    using LocalTensorRef = cutlass::TensorRef<Type, cutlass::layout::RowMajor>;
    
    LocalTensorRef ref_;

    CUTLASS_HOST_DEVICE
    WarpIterator(){};

    CUTLASS_HOST_DEVICE
    WarpIterator(LocalTensorRef ref, int lid){
        #define offset_m  (lid / Policy::WarpShape::kColumn)
        #define offset_n  (lid % Policy::WarpShape::kColumn)
        ref.add_coord_offset({offset_m * Policy::LaneMmaShape::kM, offset_n * Policy::LaneMmaShape::kN});
        ref_.reset(ref.data(), ref.stride(0));
        #undef offset_n
        #undef offset_m
    }

    CUTLASS_HOST_DEVICE
    void operator ++(){
        ref_.add_coord_offset({wpShape::kRow, 0});
    }

    CUTLASS_HOST_DEVICE
    void add_tile_offset(typename LocalTensorRef::TensorCoord const& coord){
        ref_.add_coord_offset({coord.row() * wpShape::kRow, coord.column() * wpShape::kColumn});
    }

    CUTLASS_HOST_DEVICE
    void load(Fragment& frag){
        Type* piece = reinterpret_cast<Type*>(&frag);
        for(int mma_m=0; mma_m<Iteration_Cnt::kRow; ++mma_m)
            for(int mma_n=0; mma_n<Iteration_Cnt::kColumn; ++mma_n)
                for(int m=0; m<Policy::LaneMmaShape::kM; ++m)
                    for(int n=0; n<Policy::LaneMmaShape::kN; ++n)
                        piece[(mma_m * Policy::LaneMmaShape::kM + m) * Thread_Flow_Size::kColumn + (mma_n * Policy::LaneMmaShape::kN + n)] = 
                            *(ref_.data() + ref_.offset({mma_m * Delta::kRow + m, mma_n * Delta::kColumn + n}));
    }

};
}