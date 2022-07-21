#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/pitch_linear.h"


namespace SMEM_ITERATOR{

template<typename tbShape, typename Type, typename Layout, int AdvRank, typename TMap>
struct RegularTileIterator;

template<typename tbShape_, typename Type, typename TMap>
struct RegularTileIterator<tbShape_, Type, cutlass::layout::ColumnMajor, 1, TMap>{
    using tbShape = cutlass::PitchLinearShape<tbShape_::kRow, tbShape_::kColumn>;

    using AccessType = cutlass::AlignedArray<Type, TMap::kElementsPerAccess>;

    using Fragment = cutlass::Array<Type, TMap::Iterations::kCount>;
    unsigned char* pointer;
    int stride, inc_stride, inc_advance;
    
    CUTLASS_HOST_DEVICE
    RegularTileIterator(cutlass::TensorRef<Type, cutlass::layout::ColumnMajor> ref, int tid){
        pointer = ((unsigned char*)ref.data()) + ref.offset(TMap::initial_offset(tid)) * sizeof(Type);
        stride = ref.stride()[0];
        inc_stride = stride * sizeof(Type) * TMap::Delta::kStrided;
        inc_advance = stride * sizeof(Type) * tbShape::kStrided;
    }

    CUTLASS_HOST_DEVICE
    RegularTileIterator(){};
    
    CUTLASS_HOST_DEVICE
    void store(Fragment& frag){
        AccessType* piece = (AccessType*)&frag;
        unsigned char* pnt = pointer;
        for(int s=0; s<TMap::Iterations::kStrided; ++s){
            AccessType* src = (AccessType*) pnt;
            for(int c=0; c<TMap::Iterations::kContiguous; ++c){
                int idx = c + TMap::Iterations::kContiguous * s;
                src[c * TMap::Delta::kContiguous] = piece[idx];
            }
            pnt += inc_stride;
        }
    }

    CUTLASS_HOST_DEVICE
    void operator ++(){
        pointer += inc_advance;
    }

    CUTLASS_HOST_DEVICE
    void add_tile_offset(cutlass::layout::ColumnMajor::TensorCoord const& coord){
        pointer += sizeof(Type) * (coord.row() * tbShape::kContiguous + coord.column() * tbShape::kStrided * stride);
    }
};

template<typename tbShape, typename Type, typename TMap>
struct RegularTileIterator<tbShape, Type, cutlass::layout::RowMajor, 0, TMap>{
    using Underlying = RegularTileIterator< 
            cutlass::MatrixShape<tbShape::kColumn, tbShape::kRow>,
            Type, cutlass::layout::ColumnMajor, 1, TMap>;
    
    using Fragment = typename Underlying::Fragment;
    using AccessType = typename Underlying::AccessType;

    Underlying RTIterator;

    CUTLASS_HOST_DEVICE
    RegularTileIterator(){};

    CUTLASS_HOST_DEVICE
    RegularTileIterator(cutlass::TensorRef<Type, cutlass::layout::RowMajor> ref, int tid):
            RTIterator({ref.data(), ref.stride()}, tid){};
    
    CUTLASS_HOST_DEVICE
    void operator ++(){
        ++RTIterator;
    }

    CUTLASS_HOST_DEVICE
    void add_tile_offset(cutlass::layout::RowMajor::TensorCoord coord){
        RTIterator.add_tile_offset({coord.column(), coord.row()});
    }

    CUTLASS_HOST_DEVICE
    void store(Fragment& frag){
        RTIterator.store(frag);
    }
};

}