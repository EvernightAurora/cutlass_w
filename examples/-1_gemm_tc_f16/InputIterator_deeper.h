#include "cutlass/transform/threadblock/predicated_tile_access_iterator.h"

namespace INPUT_ITERATOR{

template<typename tbShape, typename Type, 
    typename Layout,
    int kAdvanceRank,
    typename TMap,
    int AccessSize = 1,
    bool Gather=false>
struct UnpackedPTIterator;


template<typename tbShape_, typename Type_,
    int kAdvancedRank_,
    typename TMap_>
struct UnpackedPTIterator<tbShape_, Type_, cutlass::layout::RowMajor, kAdvancedRank_, TMap_, 1, false>{
    using Type = Type_;
    using tbShape = cutlass::layout::PitchLinearShape<tbShape_::kColumn, tbShape_::kRow>;
    using TMap = TMap_;

    using TensorRef = cutlass::TensorRef<Type, cutlass::layout::RowMajor>;

    static int const kAdvanceRank = !(kAdvancedRank_);              // !!!!!!
    using AccessType = cutlass::AlignedArray<Type, 1, sizeof(Type)>;
    

    
    using Fragment = cutlass::Array<Type, TMap::Iterations::kCount * TMap::kElementsPerAccess>; //4
    
    struct Params: public cutlass::transform::threadblock::PredicatedTileAccessIteratorParams{
        using Base = cutlass::transform::threadblock::PredicatedTileAccessIteratorParams;
        CUTLASS_HOST_DEVICE
        Params(cutlass::layout::RowMajor layout): PredicatedTileAccessIteratorParams(layout.stride(0),
                cutlass::transform::threadblock::MakePredicatedTileAccessIteratorDesc<
                tbShape, 
                Type, cutlass::layout::PitchLinear, kAdvanceRank, TMap>()()){}
        CUTLASS_HOST_DEVICE
        Params(){};
    };

    Params params;

    //////////////////////////INNER PTAI
    uint32_t valid_mask;
    int iter_stride;
    unsigned char* pointer;
    cutlass::layout::PitchLinear::TensorCoord matrix_shape, thread_offset, residue_offset;
    bool has_residual;

    CUTLASS_HOST_DEVICE
    void* get(){return pointer;}
    CUTLASS_HOST_DEVICE
    void inner_add(){
        ++iter_stride;
        if(iter_stride >= TMap::Iterations::kStrided){
            iter_stride = 0;
            pointer += params.inc_next_ - params.inc_advance_;          //back to origin pos
        }
        else
            pointer += params.inc_strided_;
    }
    CUTLASS_HOST_DEVICE
    void clear_mask(bool enable=false){
         valid_mask *= !enable;
    }
    CUTLASS_HOST_DEVICE
    bool valid(){
        return !!(valid_mask & (1<<iter_stride));
    }
    CUTLASS_HOST_DEVICE
    void compute_mask(cutlass::layout::PitchLinear::TensorCoord local_shape){
        valid_mask = 0;
        static_assert(TMap::Iterations::kContiguous == 1);
        for(int s=0; s < TMap::Iterations::kStrided; ++s){
            auto pos = cutlass::layout::PitchLinear::TensorCoord({0, s * TMap::Delta::kStrided}) + thread_offset;
            valid_mask |= ((pos.contiguous() < local_shape.contiguous() && pos.strided() < local_shape.strided()) << s);
        }
    }
    CUTLASS_HOST_DEVICE
    void add_pointer_offset(long offset){pointer += offset * sizeof(Type);}
    CUTLASS_HOST_DEVICE
    void set_iteration_index(int iter){iter_stride = iter;}
    CUTLASS_HOST_DEVICE
    void set_residual(int tid, cutlass::layout::PitchLinear::TensorCoord const & tb_offset){
        cutlass::layout::PitchLinear::TensorCoord extends;
        int residual;
        if(!kAdvanceRank){           //A
            residual = (matrix_shape[0] - tb_offset[0]) % tbShape::kContiguous;
            if(!residual)
                residual = tbShape::kContiguous;             //excatly align
            residue_offset = cutlass::make_Coord(residual, 0);
            extends = cutlass::make_Coord(tb_offset[0] + residual, matrix_shape[1]);
        }
        else{                       //B
            residual = (matrix_shape[1] - tb_offset[1]) % tbShape::kStrided;
            if(!residual)
                residual = tbShape::kStrided;
            residue_offset = cutlass::make_Coord(0, residual);
            extends = cutlass::make_Coord(matrix_shape[0], tb_offset[1] + residual);
        }
        thread_offset = tb_offset + TMap::initial_offset(tid);
        compute_mask(extends);
        set_iteration_index(0);
    }

    CUTLASS_HOST_DEVICE
    void operator ++(){         //add_tile_offset

        if(has_residual){
            thread_offset += residue_offset;
            compute_mask(matrix_shape);
            cutlass::layout::PitchLinear layout(params.stride_);
            add_pointer_offset(layout(residue_offset));
        }
        else{
            pointer += params.inc_advance_;
        }
        has_residual = false;
    }


    //Underlying used_underlying;

    CUTLASS_HOST_DEVICE
    UnpackedPTIterator(
        Params const& params,
        Type* pointer,
        cutlass::layout::RowMajor::TensorCoord const& raw_problem_size,
        int tid,
        cutlass::layout::RowMajor::TensorCoord const& raw_tb_offset,
        int const* indices=nullptr
    ):  params(params), pointer((unsigned char*)pointer), 
        matrix_shape(cutlass::layout::PitchLinear::TensorCoord(raw_problem_size.column(), raw_problem_size.row())),
        has_residual(true)
    {
        auto tb_offset = cutlass::layout::PitchLinear::TensorCoord(raw_tb_offset.column(), raw_tb_offset.row());
        set_residual(tid, tb_offset);
        cutlass::layout::PitchLinear layout(params.stride_);
        add_pointer_offset(layout(thread_offset));
    };
    CUTLASS_HOST_DEVICE
    void load(Fragment& fragment){
        AccessType* split = reinterpret_cast<AccessType*> (&fragment);
        static_assert(TMap::Iterations::kContiguous == 1);
        for(int s=0; s<TMap::Iterations::kStrided; ++s){
            //static_assert(Underlying::kAccessesPerVector == 1);
            int idx = s;
            set_iteration_index(idx);
            
            if(valid())
                *(split + s) = *(reinterpret_cast<AccessType*>(get()));
            inner_add();
        }
    }
};


}