

//#define PRINT
//#define PRINTLOAD
//#define PRINTOFFSET
#define PRINTSPEC gemm::Operand::kA

#include "my_helper.h"
#include <vector>

////////////////////////////0:  nothing test////////////////////////////////
using TEST0 = Testing<cutlass::half_t, RowMajor, half_t, RowMajor, half_t, RowMajor, half_t,
        GemmShape<128, 128, 32>, GemmShape<64, 64, 32>, GemmShape<16, 8, 8>>;               

///////////////////////////////1-9:   low n, k with preset ROW COL ROW between diff Linear>
using TEST1 = Testing<half_t, RowMajor, half_t, ColumnMajor, half_t, RowMajor, half_t,
        GemmShape<32, 16, 16>, GemmShape<32, 16, 16>, GemmShape<16, 8, 8>>;
using TEST2 = Testing<half_t, RowMajor, half_t, ColumnMajor, half_t, RowMajor, half_t,
        GemmShape<32, 16, 16>, GemmShape<32, 16, 16>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t ,half_t> >;
using TEST3 = Testing<half_t, RowMajor, half_t, ColumnMajor, half_t, RowMajor, half_t,
        GemmShape<32, 16, 16>, GemmShape<32, 16, 16>, GemmShape<16, 8, 8>, LinearCombination<half_t, 2, half_t ,half_t> >;
using TEST4 = Testing<half_t, RowMajor, half_t, ColumnMajor, half_t, RowMajor, half_t,
        GemmShape<32, 8, 32>, GemmShape<32, 8, 32>, GemmShape<16, 8, 8>, LinearCombination<half_t, 2, half_t ,half_t> >;
using TEST5 = Testing<half_t, RowMajor, half_t, ColumnMajor, half_t, RowMajor, half_t,
        GemmShape<32, 32, 32>, GemmShape<32, 32, 32>, GemmShape<16, 8, 8>>;
using TEST6 = Testing<half_t, RowMajor, half_t, ColumnMajor, half_t, RowMajor, half_t,
        GemmShape<32, 32, 16>, GemmShape<32, 32, 16>, GemmShape<16, 8, 8>>;
using TEST7 = Testing<half_t, RowMajor, half_t, ColumnMajor, half_t, RowMajor, half_t,
        GemmShape<32, 16, 32>, GemmShape<32, 16, 32>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t, half_t> >;
using TEST8 = Testing<half_t, RowMajor, half_t, ColumnMajor, half_t, RowMajor, half_t,
        GemmShape<64, 16, 32>, GemmShape<32, 16, 32>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t, half_t> >;
using TEST9 = Testing<half_t, RowMajor, half_t, ColumnMajor, half_t, RowMajor, half_t,
        GemmShape<128, 16, 64>, GemmShape<32, 16, 64>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t, half_t> >;


///////////////////////////10-  low n, k  with my ROW ROW ROW///////////////////////////////

using TEST10 = Testing<half_t, RowMajor, half_t, RowMajor, half_t, RowMajor, half_t,
        GemmShape<16, 16, 16>, GemmShape<16, 16, 16>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t ,half_t> >;
using TEST11 = Testing<half_t, RowMajor, half_t, RowMajor, half_t, RowMajor, half_t,
        GemmShape<32, 16, 32>, GemmShape<32, 16, 32>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t ,half_t> >;
using TEST12 = Testing<half_t, RowMajor, half_t, RowMajor, half_t, RowMajor, half_t,
        GemmShape<64, 16, 32>, GemmShape<32, 16, 32>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t ,half_t> >;
using TEST13 = Testing<half_t, RowMajor, half_t, RowMajor, half_t, RowMajor, half_t,
        GemmShape<32, 8, 32>, GemmShape<32, 8, 32>, GemmShape<16, 8, 8>, LinearCombination<half_t, 2, half_t ,half_t> >;
using TEST14 = Testing<half_t, RowMajor, half_t, RowMajor, half_t, RowMajor, half_t,
        GemmShape<128, 16, 64>, GemmShape<32, 16, 64>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t ,half_t> >;
using TEST15 = Testing<half_t, RowMajor, half_t, RowMajor, half_t, RowMajor, half_t,
        GemmShape<64, 8, 64>, GemmShape<32, 8, 64>, GemmShape<16, 8, 8>, LinearCombination<half_t, 2, half_t ,half_t> >;
using TEST16 = Testing<half_t, RowMajor, half_t, RowMajor, half_t, RowMajor, half_t,
        GemmShape<32, 8, 64>, GemmShape<32, 8, 64>, GemmShape<16, 8, 8>, LinearCombination<half_t, 2, half_t ,half_t> >;

        //// n = 8 or 16  passed
        //next:  n = 32
using TEST17 = Testing<half_t, RowMajor, half_t, RowMajor, half_t, RowMajor, half_t,
        GemmShape<32, 32, 16>, GemmShape<32, 32, 16>, GemmShape<16, 8, 8>>;
using TEST18 = Testing<half_t, RowMajor, half_t, RowMajor, half_t, RowMajor, half_t,
        GemmShape<32, 32, 32>, GemmShape<32, 32, 32>, GemmShape<16, 8, 8>>;
using TEST19 = Testing<half_t, RowMajor, half_t, RowMajor, half_t, RowMajor, half_t,
        GemmShape<64, 32, 32>, GemmShape<32, 32, 32>, GemmShape<16, 8, 8>>;
using TEST20 = Testing<half_t, RowMajor, half_t, RowMajor, half_t, RowMajor, half_t,
        GemmShape<128, 32, 64>, GemmShape<32, 32, 64>, GemmShape<16, 8, 8>>;

///////////////////////////20-  low n, k  with other type///////////////////////////////

using TEST21 = Testing<half_t, RowMajor, half_t, ColumnMajor, float, RowMajor, float,
        GemmShape<32, 16, 16>, GemmShape<32, 16, 16>, GemmShape<16, 8, 8>>;

using TEST22 = Testing<half_t, RowMajor, half_t, RowMajor, float, RowMajor, float,
        GemmShape<32, 16, 16>, GemmShape<32, 16, 16>, GemmShape<16, 8, 8>>;
////////////////////////////f16 f16 f32 f32 be the same as f16^4///////////////////

using TEST23 = Testing<int8_t, RowMajor, int8_t, ColumnMajor, int32_t, RowMajor, int32_t,
        GemmShape<32, 16, 32>, GemmShape<32, 16, 32>, GemmShape<8, 8, 16>>;


/////////////////////////////a extra major///////////////////////////////////////

using TEST24 = Testing<half_t, ColumnMajor, half_t, ColumnMajor, half_t, RowMajor, half_t,
        GemmShape<32, 16, 16>, GemmShape<32, 16, 16>, GemmShape<16, 8, 8>>;
using TEST25 = Testing<half_t, ColumnMajor, half_t, ColumnMajor, half_t, RowMajor, half_t,
        GemmShape<32, 16, 16>, GemmShape<32, 16, 16>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t ,half_t> >;
using TEST26 = Testing<half_t, ColumnMajor, half_t, ColumnMajor, half_t, RowMajor, half_t,
        GemmShape<16, 16, 16>, GemmShape<16, 16, 16>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t ,half_t> >;

using TEST27 = Testing<half_t, ColumnMajor, half_t, RowMajor, half_t, RowMajor, half_t,
        GemmShape<32, 16, 16>, GemmShape<32, 16, 16>, GemmShape<16, 8, 8>>;
using TEST28 = Testing<half_t, ColumnMajor, half_t, RowMajor, half_t, RowMajor, half_t,
        GemmShape<32, 16, 16>, GemmShape<32, 16, 16>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t ,half_t> >;
using TEST29 = Testing<half_t, ColumnMajor, half_t, RowMajor, half_t, RowMajor, half_t,
        GemmShape<16, 16, 16>, GemmShape<16, 16, 16>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t ,half_t> >;

////////////////////////////////////////////////////////////c col/////////////////////////////////////////

using TEST30 = Testing<half_t, ColumnMajor, half_t, RowMajor, half_t, ColumnMajor, half_t,
        GemmShape<16, 16, 16>, GemmShape<16, 16, 16>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t ,half_t> >;
using TEST31 = Testing<half_t, ColumnMajor, half_t, ColumnMajor, half_t, ColumnMajor, half_t,
        GemmShape<16, 16, 16>, GemmShape<16, 16, 16>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t ,half_t> >;
using TEST32 = Testing<half_t, RowMajor, half_t, RowMajor, half_t, ColumnMajor, half_t,
        GemmShape<16, 16, 16>, GemmShape<16, 16, 16>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t ,half_t> >;
using TEST33 = Testing<half_t, RowMajor, half_t, ColumnMajor, half_t, ColumnMajor, half_t,
        GemmShape<16, 16, 16>, GemmShape<16, 16, 16>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t ,half_t> >;

//////////////////////////////////////////////////////////multi warps/////////////////////////////////////////////

using TEST34 = Testing<half_t, RowMajor, half_t, RowMajor, half_t, RowMajor, half_t,
        GemmShape<32, 16, 32>, GemmShape<16, 16, 32>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t ,half_t> >;
using TEST35 = Testing<half_t, ColumnMajor, half_t, ColumnMajor, half_t, RowMajor, half_t,
        GemmShape<16, 64, 64>, GemmShape<16, 64, 64>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t ,half_t> >;


////////////////////////////////////////////////////////////free warp////////////////////////////////////////////////

using TEST36 = Testing<half_t, RowMajor, half_t, ColumnMajor, half_t, RowMajor, half_t,
        GemmShape<64, 16, 16>, GemmShape<16, 16, 16>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t ,half_t> >;
using TEST37 = Testing<half_t, RowMajor, half_t, ColumnMajor, half_t, RowMajor, half_t,
        GemmShape<32, 16, 16>, GemmShape<16, 16, 16>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t ,half_t> >;
using TEST38 = Testing<half_t, RowMajor, half_t, ColumnMajor, half_t, RowMajor, half_t,
        GemmShape<128, 16, 16>, GemmShape<16, 16, 16>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t ,half_t> >;

///////////////////////////////////////////////////////warps in low channel////////////////////////////

using TEST39 = Testing<half_t, ColumnMajor, half_t, ColumnMajor, half_t, RowMajor, half_t,
        GemmShape<32, 16, 16>, GemmShape<16, 16, 16>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t ,half_t> >;
using TEST40 = Testing<half_t, ColumnMajor, half_t, ColumnMajor, half_t, RowMajor, half_t,
        GemmShape<64, 16, 16>, GemmShape<32, 16, 16>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t ,half_t> >;
using TEST41 = Testing<half_t, ColumnMajor, half_t, ColumnMajor, half_t, RowMajor, half_t,
        GemmShape<32, 32, 16>, GemmShape<16, 16, 16>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t ,half_t> >;
///////////////////////////////////////////////////all typeof with warps///////////////////////////


using TEST42 = Testing<half_t, ColumnMajor, half_t, ColumnMajor, half_t, RowMajor, half_t,
        GemmShape<16, 32, 16>, GemmShape<16, 16, 16>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t ,half_t> >;
using TEST43 = Testing<half_t, ColumnMajor, half_t, ColumnMajor, half_t, RowMajor, half_t,
        GemmShape<32, 16, 16>, GemmShape<16, 16, 16>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t ,half_t> >;
using TEST44 = Testing<half_t, ColumnMajor, half_t, ColumnMajor, half_t, RowMajor, half_t,
        GemmShape<32, 32, 16>, GemmShape<16, 16, 16>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t ,half_t> >;
using TEST45 = Testing<half_t, RowMajor, half_t, ColumnMajor, half_t, RowMajor, half_t,
        GemmShape<16, 32, 16>, GemmShape<16, 16, 16>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t ,half_t> >;
using TEST46 = Testing<half_t, RowMajor, half_t, ColumnMajor, half_t, RowMajor, half_t,
        GemmShape<32, 16, 16>, GemmShape<16, 16, 16>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t ,half_t> >;
using TEST47 = Testing<half_t, RowMajor, half_t, ColumnMajor, half_t, RowMajor, half_t,
        GemmShape<32, 32, 16>, GemmShape<16, 16, 16>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t ,half_t> >;
using TEST48 = Testing<half_t, ColumnMajor, half_t, RowMajor, half_t, RowMajor, half_t,
        GemmShape<16, 32, 16>, GemmShape<16, 16, 16>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t ,half_t> >;
using TEST49 = Testing<half_t, ColumnMajor, half_t, RowMajor, half_t, RowMajor, half_t,
        GemmShape<32, 16, 16>, GemmShape<16, 16, 16>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t ,half_t> >;
using TEST50 = Testing<half_t, ColumnMajor, half_t, RowMajor, half_t, RowMajor, half_t,
        GemmShape<32, 32, 16>, GemmShape<16, 16, 16>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t ,half_t> >;
using TEST51 = Testing<half_t, RowMajor, half_t, RowMajor, half_t, RowMajor, half_t,
        GemmShape<16, 32, 16>, GemmShape<16, 16, 16>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t ,half_t> >;
using TEST52 = Testing<half_t, RowMajor, half_t, RowMajor, half_t, RowMajor, half_t,
        GemmShape<32, 16, 16>, GemmShape<16, 16, 16>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t ,half_t> >;
using TEST53 = Testing<half_t, RowMajor, half_t, RowMajor, half_t, RowMajor, half_t,
        GemmShape<32, 32, 16>, GemmShape<16, 16, 16>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t ,half_t> >;

/////////////////////////////////////////////////////large tbShape///////////////////////////////////////////////
using TEST54 = Testing<half_t, RowMajor, half_t, RowMajor, half_t, RowMajor, half_t,
        GemmShape<64, 16, 16>, GemmShape<16, 16, 16>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t ,half_t> >;
using TEST55 = Testing<half_t, ColumnMajor, half_t, RowMajor, half_t, RowMajor, half_t,
        GemmShape<64, 16, 16>, GemmShape<16, 16, 16>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t ,half_t> >;
using TEST56 = Testing<half_t, RowMajor, half_t, ColumnMajor, half_t, RowMajor, half_t,
        GemmShape<64, 16, 16>, GemmShape<16, 16, 16>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t ,half_t> >;
using TEST57 = Testing<half_t, ColumnMajor, half_t, ColumnMajor, half_t, RowMajor, half_t,
        GemmShape<64, 16, 16>, GemmShape<16, 16, 16>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t ,half_t> >;

using TEST58 = Testing<half_t, RowMajor, half_t, RowMajor, half_t, RowMajor, half_t,
        GemmShape<64, 64, 16>, GemmShape<16, 16, 16>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t ,half_t> >;
using TEST59 = Testing<half_t, ColumnMajor, half_t, RowMajor, half_t, RowMajor, half_t,
        GemmShape<64, 64, 16>, GemmShape<16, 16, 16>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t ,half_t> >;
using TEST60 = Testing<half_t, RowMajor, half_t, ColumnMajor, half_t, RowMajor, half_t,
        GemmShape<64, 64, 16>, GemmShape<16, 16, 16>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t ,half_t> >;
using TEST61 = Testing<half_t, ColumnMajor, half_t, ColumnMajor, half_t, RowMajor, half_t,
        GemmShape<64, 64, 16>, GemmShape<16, 16, 16>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t ,half_t> >;

using TEST62 = Testing<half_t, RowMajor, half_t, RowMajor, half_t, RowMajor, half_t,
        GemmShape<128, 16, 16>, GemmShape<16, 16, 16>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t ,half_t> >;
using TEST63 = Testing<half_t, ColumnMajor, half_t, RowMajor, half_t, RowMajor, half_t,
        GemmShape<128, 16, 16>, GemmShape<16, 16, 16>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t ,half_t> >;
using TEST64 = Testing<half_t, RowMajor, half_t, ColumnMajor, half_t, RowMajor, half_t,
        GemmShape<128, 16, 16>, GemmShape<16, 16, 16>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t ,half_t> >;
using TEST65 = Testing<half_t, ColumnMajor, half_t, ColumnMajor, half_t, RowMajor, half_t,
        GemmShape<128, 16, 16>, GemmShape<16, 16, 16>, GemmShape<16, 8, 8>, LinearCombination<half_t, 4, half_t ,half_t> >;



///////////////////////////////////////other type, they originally support row col only and don't need classed i modified////////////////////////////

using TEST66 = TestOffical<int8_t, RowMajor, int8_t, ColumnMajor, int32_t, RowMajor, int32_t>;
using TEST67 = TestOffical<cutlass::int4b_t, RowMajor, cutlass::int4b_t, ColumnMajor, int32_t, RowMajor, int32_t>;


std::vector<TestingBase*> Tests = {
    //new TEST0(), 
    
    new TEST1(), 
    new TEST2(), 
    new TEST3(), 
    new TEST4(),
    new TEST5(),
    new TEST6(),
    new TEST7(),
    new TEST8(),
    new TEST9(),
    new TEST10(),
    new TEST11(),
    new TEST12(),
    new TEST13(),
    new TEST14(),
    new TEST15(),
    new TEST16(),
    new TEST17(),
    new TEST18(),
    new TEST19(),
    new TEST20(),
    new TEST21(),
    new TEST22(),
    new TEST23(16*8, 16*5, 16*4),
    new TEST24(),
    new TEST25(),
    new TEST26(),
    new TEST27(),
    new TEST28(),
    new TEST29(),
    new TEST30(),
    new TEST31(),
    new TEST32(),
    new TEST33(),
    new TEST34(),
    new TEST35(),
    new TEST36(),
    new TEST37(),
    new TEST38(),
    new TEST39(),
    new TEST40(),
    new TEST41(),
    new TEST42(),
    new TEST43(),
    new TEST44(),
    new TEST45(),
    new TEST46(),
    new TEST47(),
    new TEST48(),
    new TEST49(),
    new TEST50(),
    new TEST51(),
    new TEST52(),
    new TEST53(),
    new TEST54(),
    new TEST55(),
    new TEST56(),
    new TEST57(),
    new TEST58(),
    new TEST59(),
    new TEST60(),
    new TEST61(),
    new TEST62(),
    new TEST63(),
    new TEST64(),
    new TEST65(),


    new TEST66(16*5, 16*4, 16*6, FILLTYPE::sRange),
    //new TEST67(128, 128, 128, FILLTYPE::sRange),

    
};

template<typename TEST>
void View(){
        SHOW_TYPE(typename TEST::GEMM::GemmKernel::Mma::Operator::IteratorA::Base::Layout);
        SHOW_TYPE(typename TEST::GEMM::GemmKernel::Mma::Operator::IteratorA::Base::Policy::LdsmShape);
        SHOW_TYPE(typename TEST::GEMM::GemmKernel::Mma::Operator::IteratorA::Base::Shape);
        SHOW_TYPE(typename TEST::GEMM::GemmKernel::Mma::Operator::IteratorA::Base::InstructionShape);
        
        std::cout<< TEST::GEMM::GemmKernel::Mma::Operator::IteratorA::Base::ME<<std::endl;
}

void View2(){
        SHOW_TYPE(TEST67::GEMM_ref::GemmKernel::Mma::Operator::IteratorA);
        
        SHOW_TYPE(TEST67::GEMM_ref::GemmKernel::Mma::Operator::IteratorB);
}


int main(){
    srand(time(nullptr));

    View2();
    for(auto test : Tests)
        test->Test();
    return 0;
}