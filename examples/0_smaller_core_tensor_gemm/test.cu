#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include <iostream>
#include <vector>
using RowMajor = cutlass::layout::RowMajor;
using ColumnMajor = cutlass::layout::ColumnMajor;
using half_t = cutlass::half_t;
using cutlass::gemm::GemmShape;
using cutlass::epilogue::thread::LinearCombination;
using cutlass::sizeof_bits;

struct TestingBase{
    static int NumTest;
    TestingBase(){};
    virtual void Test(){};
};
int TestingBase::NumTest;

template<typename LAY>
struct LAYOUT;

template<>
struct LAYOUT<RowMajor>{
    RowMajor rm;
    LAYOUT(int a,int b):rm(b){};
    operator RowMajor()const {return rm;};
    int stride(){return rm.stride()[0];}
};
template<>
struct LAYOUT<ColumnMajor>{
    ColumnMajor rm;
    LAYOUT(int a,int b):rm(a){};
    operator ColumnMajor()const {return rm;};
    int stride(){return rm.stride()[0];}
};



template<typename TA, typename LA, typename TB, typename LB, typename TC, typename LC,
            typename TD, typename TBShape, typename WShape, typename IShape,
            typename Epilogue = LinearCombination<TC, 128/sizeof_bits<TC>::value, TD, TD> >
struct Testing: public TestingBase{

    using GEMM = cutlass::gemm::device::Gemm<TA, LA, TB, LB, TC, LC, TD, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75, TBShape, WShape, IShape, Epilogue>;
    using GEMM_ref = cutlass::gemm::device::Gemm<TA, LA, TB, LB, TC, LC, TD, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75>;
    GEMM gemm;
    GEMM_ref gemm_ref;
    TA *pA, *dpA;
    TB *pB, *dpB;
    TC *pC, *dpC;
    TD *pDa, *dpDa, *pDb, *dpDb;
    int M, N, K;
    Testing(int m=80, int n=176, int k=48){
        M=m; N=n; K=k;
        pA = new TA[m*k];
        pB = new TB[k*n];
        pC = new TC[m*n];
        pDa = new TD[m*n];
        pDb = new TD[m*n];
        cudaMalloc(&pA, m*k*2);
        cudaMalloc(&pB, n*k*2);
        cudaMalloc(&pC, m*n*2);
        cudaMalloc(&pDa, m*n*2);
        cudaMalloc(&pDb, m*n*2);
        
    }
    void Test(){
        TD one((float)1.);
        
        typename GEMM::Arguments arg = {
            {M, N, K},
            {dpA, LAYOUT<LA>(M, K).stride()},
            {dpB, LAYOUT<LB>(K, N).stride()},
            {dpC, LAYOUT<LC>(M, N).stride()},
            {dpDa, LAYOUT<LC>(M, N).stride()},
            {one, one}};
        typename GEMM_ref::Arguments arg0 = {
            {M, N, K},
            {dpA, LAYOUT<LA>(M, K).stride()},
            {dpB, LAYOUT<LB>(K, N).stride()},
            {dpC, LAYOUT<LC>(M, N).stride()},
            {dpDb, LAYOUT<LC>(M, N).stride()},
            {one, one}};
        std::cout<<"On Test "<<++NumTest<<" \t";
        auto r_ca = gemm(arg);
        if(r_ca != cutlass::Status::kSuccess)
            std::cout<<"Gemm Run Error! ";
        r_ca = gemm_ref(arg0);
        if(r_ca != cutlass::Status::kSuccess)
            std::cout<<"Ref Gemm Run Error! ";
        auto r_cu = cudaDeviceSynchronize();
        if(r_cu != cudaSuccess)
            std::cout<<"Sync error! ";
        //MakeMatrix<TD, LC>::compare(pDa, dpDa, pDb, dpDb, M, N, (half_t)0.001);
    }
};


using TEST13 = Testing<half_t, RowMajor, half_t, ColumnMajor, half_t, RowMajor, half_t,
        GemmShape<16, 8, 32>, GemmShape<16, 8, 32>, GemmShape<16, 8, 8>, LinearCombination<half_t, 2, half_t ,half_t> >;

int main(){
    srand(11);
    TEST13();

}