#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include <iostream>

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


using RowMajor = cutlass::layout::RowMajor;
using ColumnMajor = cutlass::layout::ColumnMajor;
using half_t = cutlass::half_t;

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

template<typename T, typename MAJOR>
struct MakeMatrix{
    static cudaError_t alloc(T*&pm, T*&dpm, int a, int b, int filltype=1){
        MAJOR mj = LAYOUT<MAJOR>(a, b);
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
        }else if (filltype == 4)
        {
            int wia;
            for(wi=0; wi<a; ++wi)
                for(wia=0; wia<b; ++wia)
                    if(wia > wi)
                        pm[mj((cutlass::MatrixCoord){wi, wia})] = 1. / (1. + wi + wia);
                    else if(wia <wi)
                        pm[mj((cutlass::MatrixCoord){wi, wia})] = -1. / (1. + wi + wia);
                    else
                        pm[mj((cutlass::MatrixCoord){wi, wia})] = 1;
        }
        cudaMemcpy(dpm, pm, a*b*item_size, cudaMemcpyHostToDevice);
        return err_t;
    }
    static void print(T*pm, T*dpm, int a, int b){
        MAJOR mj = LAYOUT<MAJOR>(a, b);
        cudaMemcpy(pm, dpm, a*b*sizeof(T), cudaMemcpyDeviceToHost);
        int wi, wia;
        for(wi=0; wi<a; ++wi){
            std::cout<<"| ";
            for(wia=0; wia<b; ++wia)
                std::cout<<pm[mj({wi, wia})]<<' ';
            std::cout<<std::endl;
        }
    }
    static bool compare(T* pa, T* dpa, T* pb, T* dpb, int a, int b, double maxeps_rate){
        MAJOR mj = LAYOUT<MAJOR>(a, b);
        cudaMemcpy(pa, dpa, a*b*sizeof(T), cudaMemcpyDeviceToHost);
        cudaMemcpy(pb, dpb, a*b*sizeof(T), cudaMemcpyDeviceToHost);
        int wi,wia;
        auto abs = [](T a){return a>0? a:-a;};
        auto max = [](T a, T b){return a>b? a:b;};
        for(wi=0; wi<a; ++wi)
            for(wia=0; wia<b; ++wia){
                T va = pa[mj({wi, wia})], vb = pb[mj({wi, wia})];
                double max_eps = abs(max(va, vb)) * maxeps_rate;
                if (abs(va-vb) > max_eps){
                    std::cout<<"Not Match!  ";
                    std::cout<<'<'<<wi<<','<<wia<<"> ";
                    std::cout<<abs(va - vb)<<' '<<va<<' '<<vb<<std::endl;
                    return false;
                } 
            }
        std::cout<<"Match!"<<std::endl;
        return true;
    }
};

using cutlass::gemm::GemmShape;
using cutlass::epilogue::thread::LinearCombination;
using cutlass::sizeof_bits;
using cutlass::tfloat32_t;
struct TestingBase{
    static int NumTest;
    TestingBase(){};
    virtual void Test(){};
};
int TestingBase::NumTest;

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
    Testing(int m=168, int n=248, int k=104){
        M = m;
        N = n;
        K = k;
        MakeMatrix<TA, LA>::alloc(pA, dpA, m, k, 2);
        MakeMatrix<TB, LB>::alloc(pB, dpB, k, n, 2);
        MakeMatrix<TC, LC>::alloc(pC, dpC, m, n, 2);
        MakeMatrix<TD, LC>::alloc(pDa, dpDa, m, n, 1);
        MakeMatrix<TD, LC>::alloc(pDb, dpDb, m, n, 1);
    }

    void Test(){
        TC one((float)1.);
        
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
        MakeMatrix<TD, LC>::compare(pDa, dpDa, pDb, dpDb, M, N, (half_t)0.001);
    }

};