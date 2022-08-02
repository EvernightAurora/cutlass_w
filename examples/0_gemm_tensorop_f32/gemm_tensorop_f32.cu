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

using LayoutABCD = cutlass::layout::RowMajor;
using TypeABCD = cutlass::half_t;
using TA = TypeABCD;
using TB = TypeABCD;
using TC = TypeABCD;
using TD = TypeABCD;
using LayoutA = LayoutABCD;
using LayoutB = LayoutABCD;
using LayoutC = LayoutABCD;
using LayoutD = LayoutABCD;


const int M=333, N=222, K=99;
TA* pA, *dpA;
TB *pB, *dpB;
TC *pC, *dpC;
TD *pD, *dpD;

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
                    std::cout<<'<'<<wi<<','<<wia<<'>'<<std::endl;
                    std::cout<<abs(va - vb)<<' '<<va<<' '<<vb<<std::endl;
                    return false;
                } 
            }
        std::cout<<"Match!"<<std::endl;
        return true;
    }
};

void Init_MM(){
    MakeMatrix<TA, LayoutA>::alloc(pA, dpA, M, K, LayoutA(K), 3);
    MakeMatrix<TB, LayoutB>::alloc(pB, dpB, K, N, LayoutB(N), 3);
    MakeMatrix<TC, LayoutC>::alloc(pC, dpC, M, N, LayoutC(N), 0);
    MakeMatrix<TD, LayoutD>::alloc(pD, dpD, M, N, LayoutD(N), 0);        
}


using Gemm = cutlass::gemm::device::Gemm<
    TA, LayoutA, TB, LayoutB, TC, LayoutC,
    TD, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>>;

void Show(){
    SHOW_TYPE(Gemm::DefaultType::MmaType::ThreadblockMma::Operator::IteratorC::Policy::MmaIterations);
    std::cout<<(Gemm::DefaultType::MmaType::ThreadblockMma::Operator::IteratorC::Fragment::kElements)<<std::endl;
    SHOW_TYPE(Gemm::DefaultType::Epilogue::SharedStorage::StorageShape);
    SHOW_TYPE(Gemm::DefaultType::Mma::Operator::IteratorC::Shape);
    std::cout<<"Gemm::EpilogueShell:  "<<Gemm::DefaultType::EpilogueShell::SIGN_LINE<<std::endl;
    std::cout<<"Gemm::EpilogueShell::OutputTileIterator:  "<<Gemm::DefaultType::EpilogueShell::OutputTileIterator::SIGN_LINE<<std::endl;
    std::cout<<"Gemm::EpilogueShell::OutputTileThreadMapShell:  "<<Gemm::DefaultType::EpilogueShell::OutputTileThreadMapShell::SIGN_LINE<<std::endl;
    std::cout<<"Gemm::EpilogueShell::AccumulatorFragmentIterator::Policy:  "<<Gemm::DefaultType::EpilogueShell::AccumulatorFragmentIterator::Policy::SIGN_LINE<<std::endl;
    std::cout<<"Gemm::EpilogueShell::OutputTileThreadMap::RowArrange::kAccessRow:  "<<Gemm::DefaultType::EpilogueShell::OutputTileThreadMap::Detail::RowArrangement::kAccessRows<<std::endl;
    std::cout<<"Gemm::EpilogueShell::TileIteratorTensorOp:  "<<Gemm::DefaultType::EpilogueShell::WarpTileIterator::SIGN_LINE<<std::endl;
    std::cout<<"Gemm::EpilogueShell::SharedLoadIterator:  "<<Gemm::DefaultType::EpilogueShell::SharedLoadIterator::SIGN_LINE<<std::endl;
    SHOW_TYPE(Gemm::DefaultType::EpilogueShell::OutputTileThreadMap::Iterations);
    SHOW_TYPE(Gemm::DefaultType::EpilogueShell::OutputTileThreadMap::Delta);
    SHOW_TYPE(Gemm::DefaultType::EpilogueShell::OutputTileThreadMap::Detail::WarpPartitions);

    
}

int main(){
    Show();
    return 0;
}