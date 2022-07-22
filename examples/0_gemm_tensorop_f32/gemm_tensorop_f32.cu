#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"


using LayoutABCD = cutlass::layout::RowMajor;
using TypeAPCD = float;
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
ElementOutput *pD, *dpD;

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
    TD, cutlass::arch::OpClassTensorOp>;



int main(){
    return 0;
}