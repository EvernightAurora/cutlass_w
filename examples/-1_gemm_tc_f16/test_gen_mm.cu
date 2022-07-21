#include<cuda.h>

#include<iostream>
#include"cutlass/cutlass.h"
#include"cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemm.h"

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
using ColumnMajor = cutlass::layout::ColumnMajor;
using RowMajor = cutlass::layout::RowMajor;
using TA=float;
using TB=float;
using TC=float;
using ElementOutput=float;
using LayoutA = RowMajor;
using LayoutB = RowMajor;
using LayoutC = RowMajor;

using LayoutOutput=LayoutC;

const int M=8, N=8, K=8;
TA* pA, *dpA;
TB *pB, *dpB;
TC *pC, *dpC;
ElementOutput *pD, *dpD;

template<typename T, typename MAJOR>
struct MakeMatrix{
    static cudaError_t alloc(T*&pm, T*&dpm, int a, int b, MAJOR mj, bool fill=true){
        size_t item_size = sizeof(T);
        pm = new T[a * b];
        auto err_t = cudaMalloc((void**)&dpm, a*b*item_size);
        int wi;
        memset(pm, 0, a*b*item_size);
        if(fill)
            for(wi=0;wi<a && wi<b; ++wi)
                pm[mj((cutlass::MatrixCoord){wi, wi})] = 1;
        cudaMemcpy(dpm, pm, a*b*item_size, cudaMemcpyHostToDevice);
        return err_t;
    }
    static void print(T*pm, T*dpm, int a, int b, MAJOR mj){
        cudaMemcpy(pm, dpm, a*b*sizeof(T), cudaMemcpyDeviceToHost);
        int wi, wia;
        for(wi=0; wi<a; ++wi){
            for(wia=0; wia<b; ++wia)
                std::cout<<pm[mj({wi, wia})]<<' ';
            std::cout<<std::endl;
        }
    }
};

void Init_MM(){
    MakeMatrix<TA, LayoutA>::alloc(pA, dpA, M, K, LayoutA(K));
    MakeMatrix<TB, LayoutB>::alloc(pB, dpB, K, N, LayoutB(N));
    MakeMatrix<TC, LayoutC>::alloc(pC, dpC, M, N, LayoutC(N));
    MakeMatrix<ElementOutput, LayoutOutput>::alloc(pD, dpD, M, N, LayoutOutput(N), false);        
}

cudaError_t CutlassSgemmNN(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc) {

  // Define type definition for single-precision CUTLASS GEMM with column-major
  // input matrices and 128x128x8 threadblock tile size (chosen by default).
  //
  // To keep the interface manageable, several helpers are defined for plausible compositions
  // including the following example for single-precision GEMM. Typical values are used as
  // default template arguments. See `cutlass/gemm/device/default_gemm_configuration.h` for more details.
  //
  // To view the full gemm device API interface, see `cutlass/gemm/device/gemm.h`


  using CutlassGemm = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                                  RowMajor,  // Layout of A matrix
                                                  float,        // Data-type of B matrix
                                                  RowMajor,  // Layout of B matrix
                                                  float,        // Data-type of C matrix
                                                  RowMajor,
                                                    float,
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm70>; // Layout of C matrix
  SHOW_TYPE(CutlassGemm::ThreadblockShape);
  SHOW_TYPE(CutlassGemm::WarpShape);
  SHOW_TYPE(CutlassGemm::InstructionShape);
  CutlassGemm gemm_operator;
  CutlassGemm::Arguments args({M , N, K},  // Gemm Problem dimensions
                              {A, lda},    // Tensor-ref for source matrix A
                              {B, ldb},    // Tensor-ref for source matrix B
                              {C, ldc},    // Tensor-ref for source matrix C
                              {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                              {alpha, beta}); // Scalars used in the Epilogue

  cutlass::Status status = gemm_operator(args);

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }
  return cudaSuccess;
}



int main(){

    Init_MM();
    CutlassSgemmNN(M, N, K, 1., 
    dpA, K,
    dpB, N,
    1.,
    dpC, N);
    MakeMatrix<TC, LayoutC>::print(pC, dpC, M, N, LayoutC(N));
    return 0;
}