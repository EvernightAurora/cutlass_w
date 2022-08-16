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
enum FILLTYPE{
    None = 0,
    Diag = 1,
    Random = 2,
    Hilbert = 3,
    xHilbert = 4,
    Range = 5,
    cRange = 6,
    sRange = 7,
};




using cutlass::gemm::GemmShape;
using cutlass::epilogue::thread::LinearCombination;
using cutlass::sizeof_bits;
using cutlass::tfloat32_t;

template<typename T, typename MAJOR>
struct MakeMatrix{

    class STORAGE{
        public:
        T* point;
        int byte_offset;
        int inbyte_offset;
        int item_size_bits;
        STORAGE(T*p, int byte_offset_):
            point(p), byte_offset(byte_offset_), inbyte_offset(0), item_size_bits(sizeof_bits<T>::value){
                assert(item_size_bits >= 8);
            };
        STORAGE(T*p, int byte_offset_, int inbyte_offset_):
            point(p), byte_offset(byte_offset_), inbyte_offset(inbyte_offset_), item_size_bits(sizeof_bits<T>::value){
                assert(item_size_bits < 8);
            }
        
        operator T(){
            if(item_size_bits >= 8)
                return *reinterpret_cast<T*>(reinterpret_cast<char*>(point) + byte_offset);
            return T(((*(reinterpret_cast<char*>(point) + byte_offset)) >> inbyte_offset) & ((1<<item_size_bits) - 1) );
        }
        STORAGE& operator =(const T&i){
            if(item_size_bits >= 8)
                *reinterpret_cast<T*>(reinterpret_cast<char*>(point) + byte_offset) = i;
            else{
                //clear
                (*(reinterpret_cast<char*>(point) + byte_offset)) &= ~ (((1<<item_size_bits) - 1) << inbyte_offset);
                //set
                (*(reinterpret_cast<char*>(point) + byte_offset)) |=  (((long)i) << inbyte_offset);
            }
            return *this;
        }
        template<typename nT>
        STORAGE& operator =(const nT&i){
            *this = T(i);
        }
        bool operator ==(const STORAGE&i){
            return T(*this) == T(i);
        }
        friend std::ostream& operator <<(std::ostream&o, STORAGE i){
            return o<<(T(i));
        }
    };

    static STORAGE locate(T*v, int offset){
        if(sizeof_bits<T>::value >= 8)
            return STORAGE(v, offset*sizeof(T));
        int byte_offset = offset * sizeof_bits<T>::value / 8;
        int inbyte_offset = 8 - (offset * sizeof_bits<T>::value % 8 + sizeof_bits<T>::value);
        return STORAGE(v, byte_offset, inbyte_offset);
        }
    
    
    static cudaError_t alloc(T*&pm, T*&dpm, int a, int b, FILLTYPE filltype=FILLTYPE::None){
        MAJOR mj = LAYOUT<MAJOR>(a, b);
        size_t item_size_bits = sizeof_bits<T>::value;
        pm = reinterpret_cast<T*>(new char[a * b * item_size_bits / 8]);
        auto err_t = cudaMalloc((void**)&dpm, a*b*item_size_bits / 8);
        int wi, wia;
        memset(pm, 0, a*b*item_size_bits / 8);
        switch(filltype){
            case FILLTYPE::Diag:
                for(wi=0;wi<a && wi<b; ++wi)
                    locate(pm, mj((cutlass::MatrixCoord){wi, wi})) = 1;
                break;

            case FILLTYPE::Random:
                for(wi=0; wi<a; ++wi)
                    for(wia=0; wia<b; ++wia)
                        locate(pm, mj((cutlass::MatrixCoord){wi, wia})) = rand() * 1.0 / RAND_MAX;
                break;

            case FILLTYPE::Hilbert:
                for(wi=0; wi<a; ++wi)
                    for(wia=0; wia<b; ++wia)
                        locate(pm, mj((cutlass::MatrixCoord){wi, wia})) = 1. / (1. + wi + wia);
                break;

            case FILLTYPE::xHilbert:
                for(wi=0; wi<a; ++wi)
                    for(wia=0; wia<b; ++wia)
                        if(wia > wi)
                            locate(pm, mj((cutlass::MatrixCoord){wi, wia})) = 1. / (1. + wi + wia);
                        else if(wia <wi)
                            locate(pm, mj((cutlass::MatrixCoord){wi, wia})) = -1. / (1. + wi + wia);
                        else
                            locate(pm, mj((cutlass::MatrixCoord){wi, wia})) = 1;
                break;
            
            case FILLTYPE::Range:
                for(wi=0; wi<a; ++wi)
                    for(wia=0; wia<b; ++wia)
                        locate(pm, mj((cutlass::MatrixCoord){wi, wia})) = wi*b + wia;
                break;
            
            case FILLTYPE::cRange:
                 for(wi=0; wi<a; ++wi)
                    for(wia=0; wia<b; ++wia)
                        locate(pm, mj((cutlass::MatrixCoord){wi, wia})) = wi + wia*a;
            break;

            case FILLTYPE::sRange:
                 for(wi=0; wi<a; ++wi)
                    for(wia=0; wia<b; ++wia)
                        locate(pm, mj((cutlass::MatrixCoord){wi, wia})) = std::max(wi, wia);
            break;
        }
        cudaMemcpy(dpm, pm, a*b*item_size_bits / 8, cudaMemcpyHostToDevice);
        return err_t;
    }
    static void print(T*pm, T*dpm, int a, int b){
        MAJOR mj = LAYOUT<MAJOR>(a, b);
        cudaMemcpy(pm, dpm, a*b*sizeof_bits<T>::value / 8, cudaMemcpyDeviceToHost);
        int wi, wia;
        for(wi=0; wi<a; ++wi){
            std::cout<<"| ";
            for(wia=0; wia<b; ++wia)
                std::cout<<T(locate(pm, mj({wi, wia})))<<' ';
            std::cout<<std::endl;
        }
    }
    static void sync(T* pa, T* dpa, int a, int b){
        cudaMemcpy(pa, dpa, a*b*sizeof_bits<T>::value / 8, cudaMemcpyDeviceToHost);
    }
    static bool compare(T* pa, T* dpa, T* pb, T* dpb, int a, int b, double maxeps_rate){
        MAJOR mj = LAYOUT<MAJOR>(a, b);
        cudaMemcpy(pa, dpa, a*b*sizeof_bits<T>::value / 8, cudaMemcpyDeviceToHost);
        cudaMemcpy(pb, dpb, a*b*sizeof_bits<T>::value / 8, cudaMemcpyDeviceToHost);
        int wi,wia;
        auto abs = [](T a){return a>0? a:-a;};
        auto max = [](T a, T b){return a>b? a:b;};
        for(wi=0; wi<a; ++wi)
            for(wia=0; wia<b; ++wia){
                T va = locate(pa, mj({wi, wia})), vb = locate(pb, mj({wi, wia}));
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
    Testing(int m=168, int n=248, int k=104, FILLTYPE first_fill_type=FILLTYPE::Random, FILLTYPE second_fill_type=FILLTYPE::Diag){
        M = m;
        N = n;
        K = k;
        MakeMatrix<TA, LA>::alloc(pA, dpA, m, k, first_fill_type);
        MakeMatrix<TB, LB>::alloc(pB, dpB, k, n, first_fill_type);
        MakeMatrix<TC, LC>::alloc(pC, dpC, m, n, second_fill_type);
        MakeMatrix<TD, LC>::alloc(pDa, dpDa, m, n, second_fill_type);
        MakeMatrix<TD, LC>::alloc(pDb, dpDb, m, n, second_fill_type);
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
        cutlass::Status r_ca;
        r_ca = gemm(arg);
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


template<typename TA, typename LA, typename TB, typename LB, typename TC, typename LC,
            typename TD>
struct TestOffical: public TestingBase{
    using GEMM_ref = cutlass::gemm::device::Gemm<TA, LA, TB, LB, TC, LC, TD, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75>;
    GEMM_ref gemm_ref;
    TA*pA, *dpA;
    TB*pB, *dpB;
    TC*pC, *dpC;
    TD*pD, *dpD;
    int M, N, K;
    TestOffical(int m=168, int n=248, int k=104, FILLTYPE first_fill_type=FILLTYPE::Range, FILLTYPE second_fill_type=FILLTYPE::Diag){
        M = m;
        N = n;
        K = k;
        MakeMatrix<TA, LA>::alloc(pA, dpA, m, k, first_fill_type);
        MakeMatrix<TB, LB>::alloc(pB, dpB, k, n, first_fill_type);
        MakeMatrix<TC, LC>::alloc(pC, dpC, m, n, second_fill_type);
        MakeMatrix<TD, LC>::alloc(pD, dpD, m, n, second_fill_type);
    }
    void Test(){
        TC one((float)1.);
        typename GEMM_ref::Arguments arg0 = {
            {M, N, K},
            {dpA, LAYOUT<LA>(M, K).stride()},
            {dpB, LAYOUT<LB>(K, N).stride()},
            {dpC, LAYOUT<LC>(M, N).stride()},
            {dpD, LAYOUT<LC>(M, N).stride()},
            {one, one}};
        cutlass::Status r_ca;
        std::cout<<"On Official Test "<<++NumTest<<" \t";
        r_ca = gemm_ref(arg0);
        if(r_ca != cutlass::Status::kSuccess)
            std::cout<<"Ref Gemm Run Error! ";
        auto r_cu = cudaDeviceSynchronize();
        if(r_cu != cudaSuccess)
            std::cout<<"Sync error! ";
        MakeMatrix<TD, LC>::sync(pD, dpD, M, N);
        LC major = LAYOUT<LC>(M, N);
        std::cout<<"the final val is"<<MakeMatrix<TC, LC>::locate(pD, major({M-1, N-1}))<<std::endl;
    }
};