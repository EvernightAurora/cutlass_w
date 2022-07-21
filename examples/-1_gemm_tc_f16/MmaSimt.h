
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/warp/mma.h"
#include "cutlass/gemm/thread/mma.h"
#include "cutlass/gemm/warp/mma_simt_tile_iterator.h"
#include "cutlass/gemm/warp/mma_simt_policy.h"
#include "WarpIterator.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace MMA_SIMT {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math instructions.
template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape_,
  /// Data type of A elements
  typename ElementA_,
  /// Layout of A matrix (concept: MatrixLayout)
  typename LayoutA_,
  /// Data type of B elements
  typename ElementB_,
  /// Layout of B matrix (concept: MatrixLayout)
  typename LayoutB_,
  /// Element type of C matrix
  typename ElementC_,
  /// Layout of C matrix (concept: MatrixLayout)
  typename LayoutC_,
  /// Shape of the warp in units of thread (concept: MmaSimtPolicy)
  typename Policy_,
  /// Number of partitions along K dimension
  int PartitionsK = 1,
  /// Complex transformation on operand A
  cutlass::ComplexTransform TransformA = cutlass::ComplexTransform::kNone,
  /// Complex transformation on operand B
  cutlass::ComplexTransform TransformB = cutlass::ComplexTransform::kNone,
  /// Used for partial specialization
  typename Enable = bool
>
class MmaSimt {
public:
  /// Shape of warp-level matrix operation (concept: GemmShape)
  using Shape = Shape_;
  static auto const SIGN_LINE = __LINE__;
  /// Data type of multiplicand A
  using ElementA = ElementA_;

  /// Layout of multiplicand A
  using LayoutA = LayoutA_;

  /// Data type of multiplicand B
  using ElementB = ElementB_;

  /// Layout of multiplicand B
  using LayoutB = LayoutB_;

  /// Data type of accumulator matrix C
  using ElementC = ElementC_;

  /// Layout of accumulator matrix C
  using LayoutC = LayoutC_;

  /// Shape of the warp in units of thread (concept: MmaLanePolicySimt)
  using Policy = Policy_;

  /// Indicates class of matrix operator
  using OperatorClass = cutlass::arch::OpClassSimt;

  /// Hard-coded for now
  using ArchTag = cutlass::arch::Sm50;

  /// Complex transform on A operand
  static cutlass::ComplexTransform const kTransformA = TransformA;

  /// Complex transform on B operand
  static cutlass::ComplexTransform const kTransformB = TransformB;

  /// Layout of threads
  using ThreadLayoutA = LayoutA;
  
  using ThreadLayoutB = LayoutB;

  static constexpr bool use_dp4a = false;
  /*
                                    (cutlass::platform::is_same< cutlass::layout::ColumnMajorInterleaved<4>, LayoutA>::value || 
                                    cutlass::platform::is_same< cutlass::layout::RowMajorInterleaved<4>, LayoutA >::value) && 
                                    cutlass::platform::is_same< ElementA, int8_t >::value && 
                                    cutlass::platform::is_same< ElementB, int8_t >::value;
*/
  using dp4a_type = typename cutlass::platform::conditional< use_dp4a , int8_t, bool >::type;

  /// Thread-level matrix multiply accumulate operator
  using ThreadMma = cutlass::gemm::thread::Mma<
    cutlass::gemm::GemmShape<
      Shape::kM / Policy::WarpShape::kRow,
      Shape::kN / Policy::WarpShape::kColumn,
      Policy::LaneMmaShape::kK>,
    ElementA,
    ThreadLayoutA,
    ElementB,
    ThreadLayoutB,
    ElementC,
    LayoutC,
    cutlass::arch::OpMultiplyAdd,
    dp4a_type
  >;

  /// Underlying matrix multiply operator (concept: arch::Mma)
  using ArchMmaOperator = typename ThreadMma::ArchMmaOperator;

  /// Indicates math operator 
  using MathOperator = typename ArchMmaOperator::Operator;
  
  /// Shape of the underlying instruction
  using InstructionShape = cutlass::gemm::GemmShape<1,1,use_dp4a ? 4 : 1>;

public:
          //LaneMmaShape: <4, 4, 1> ,   WShape: <32, 64, 8>
  /// Iterates over the A operand in memory
  using IteratorA = WARP_ITERATOR::WarpIterator<
    cutlass::MatrixShape<Shape::kM, Policy::LaneMmaShape::kK>,          //32, 1
    cutlass::gemm::Operand::kA,               //0
    ElementA,
    LayoutA,
    Policy,
    PartitionsK,                          //1
    Shape::kK                             //wshape::kK,   8
  >;

  /// Storage for A tile
  using FragmentA = typename IteratorA::Fragment;

  /// Storage for transformed A tile
  using TransformedFragmentA = FragmentA;

  /// Iterates over the B operand in memory
  using IteratorB = WARP_ITERATOR::WarpIterator<
    cutlass::MatrixShape<Policy::LaneMmaShape::kK, Shape::kN>,            //1, 64
    cutlass::gemm::Operand::kB,
    ElementB,
    LayoutB,
    Policy,
    PartitionsK,
    Shape::kK
  >;

  /// Storage for B tile
  using FragmentB = typename IteratorB::Fragment;

  /// Storage for transformed A tile
  using TransformedFragmentB = FragmentB;

  /// Iterates over the C operand in memory
  using IteratorC = WARP_ITERATOR::WarpIterator<
    cutlass::MatrixShape<Shape::kM, Shape::kN>,
    cutlass::gemm::Operand::kC,
    ElementC,
    LayoutC,
    Policy
  >;

  /// Storage for C tile
  using FragmentC = typename ThreadMma::FragmentC;

public:

  //
  // Methods
  //

  /// Ctor
  CUTLASS_DEVICE
  MmaSimt() {}

  /// Performs a warp-level matrix multiply-accumulate operation
  CUTLASS_DEVICE
  void operator()(
    FragmentC &d, 
    FragmentA a, 
    FragmentB b, 
    FragmentC const &c, int group_idx = 0) const {

    ThreadMma mma;

    if (kTransformA == cutlass::ComplexTransform::kConjugate) {
      cutlass::conjugate<FragmentA> conj_a;
      a = conj_a(a);
    }

    if (kTransformB == cutlass::ComplexTransform::kConjugate) {
      cutlass::conjugate<FragmentB> conj_b;
      b = conj_b(b);
    }

    mma(d, a, b, c);
  }

  /// Transform the mma operands to the required types
  CUTLASS_DEVICE
  void transform(TransformedFragmentA &dst_A, TransformedFragmentB &dst_B,
                 FragmentA const &A, FragmentB const &B) const {
    //TODO: Implement this
    dst_A = A;
    dst_B = B;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} 