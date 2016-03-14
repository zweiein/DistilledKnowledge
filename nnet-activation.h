// nnet/nnet-activation.h

// Copyright 2011-2013  Brno University of Technology (author: Karel Vesely)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#ifndef KALDI_NNET_NNET_ACTIVATION_H_
#define KALDI_NNET_NNET_ACTIVATION_H_

#include "nnet/nnet-component.h"
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"
#include "cudamatrix/cu-rand.h"
#include "util/text-utils.h"

namespace kaldi {
namespace nnet1 {

class Softmax : public Component {
 public:
  Softmax(int32 dim_in, int32 dim_out) 
    : Component(dim_in, dim_out)
  { }
  ~Softmax()
  { }

  Component* Copy() const { return new Softmax(*this); }
  ComponentType GetType() const { return kSoftmax; }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {

    // 2016-03-04-11:47 測試在softmax加入讀檔，
    // 如果溫度不是1才做
    if ( softmax_tempurature_ > 1.0 )  {
      // y = e^x_j/sum_j(e^x_j)  // void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out)
      out -> CopyFromMat(in);
      out -> Scale(1.0/softmax_tempurature_) ;
      out->ApplySoftMaxPerRow(*out);

    } // end if
    else {
    // ----------------2016-03-04-11:47------------------


      // y = e^x_j/sum_j(e^x_j)
      out->ApplySoftMaxPerRow(in);

    } // else // 2016-03-04-11:47
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {

    // simply copy the error derivative
    // (ie. assume crossentropy error function, 
    // while in_diff contains (net_output-target) :
    // this is already derivative of the error with 
    // respect to activations of last layer neurons)
    in_diff->CopyFromMat(out_diff);
    
  }

  // 2016-03-04-11:47 測試在softmax加入讀檔，
  //  // 把temurature放在softmaxlabel 下面 
  // // (類似nnet格式裡面： <AffineTransform> 下面的<LearnRateCoef> 1 <BiasLearnRateCoef> 1 <MaxNorm> 1 )
  void ReadData(std::istream &is, bool binary) {
    if ('T' == PeekToken(is, binary)) {
      ExpectToken(is, binary, "<Tempurature>");
      ReadBasicType(is, binary, &softmax_tempurature_);
    }
    else { 
        softmax_tempurature_ = 1.0 ;
    }

    KALDI_ASSERT(softmax_tempurature_ > 0.0);
  }
  // ^^^^^^^^^^^2016-03-04-11:47^^^^^^^^^^^

  private:
  BaseFloat softmax_tempurature_;
};

// ---------------------------------------------------------------------------------------

class Tempurature : public Component {
 public:
  Tempurature(int32 dim_in, int32 dim_out):
      Component(dim_in, dim_out), softmax_tempurature_(20.0)
  { }
  ~Tempurature()
  { }

  Component* Copy() const { return new Tempurature(*this); }
  ComponentType GetType() const { return kTempurature; }

  void InitData(std::istream &is) {
    is >> std::ws; // eat-up whitespace
    // parse config
    std::string token; 
    while (!is.eof()) {

      KALDI_ERR << "softmax_tempurature_ = " << softmax_tempurature_;
      ReadToken(is, false, &token); 

      /**/ if (token == "<Tempurature>") ReadBasicType(is, false, &softmax_tempurature_);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (Tempurature)";
    }
    KALDI_ASSERT(softmax_tempurature_ > 0.0);
  }

  void ReadData(std::istream &is, bool binary) {
    if ('<' == Peek(is, binary)) {
      ExpectToken(is, binary, "<Tempurature>");
      ReadBasicType(is, binary, &softmax_tempurature_);
    }
    KALDI_ASSERT(softmax_tempurature_ > 0.0);
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<Tempurature>");
    WriteBasicType(os, binary, softmax_tempurature_);
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    // y = e^x_j/sum_j(e^x_j)
    out -> CopyFromMat(in);
    out -> Scale(1.0/softmax_tempurature_) ;
    out->ApplySoftMaxPerRow(*out);
    
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    // simply copy the error derivative
    // (ie. assume crossentropy error function, 
    // while in_diff contains (net_output-target) :
    // this is already derivative of the error with 
    // respect to activations of last layer neurons)
    in_diff->CopyFromMat(out_diff);
  }
  
  BaseFloat GetSoftmaxTemurature() {
    return softmax_tempurature_;
  }

  void SetSoftmaxTemurature(BaseFloat st) {
    softmax_tempurature_ = st;
    KALDI_ERR << "setting tempurature " << softmax_tempurature_;
    KALDI_ASSERT(softmax_tempurature_ > 0.0); // > 0.0
  }


 private:
  BaseFloat softmax_tempurature_;
};

/*
// ---------------------------------------------------------------------------------------
class BlockSoftmaxTempurature : public Component {
 public:
  BlockSoftmaxTempurature(int32 dim_in, int32 dim_out) 
    : Component(dim_in, dim_out)
  { }
  ~BlockSoftmaxTempurature()
  { }

  Component* Copy() const { return new BlockSoftmaxTempurature(*this); }
  ComponentType GetType() const { return kBlockSoftmaxTempurature; }
  
  void InitData(std::istream &is) {
    // parse config
    std::string token,
      tempurature_str;
    while (!is.eof()) {
      ReadToken(is, false, &token); 

      if (token == "<BlockSoftmaxTempurature>") is >> tempurature_str;
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (BlockSoftmaxTempurature)";
      is >> std::ws; // eat-up whitespace
    }
    // parse dims,
    if (!kaldi::SplitStringToIntegers(tempurature_str, ":", false, &block_tempuratures))
      KALDI_ERR << "Invalid block-dims " << tempurature_str;
    // sanity check
    int32 sum = 0;
    for (int32 i=0; i<block_tempuratures.size(); i++) {
      sum += block_tempuratures[i];
    }
    KALDI_ASSERT(sum == OutputDim()); 
  }

  void ReadData(std::istream &is, bool binary) {

    ReadIntegerVector(is, binary, &block_tempuratures);
    block_offset.resize(block_tempuratures.size()+1, 0);
    for (int32 i = 0; i < block_tempuratures.size(); i++) {
      block_offset[i+1] = block_offset[i] + block_tempuratures[i];
    }
    // check
    KALDI_ASSERT(OutputDim() == block_offset[block_offset.size()-1]);
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteIntegerVector(os, binary, block_tempuratures);
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    
    // y = e^x_j/sum_j(e^x_j)
    //out -> CopyFromMat(in);
    //out -> Scale(1.0/softmax_tempurature_) ;
    //out->ApplySoftMaxPerRow(*out);
    
    // perform softmax per block:
    for (int32 bl = 0; bl < block_tempuratures.size(); bl++) {
      CuSubMatrix<BaseFloat> in_bl = in.ColRange(block_offset[bl], block_tempuratures[bl]);
      CuSubMatrix<BaseFloat> out_bl = out->ColRange(block_offset[bl], block_tempuratures[bl]);
      // y = e^x_j/sum_j(e^x_j)
      out_bl.ApplySoftMaxPerRow(in_bl);
    }
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    // copy the error derivative:
    // (assuming we already got softmax-cross-entropy derivative in out_diff)
    in_diff->CopyFromMat(out_diff);
    
    // zero-out line-in-block, where sum different from zero,
    // process per block:
    for (int32 bl = 0; bl < block_tempuratures.size(); bl++) {
      CuSubMatrix<BaseFloat> diff_bl = in_diff->ColRange(block_offset[bl], block_tempuratures[bl]);
      CuVector<BaseFloat> row_sum(diff_bl.NumRows());
      row_sum.AddColSumMat(1.0, diff_bl, 0.0); // 0:keep, 1:zero-out
      // we'll scale rows by 0/1 masks
      CuVector<BaseFloat> row_diff_mask(row_sum);
      row_diff_mask.Scale(-1.0); // 0:keep, -1:zero-out
      row_diff_mask.Add(1.0); // 1:keep, 0:zero-out
      // here we should have only 0 and 1
      diff_bl.MulRowsVec(row_diff_mask);
    }
  }

  std::string Info() const {
    return "\n  softmax-tempuratures " + ToString(block_tempuratures);
  }

  std::vector<int32> block_tempuratures;
  std::vector<int32> block_offset;
};
// ---------------------------------------------------------------------------------------
*/

class BlockSoftmax : public Component {
 public:
  BlockSoftmax(int32 dim_in, int32 dim_out) 
    : Component(dim_in, dim_out)
  { }
  ~BlockSoftmax()
  { }

  Component* Copy() const { return new BlockSoftmax(*this); }
  ComponentType GetType() const { return kBlockSoftmax; }
  
  void InitData(std::istream &is) {
    // parse config
    std::string token,
      dims_str, tempurature_str;
    while (!is.eof()) {
      ReadToken(is, false, &token); 

      if (token == "<BlockDims>") is >> dims_str;
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (BlockDims)";
      is >> std::ws; // eat-up whitespace
      

      // ------2016-03-10-09:13 blocksoftmax也可讀取溫度參數blocksoftmaxtempurature
      if ('B' == PeekToken(is, false)) {
        ReadToken(is, false, &token); 
        if (token == "<BlockSoftmaxTempurature>") is >> tempurature_str;
        is >> std::ws; // eat-up whitespace
      } // peakToken
      // ^^^^^^ 2016-03-10-09:13 ^^^^^^
    }
    // parse dims,
    if (!kaldi::SplitStringToIntegers(dims_str, ":", false, &block_dims))
      KALDI_ERR << "Invalid block-dims " << dims_str;
     
    // ------2016-03-10-09:13 blocksoftmax也可讀取溫度參數blocksoftmaxtempurature
    // parse tempuratures,
    if (!kaldi::SplitStringToIntegers(tempurature_str, ":", false, &block_tempuratures))
      KALDI_ERR << "Invalid block-tempuratures " << tempurature_str;
    // ^^^^^^ 2016-03-10-09:13 ^^^^^^

    // sanity check
    int32 sum = 0;
    for (int32 i=0; i<block_dims.size(); i++) {
      sum += block_dims[i];
    }
    KALDI_ASSERT(sum == OutputDim()); 
  }

  void ReadData(std::istream &is, bool binary) {

    ReadIntegerVector(is, binary, &block_dims);
    ReadIntegerVector(is, binary, &block_tempuratures);
    // ex: block_dims = 3400 144 ; block_tempuratures = 10 20
    //     block_dims.size()= 2 ; block_tempuratures.size()= 2

    block_offset.resize(block_dims.size()+1, 0);

    for (int32 i = 0; i < block_dims.size(); i++) {
      block_offset[i+1] = block_offset[i] + block_dims[i];
    }

    // ------2016-03-10-09:13 
    //KALDI_ERR << "(ReadData) block_dims.size()= " << block_dims.size() 
    //          << "; block_tempuratures.size()= " << block_tempuratures.size()
    //          << " ; block_offset.size() = " << block_offset.size() 
    //          << " ; block_offset[0] = " << block_offset[0]     // 0
    //          << " ; block_offset[1] = " << block_offset[1]     // 3400
    //          << " ; block_offset[2] = " << block_offset[2]  ;  // 3544
    // ^^^^^^ 2016-03-10-09:13 ^^^^^^


    // check
    KALDI_ASSERT(OutputDim() == block_offset[block_offset.size()-1]);
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteIntegerVector(os, binary, block_dims);
    WriteIntegerVector(os, binary, block_tempuratures);  // ------2016-03-10-09:13 
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    // perform softmax per block:
    for (int32 bl = 0; bl < block_dims.size(); bl++) {
      CuSubMatrix<BaseFloat> in_bl = in.ColRange(block_offset[bl], block_dims[bl]);  
      CuSubMatrix<BaseFloat> out_bl = out->ColRange(block_offset[bl], block_dims[bl]);
      // y = e^x_j/sum_j(e^x_j)
      /*
      out -> CopyFromMat(in);
      out -> Scale(1.0/softmax_tempurature_) ;
      out->ApplySoftMaxPerRow(*out);

      // y = e^x_j/sum_j(e^x_j)
      out_bl_col -> CopyFromMat( in_bl );
      out_bl_col -> Scale( 1.0 / block_tempuratures[bl] ) ;
      out_bl.ApplySoftMaxPerRow(out_bl_col);
      */
      in_bl.Scale( 1.0 / block_tempuratures[bl] ); // ------2016-03-10-09:13 
      out_bl.ApplySoftMaxPerRow(in_bl);

      //KALDI_ERR << 
    }
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    // copy the error derivative:
    // (assuming we already got softmax-cross-entropy derivative in out_diff)
    in_diff->CopyFromMat(out_diff);
    
    // zero-out line-in-block, where sum different from zero,
    // process per block:
    for (int32 bl = 0; bl < block_dims.size(); bl++) {
      CuSubMatrix<BaseFloat> diff_bl = in_diff->ColRange(block_offset[bl], block_dims[bl]);
      
      // ------2016-03-10-09:13 ------- tempurature softmax 微分要除溫度
      //KALDI_LOG << "@Current Softmax Tempurature at BLOCK (" << block_offset[bl] << ", " << block_dims[bl] 
      //          << ") is : " << block_tempuratures[bl] ;

      diff_bl.Scale( 1.0 / block_tempuratures[bl] ); 
      // ^^^^^^ 2016-03-10-09:13 ^^^^^^

      CuVector<BaseFloat> row_sum(diff_bl.NumRows());
      row_sum.AddColSumMat(1.0, diff_bl, 0.0); // 0:keep, 1:zero-out
      // we'll scale rows by 0/1 masks
      CuVector<BaseFloat> row_diff_mask(row_sum);
      row_diff_mask.Scale(-1.0); // 0:keep, -1:zero-out
      row_diff_mask.Add(1.0); // 1:keep, 0:zero-out
      // here we should have only 0 and 1
      diff_bl.MulRowsVec(row_diff_mask);
    }
  }

  std::string Info() const {
    return "\n  softmax-dims " + ToString(block_dims);
  }

  std::vector<int32> block_dims;
  std::vector<int32> block_offset;
  std::vector<int32> block_tempuratures;
};




class Sigmoid : public Component {
 public:
  Sigmoid(int32 dim_in, int32 dim_out) 
    : Component(dim_in, dim_out)
  { }
  ~Sigmoid()
  { }

  Component* Copy() const { return new Sigmoid(*this); }
  ComponentType GetType() const { return kSigmoid; }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    // y = 1/(1+e^-x)
    out->Sigmoid(in);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    // ey = y(1-y)ex
    in_diff->DiffSigmoid(out, out_diff);
  }
};



class Tanh : public Component {
 public:
  Tanh(int32 dim_in, int32 dim_out) 
    : Component(dim_in, dim_out)
  { }
  ~Tanh()
  { }

  Component* Copy() const { return new Tanh(*this); }
  ComponentType GetType() const { return kTanh; }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    // y = (e^x - e^(-x)) / (e^x + e^(-x))
    out->Tanh(in);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    // ey = (1 - y^2)ex
    in_diff->DiffTanh(out, out_diff);
  }
};



class Dropout : public Component {
 public:
  Dropout(int32 dim_in, int32 dim_out):
      Component(dim_in, dim_out), dropout_retention_(0.5)
  { }
  ~Dropout()
  { }

  Component* Copy() const { return new Dropout(*this); }
  ComponentType GetType() const { return kDropout; }

  void InitData(std::istream &is) {
    is >> std::ws; // eat-up whitespace
    // parse config
    std::string token; 
    while (!is.eof()) {
      ReadToken(is, false, &token); 
      /**/ if (token == "<DropoutRetention>") ReadBasicType(is, false, &dropout_retention_);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (DropoutRetention)";
    }
    KALDI_ASSERT(dropout_retention_ > 0.0 && dropout_retention_ <= 1.0);
  }

  void ReadData(std::istream &is, bool binary) {
    if ('<' == Peek(is, binary)) {
      ExpectToken(is, binary, "<DropoutRetention>");
      ReadBasicType(is, binary, &dropout_retention_);
    }
    KALDI_ASSERT(dropout_retention_ > 0.0 && dropout_retention_ <= 1.0);
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<DropoutRetention>");
    WriteBasicType(os, binary, dropout_retention_);
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    out->CopyFromMat(in);
    // switch off 50% of the inputs...
    dropout_mask_.Resize(out->NumRows(),out->NumCols());
    dropout_mask_.Set(dropout_retention_);
    rand_.BinarizeProbs(dropout_mask_,&dropout_mask_);
    out->MulElements(dropout_mask_);
    // rescale to keep same dynamic range as w/o dropout
    out->Scale(1.0/dropout_retention_);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    in_diff->CopyFromMat(out_diff);
    // use same mask on the error derivatives...
    in_diff->MulElements(dropout_mask_);
    // enlarge output to fit dynamic range w/o dropout
    in_diff->Scale(1.0/dropout_retention_);
  }
  
  BaseFloat GetDropoutRetention() {
    return dropout_retention_;
  }

  void SetDropoutRetention(BaseFloat dr) {
    dropout_retention_ = dr;
    KALDI_ASSERT(dropout_retention_ > 0.0 && dropout_retention_ <= 1.0);
  }

 private:
  CuRand<BaseFloat> rand_;
  CuMatrix<BaseFloat> dropout_mask_;
  BaseFloat dropout_retention_;
};



} // namespace nnet1
} // namespace kaldi

#endif

