#ifndef CAFFE_TENSOR_PRODUCT_LAYER_HPP_
#define CAFFE_TENSOR_PRODUCT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
class TensorProductLayer : public Layer<Dtype> {
 public:
  explicit TensorProductLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  //virtual inline LayerParameter_LayerType type() const {
  //  return LayerParameter_LayerType_TENSOR_PRODUCT;
  //}

  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);

  int M_;
  int K_;
  int L_;
  int N_;
  bool bias_term_;
  Blob<Dtype> bias_multiplier_;
  Blob<Dtype> bias_multiplier2_;

  Blob<Dtype> T_;
  Blob<Dtype> S_;
  Blob<Dtype> LN_;
  Blob<Dtype> NK_;
};

}  // namespace caffe
#endif  // CAFFE_TENSOR_PRODUCT_LAYER_HPP_
