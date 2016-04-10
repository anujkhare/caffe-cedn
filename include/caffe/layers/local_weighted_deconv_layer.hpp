
#ifndef CAFFE_LOCAL_WEIGHTED_DECONV_LAYER_HPP_
#define CAFFE_LOCAL_WEIGHTED_DECONV_LAYER_HPP_

#include <vector>
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

/**
 * @brief Deconvolves the input image with a bank of locally learned filters,
 *        and (optionally) adds biases.
 */
template <typename Dtype>
class LocalWeightedDeconvolutionLayer : public Layer<Dtype> {
 public:
  explicit LocalWeightedDeconvolutionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_LOCAL_WEIGHTED_DECONVOLUTION;
  }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline bool EqualNumBottomTopBlobs() const { return true; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);


  int kernel_size_;
  int stride_;
  int num_;
  int channels_input_;
  int pad_;
  int height_, width_;
  int height_out_, width_out_;
  int channels_output_;
  bool bias_term_;

  int M_;
  int K_;
  int N_;

  Blob<Dtype> col_buffer_;
};
}  // namespace caffe
#endif  // CAFFE_LOCAL_WEIGHTED_DECONV_LAYER_HPP_
