#ifndef CAFFE_UNPOOLING_LAYER_HPP_
#define CAFFE_UNPOOLING_LAYER_HPP_

#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Unpools the input image by reserving the pooling operation within regions.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class UnpoolingLayer : public Layer<Dtype> {
 public:
  explicit UnpoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  //virtual inline LayerParameter_LayerType type() const {
  //  return LayerParameter_LayerType_UNPOOLING;
  //}
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline int MinBottomBlobs() const { return 1; }
  // MAX UNPOOL layers takes as input an extra bottom blob for the mask;
  // others take only one bottom blob.
  virtual inline int MaxBottomBlobs() const {
    return (this->layer_param_.unpooling_param().unpool() ==
            UnpoolingParameter_UnpoolMethod_MAX) ? 2 : 1;
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);

  int kernel_h_, kernel_w_;
  int stride_h_, stride_w_;
  int pad_h_, pad_w_;
  int channels_;
  int height_, width_;
  int unpooled_height_, unpooled_width_;
  Blob<Dtype> rand_idx_;
};

}  // namespace caffe
#endif  // CAFFE_UNPOOLING_LAYER_HPP_
