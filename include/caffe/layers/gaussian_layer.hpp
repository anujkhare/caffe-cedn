#ifndef CAFFE_GAUSSIAN_LAYER_HPP_
#define CAFFE_GAUSSIAN_LAYER_HPP_

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 * @brief add gaussian noises to input data @f$
 */
template <typename Dtype>
class GaussianLayer : public Layer<Dtype> {
 public:
  explicit GaussianLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  //virtual inline LayerParameter_LayerType type() const {
  //  return LayerParameter_LayerType_GAUSSIAN;
  //}
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }


 protected:
  /// @copydoc Gaussian
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  /**
   * @brief Computes the gaussian gradient
   *
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);

  Blob<Dtype> noise_;
  Dtype sigma_;

};

}  // namespace caffe
#endif  // CAFFE_GAUSSIAN_LAYER_HPP_
