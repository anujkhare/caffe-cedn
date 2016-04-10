#ifndef CAFFE_SIGMOID_BERNOULLI_LAYER_HPP_
#define CAFFE_SIGMOID_BERNOULLI_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

/**
 * @brief Sample from sigmoid activation function @f$
 */
template <typename Dtype>
class SigmoidBernoulliLayer : public NeuronLayer<Dtype> {
 public:
  explicit SigmoidBernoulliLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param),
          sigmoid_layer_(new SigmoidLayer<Dtype>(param)),
          sigmoid_output_(new Blob<Dtype>()) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_SIGMOID_BERNOULLI;
  }

 protected:
  /// @copydoc SigmoidBernoulli
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  /**
   * @brief Computes the sigmoid bernoulli gradient
   * by measuring the difference between sigmoid activations and binary samples,
   * which is similar to sigmoid cross entropy loss gradient.
   *
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);

  /// The internal SigmoidLayer used to map predictions to probabilities.
  shared_ptr<SigmoidLayer<Dtype> > sigmoid_layer_;
  /// sigmoid_output stores the output of the SigmoidLayer.
  shared_ptr<Blob<Dtype> > sigmoid_output_;
  /// bottom vector holder to call the underlying SigmoidLayer::Forward
  vector<Blob<Dtype>*> sigmoid_bottom_vec_;
  /// top vector holder to call the underlying SigmoidLayer::Forward
  vector<Blob<Dtype>*> sigmoid_top_vec_;
};

}  // namespace caffe
#endif  // CAFFE_SIGMOID_BERNOULLI_LAYER_HPP_
