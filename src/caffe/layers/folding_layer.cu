#include <vector>

//#include "caffe/layer.hpp"
//#include "caffe/util/math_functions.hpp"
//#include "caffe/vision_layers.hpp"
#include "caffe/layers/folding_layer.hpp"

namespace caffe {

template <typename Dtype>
void FoldingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  (*top)[0]->ShareData(*bottom[0]);
}

template <typename Dtype>
void FoldingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  (*bottom)[0]->ShareDiff(*top[0]);
}

INSTANTIATE_CLASS(FoldingLayer);

}  // namespace caffe
