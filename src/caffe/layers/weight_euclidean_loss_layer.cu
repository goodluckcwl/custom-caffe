#include <vector>

#include "caffe/layers/weight_euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void WeightEuclideanLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  this->Forward_cpu(bottom, top);
//  int count = bottom[0]->count();
//  caffe_gpu_sub(
//      count,
//      bottom[0]->gpu_data(),
//      bottom[1]->gpu_data(),
//      diff_.mutable_gpu_data());
//  caffe_gpu_mul(count, l2_weight_.gpu_data(), diff_.gpu_data(), diff_.mutable_gpu_data());
//
//  if (bottom.size() == 3) {
//    caffe_gpu_mul(count, bottom[2]->gpu_data(), diff_.gpu_data(), diff_.mutable_gpu_data());
//  }
//  Dtype dot;
//  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
//  Dtype loss = dot / bottom[0]->num() / Dtype(2);
//  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void WeightEuclideanLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  this->Backward_cpu(top, propagate_down ,bottom);
//  for (int i = 0; i < 2; ++i) {
//    if (propagate_down[i]) {
//      const Dtype sign = (i == 0) ? 1 : -1;
//      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
//      int count = bottom[0]->count();
//      caffe_gpu_mul(count, diff_.gpu_data(), l2_weight_.gpu_data(), diff_.mutable_gpu_data());
//      caffe_gpu_axpby(
//              bottom[i]->count(),              // count
//              alpha,                              // alpha
//              diff_.gpu_data(),                   // a
//              Dtype(0),                           // beta
//              bottom[i]->mutable_gpu_diff());  // b
//      Dtype* s = bottom[i]->mutable_gpu_diff();
//      for (int j = 0; j < count; ++j) {
//        s++;
//      }
//    }
//  }
}

INSTANTIATE_LAYER_GPU_FUNCS(WeightEuclideanLossLayer);

}  // namespace caffe
