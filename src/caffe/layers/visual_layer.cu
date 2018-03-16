#include <vector>

#include "caffe/layers/visual_layer.hpp"

namespace caffe {

template <typename Dtype>
void VisualLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    this->Forward_cpu(bottom, top);
}

template <typename Dtype>
void VisualLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
}

INSTANTIATE_LAYER_GPU_FUNCS(VisualLayer);

}  // namespace caffe
