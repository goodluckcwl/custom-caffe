#include <vector>

#include "caffe/layers/coordinate_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CoordinateLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    Forward_cpu(bottom, top);
}
template <typename Dtype>
void CoordinateLayer<Dtype>::Backward_gpu(const std::vector<caffe::Blob<Dtype> *> &top,
                                          const std::vector<bool> &propagate_down,
                                          const std::vector<caffe::Blob<Dtype> *> &bottom) {
    //Nothing
}


INSTANTIATE_LAYER_GPU_FUNCS(CoordinateLayer);

}  // namespace caffe
