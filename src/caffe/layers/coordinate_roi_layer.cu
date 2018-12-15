#include <vector>

#include "caffe/layers/coordinate_roi_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CoordinateROILayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    Forward_cpu(bottom, top);
}

template <typename Dtype>
void CoordinateROILayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    // Nothing
}

INSTANTIATE_LAYER_GPU_FUNCS(CoordinateROILayer);

}  // namespace caffe
