#include <vector>

#include "caffe/layers/lm_heatmap_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LMHeatmapLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    Forward_cpu(bottom, top);
}



INSTANTIATE_LAYER_GPU_FUNCS(LMHeatmapLayer);

}  // namespace caffe
