#include <algorithm>
#include <vector>

#include "caffe/layers/coordinate_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/common.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

  template <typename Dtype>
  void CoordinateLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top) {
      width_ = this->layer_param_.coordinate_param().width();
      height_ = this->layer_param_.coordinate_param().height();

  }

  template <typename Dtype>
  void CoordinateLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top) {
      num_pts_ = bottom[0]->shape(1);

      std::vector<int> top_shape;
      top_shape.push_back(bottom[0]->shape(0));
      top_shape.push_back(num_pts_ * 2);
      top[0]->Reshape(top_shape);
  }

  template <typename Dtype>
  void CoordinateLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top) {
      // Input coordinate
      const Dtype* pts_data= bottom[0]->cpu_data();
      Dtype* top_data = top[0]->mutable_cpu_data();
      int n = bottom[0]->shape(0);
      int dim = top[0]->shape(1);
      int w = bottom[0]->shape(1);
      //
      for (int batch_idx = 0;  batch_idx < n; ++batch_idx) {

          for (int roi_n = 0; roi_n < num_pts_; ++roi_n) {
              Dtype x = 0;
              Dtype y = 0;
              int idx = 0;
              idx = pts_data[batch_idx * num_pts_ + roi_n];
              // Calculate landmarks. C-index
              x = idx % width_;
              y = idx / width_;
              // Normalized
              x = x * 1.0 / width_;
              y = y * 1.0 / height_;
              top_data[batch_idx * dim + roi_n] = x;
              top_data[batch_idx * dim + roi_n + num_pts_] = y;
          }
      }
  }



#ifdef CPU_ONLY
  STUB_GPU(CoordinateLayer);
#endif

INSTANTIATE_CLASS(CoordinateLayer);
REGISTER_LAYER_CLASS(Coordinate);
}  // namespace caffe