#include <algorithm>
#include <vector>

#include "caffe/layers/lm_heatmap_layer.hpp"
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
  void LMHeatmapLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top) {
      width_ = this->layer_param_.lm_heatmap_param().width();
      height_ = this->layer_param_.lm_heatmap_param().height();
      gaussion_sigma_ = this->layer_param_.lm_heatmap_param().sigma();
      roi_ = this->layer_param_.lm_heatmap_param().roi();
      num_pts_ = this->layer_param_.lm_heatmap_param().num_pts();
  }

  template <typename Dtype>
  void LMHeatmapLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top) {
      CHECK_EQ(num_pts_, bottom[0]->shape(1) / 2)
          <<"Number pts should equal to that calcalate from the bottom[0]";

      std::vector<int> top_shape;
      int num = bottom[0]->shape(0);
      top_shape.push_back(num);
      top_shape.push_back(num_pts_);
      top_shape.push_back(height_);
      top_shape.push_back(width_);
      top[0]->Reshape(top_shape);
      int roi_w = roi_ * 2 + 1;
      int roi_h = roi_ * 2 + 1;
      prob_groundtruth_.Reshape(num, num_pts_, roi_w, roi_h);
  }

  template <typename Dtype>
  void LMHeatmapLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top) {
      // Input coordinate
      const Dtype *label = bottom[0]->cpu_data();
      Dtype *top_data = top[0]->mutable_cpu_data();
      // Set to zero
      caffe_set(top[0]->count(), Dtype(0), top_data);

      int num_images = top[0]->shape(0);
      int channels = top[0]->shape(1);
      int dim1 = bottom[0]->count() / num_images;
      int n_step = top[0]->count(1);
      int c_step = top[0]->count(2);
      // Generate groundtruth probability map
      int roi_w = roi_ * 2 + 1;
      int roi_h = roi_ * 2 + 1;
      for (int n = 0; n < num_images; ++n) {
          for (int c = 0; c < channels; ++c) {
              Dtype *dst = top_data + n * n_step + c * c_step;

              // For each landmark
              Dtype pts_x = label[n * dim1 + c] * width_;
              Dtype pts_y = label[n * dim1 + c + channels] * height_;
              int x1 = round(pts_x) - roi_;
              int y1 = round(pts_y) - roi_;
              int x2 = round(pts_x) + roi_;
              int y2 = round(pts_y) + roi_;


              Dtype prob_normalizer = 0;
              for (int y = y1; y < y2; ++y) {
                  for (int x = x1; x < x2; ++x) {

                      // -1/(2*PI*sigma^2)*exp(-0.5*d^2/sigma^2)
                      dst[y * width_ + x] = 1 / (sqrt(2 * M_PI) * gaussion_sigma_) *
                                            exp(-0.5 * ((x - pts_x) * (x - pts_x) + (y - pts_y) * (y - pts_y)) /
                                                (gaussion_sigma_ * gaussion_sigma_));

                      prob_normalizer += dst[y * width_ + x];
                  }
              }
              // Normalize for each channels
              caffe_scal(c_step, Dtype(1.0 / prob_normalizer), dst);
          }
      }

  }

#ifdef CPU_ONLY
  STUB_GPU(LMHeatmapLayer);
#endif

INSTANTIATE_CLASS(LMHeatmapLayer);
REGISTER_LAYER_CLASS(LMHeatmap);
}  // namespace caffe