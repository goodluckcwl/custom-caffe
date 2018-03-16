#include <algorithm>
#include <vector>

#include "caffe/layers/coordinate_roi_layer.hpp"
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
  void CoordinateROILayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top) {
      roi_width_ = this->layer_param_.coordinate_roi_param().roi_width();
      roi_height_ = this->layer_param_.coordinate_roi_param().roi_height();
      coord_type_ = this->layer_param_.coordinate_roi_param().coord_type();
      CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0))<<"Number of predictions must equal to the number of images";

  }

  template <typename Dtype>
  void CoordinateROILayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top) {
      if(coord_type_ == CoordinateROIParameter_CoordType_NORM){
          num_pts_ = bottom[0]->shape(1) / 2;
      }else{
          num_pts_ = bottom[0]->shape(1);
      }

      std::vector<int> top_shape;
      top_shape.push_back(bottom[0]->shape(0) * num_pts_);
      top_shape.push_back(5);
      top[0]->Reshape(top_shape);
      // Bottom[1]: Image blob
      width_ = bottom[1]->shape(3);
      height_ = bottom[1]->shape(2);
  }

  template <typename Dtype>
  void CoordinateROILayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top) {
      const Dtype* pts_data= bottom[0]->cpu_data();
      Dtype* top_data = top[0]->mutable_cpu_data();
      int n = bottom[0]->shape(0);
      int pts_dim = bottom[0]->shape(1);
      for (int batch_idx = 0;  batch_idx < n; ++batch_idx) {
          //std::cout<<" "<<std::endl;
          for (int roi_n = 0; roi_n < num_pts_; ++roi_n) {
              Dtype x = 0;
              Dtype y = 0;
              int idx = 0;
              switch(coord_type_){
                  case CoordinateROIParameter_CoordType_NORM:
                      x = pts_data[batch_idx * pts_dim + roi_n] * width_;
                      y = pts_data[batch_idx * pts_dim + roi_n + num_pts_] * height_;
                      break;
                  case CoordinateROIParameter_CoordType_DIRECT:
                      idx = pts_data[batch_idx * pts_dim + roi_n];
                      x = idx % width_;
                      y = idx / width_;
                  //std::cout<<idx<<" ";
                      break;
                  default:
                      LOG(FATAL) << "Unknown coord type.";
              }

              int x1 = static_cast<int>(ceilf(x - roi_width_/2));
              int y1 = static_cast<int>(ceilf(y - roi_height_/2));
              int x2 = static_cast<int>(ceilf(x + roi_width_/2));
              int y2 = static_cast<int>(ceilf(y + roi_height_/2));
              x1 = min(max(x1, 0), width_ - 1);
              x2 = min(max(x2, 0), width_ - 1);
              y1 = min(max(y1, 0), height_ - 1);
              y2 = min(max(y2, 0), height_ - 1);

              *top_data++ = batch_idx;
              *top_data++ = x1;
              *top_data++ = y1;
              *top_data++ = x2;
              *top_data++ = y2;
              /* *top_data++ = 51;
              *top_data++ = 51;
              *top_data++ = 100;
              *top_data++ = 100; */
          }
      }
  }

  template <typename Dtype>
  void CoordinateROILayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    NOT_IMPLEMENTED;
  }


#ifdef CPU_ONLY
  STUB_GPU(CoordinateROILayer);
#endif

INSTANTIATE_CLASS(CoordinateROILayer);
REGISTER_LAYER_CLASS(CoordinateROI);
}  // namespace caffe