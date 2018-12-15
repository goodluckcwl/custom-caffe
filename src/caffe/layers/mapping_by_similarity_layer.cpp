#include <algorithm>
#include <vector>

#include "caffe/layers/mapping_by_similarity_layer.hpp"
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
  void MappingBySimilarityLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top) {
      alpha_ = this->layer_param_.mapping_by_similarity_param().alpha();
      CHECK_LE(alpha_, 1) << "alpha must be 0~1";
      CHECK_GE(alpha_, 0) << "alpha must be 0~1";
  }

  template <typename Dtype>
  void MappingBySimilarityLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top) {
      CHECK_EQ(bottom.size(), 2)<<"Bottom size must be 2.";
      int n_images = bottom[0]->shape(0);
      int channels = bottom[0]->shape(1);
      int height = bottom[0]->shape(2);
      int width = bottom[0]->shape(3);
      CHECK_EQ(bottom[1]->shape(1), 1)<<"Channels of similarity map must be 1.";
      CHECK_EQ(n_images/2, bottom[1]->shape(0))<<
      "The number of images in bottom[0]: ("<<n_images<< ")must be two times of that in bottom[1]("
                                                         <<bottom[1]->shape(0)<<")";
      CHECK_EQ(height, bottom[1]->shape(2))<<"Image height must eaual to height of similarity map";
      CHECK_EQ(width, bottom[1]->shape(3))<<"Image width must eaual to width of similarity map";

      std::vector<int> top_shape(4);
      top_shape[0] = n_images;
      top_shape[1] = channels;
      top_shape[2] = height;
      top_shape[3] = width;
      top[0]->Reshape(top_shape);
  }

  template <typename Dtype>
  void MappingBySimilarityLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top) {
      const Dtype *bottom_data = bottom[0]->cpu_data();
      const Dtype *sim_data = bottom[1]->cpu_data();
      Dtype *top_data = top[0]->mutable_cpu_data();
      int n_images = bottom[0]->shape(0);
      int channels = bottom[0]->shape(1);
      int height = bottom[0]->shape(2);
      int width = bottom[0]->shape(3);

      int channel_size = bottom[0]->count(2);
      for (int n = 0; n < n_images/2; ++n) {
          for (int c = 0; c < channels; ++c) {
              const Dtype *s1 = bottom_data + 2*n * bottom[0]->count(1) + c * channel_size;
              const Dtype *s2 = bottom_data + (2*n+1) * bottom[0]->count(1) + c * channel_size;
              Dtype *d1 = top_data + 2*n * top[0]->count(1) + c * channel_size;
              Dtype *d2 = top_data + (2*n+1) * top[0]->count(1) + c * channel_size;
              for (int h = 0; h < height; ++h) {
                  for (int w = 0; w < width; ++w) {
                      Dtype beta = *(sim_data + n * bottom[1]->count(1) + h * width + w);
                      Dtype x1 = *(s1 + h * width + w);
                      Dtype x2 = *(s2 + h * width + w);
                      *(d1++) = (x1 + alpha_ * (1-beta) * x2)/(1 + alpha_ - alpha_ * beta);
                      *(d2++) = (x2 + alpha_ * (1-beta) * x1)/(1 + alpha_ - alpha_ * beta);
                  }
              }
          }

      }
  }

  template <typename Dtype>
  void MappingBySimilarityLayer<Dtype>::Backward_cpu(const std::vector<caffe::Blob<Dtype> *> &top,
                                                     const std::vector<bool> &propagate_down,
                                                     const std::vector<caffe::Blob<Dtype> *> &bottom) {
      if (propagate_down[0]) {
          std::cout<<"Enter backward0"<<std::endl;
          // gradient w.r.t. image. Note that we will accumulate diffs.
          Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
          const Dtype *top_diff = top[0]->cpu_diff();
          const Dtype *sim_data = bottom[1]->cpu_data();
          int n_images = bottom[0]->shape(0);
          int channels = bottom[0]->shape(1);
          int height = bottom[0]->shape(2);
          int width = bottom[0]->shape(3);
          // Clear grad
          caffe_set(bottom[0]->count(0), Dtype(0.0), bottom_diff);
          int channel_size = bottom[0]->count(2);
          for (int n = 0; n < n_images/2; ++n) {
              for (int c = 0; c < channels; ++c) {
                  Dtype *diff1 = bottom_diff + 2*n * bottom[0]->count(1) + c * channel_size;
                  Dtype *diff2 = bottom_diff + (2*n+1) * bottom[0]->count(1) + c * channel_size;
                  const Dtype *d1 = top_diff + 2*n * top[0]->count(1) + c * channel_size;
                  const Dtype *d2 = top_diff + (2*n+1) * top[0]->count(1) + c * channel_size;
                  for (int h = 0; h < height; ++h) {
                      for (int w = 0; w < width; ++w) {
                          Dtype beta = *(sim_data + n * bottom[1]->count(1) + h * width + w);
                          Dtype d1_diff = *(d1 + h * width + w);
                          Dtype d2_diff = *(d2 + h * width + w);
                          Dtype factor = (1+alpha_-alpha_*beta);
                          *(diff1 + h * width + w) += 1/factor * d1_diff + alpha_*(1-beta)/ factor*d2_diff;
                          *(diff2 + h * width + w) += 1/factor * d2_diff + alpha_*(1-beta)/ factor*d1_diff;
                      }
                  }
              }
          }
      }
      if (propagate_down[1]){
          // gradient w.r.t. similarity map. Note that we will accumulate diffs.
          std::cout<<"Enter backward1"<<std::endl;
          Dtype* sim_diff = bottom[1]->mutable_cpu_diff();
          const Dtype *sim_data = bottom[1]->cpu_data();

          const Dtype* bottom_data = bottom[0]->cpu_data();
          const Dtype *top_diff = top[0]->cpu_diff();

          int n_images = bottom[0]->shape(0);
          int channels = bottom[0]->shape(1);
          int height = bottom[0]->shape(2);
          int width = bottom[0]->shape(3);
          int channel_size = bottom[0]->count(2);
          // Clear grads. Because we need to accumulate diffs.
          // Note that the similarity map has only one channel.
          caffe_set(bottom[1]->count(0), Dtype(0.0), sim_diff);
          for (int n = 0; n < n_images/2; ++n) {
              for (int c = 0; c < channels; ++c) {
                  const Dtype *s1 = bottom_data + 2*n * bottom[0]->count(1) + c * channel_size;
                  const Dtype *s2 = bottom_data + (2*n+1) * bottom[0]->count(1) + c * channel_size;
                  Dtype *diff = sim_diff + n * bottom[1]->count(1) ;
                  const Dtype *d1 = top_diff + 2*n * top[0]->count(1) + c * channel_size;
                  const Dtype *d2 = top_diff + (2*n+1) * top[0]->count(1) + c * channel_size;
                  for (int h = 0; h < height; ++h) {
                      for (int w = 0; w < width; ++w) {
                          Dtype beta = *(sim_data + n * bottom[1]->count(1) + h * width + w);

                          Dtype x1 = *(s1 + h * width + w);
                          Dtype x2 = *(s2 + h * width + w);
                          Dtype factor = 1 + alpha_ - alpha_ * beta;
                          Dtype factor1 = (alpha_ * x1 - alpha_ * x2 )/(factor*factor);
                          Dtype factor2 = (alpha_ * x2 - alpha_ * x1 )/(factor*factor);
                          // Accumulate diffs
                          Dtype d1_diff = *(d1 + h * width + w);
                          Dtype d2_diff = *(d2 + h * width + w);
                          *(diff++) += factor1 * d1_diff + factor2 * d2_diff;

                      }
                  }
              }
          }
      }
  }

#ifdef CPU_ONLY
  STUB_GPU(MappingBySimilarityLayer);
#endif

INSTANTIATE_CLASS(MappingBySimilarityLayer);
REGISTER_LAYER_CLASS(MappingBySimilarity);
}  // namespace caffe