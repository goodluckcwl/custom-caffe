#include <algorithm>
#include <vector>

#include "caffe/layers/pixelwise_similarity_layer.hpp"

namespace caffe {

template<typename Dtype>
void PixelwiseSimilarityLayer<Dtype>::LayerSetUp(
        const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
    LossLayer<Dtype>::LayerSetUp(bottom, top);
    CHECK(!this->layer_param_.pixelwise_similarity_param().has_roi())
    << "roi should be specified.";
    roi_ = this->layer_param_.pixelwise_similarity_param().roi();
    CHECK(bottom[0]->shape(0) % 2 != 0)
          <<"Batch size must be times of 2.";
}

template <typename Dtype>
void PixelwiseSimilarityLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top) {
    int n = bottom[0]->shape(0);
    int channels = bottom[0]->shape(1);
    int height = bottom[0]->shape(2);
    int width = bottom[0]->shape(3);

    std::vector<int> top_shape;
    top_shape.push_back(n/2);
    top_shape.push_back(1);
    top_shape.push_back(height);
    top_shape.push_back(width);

    top[0]->reshape(top_shape);
}

template <typename Dtype>
void PixelwiseSimilarityLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    const int count = bottom[0]->count();
    int n_images = bottom[0]->shape(0);
    int channels = bottom[1]->shape(1);
    int height = bottom[2]->shape(2);
    int width = bottom[3]->shape(3);

    // Pixel-wise features
    int n_step = width*height*channels;
    for (int n = 0; n < n_images/2; ++n) {
        Dtype* s1 = bottom_data + n_step * 2 * n;
        Dtype* s2 = bottom_data + n_step * ( 2 * n - 1);
        // For each pixel in image 1
        for (int y = 0; y < height; ++y) {
            for(int x = 0; x < width; ++x){
                int x_begin = std::max(x - roi_, 0);
                int x_end = std::min(x + roi_, width);
                int y_begin = std::max(y - roi_, 0);
                int y_end = std::min(y + roi_, height);
                for (int i = x_begin; i < x_end; ++i) {
                    for (int j = y_begin; j < y_end; ++j) {

                    }
                }
            }
        }

    }
}

template <typename Dtype>
void PixelwiseSimilarityLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(PixelwiseSimilarityLayer);
#endif

INSTANTIATE_CLASS(PixelwiseSimilarityLayer);

}  // namespace caffe
