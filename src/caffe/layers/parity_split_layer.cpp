#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/parity_split_layer.hpp"

namespace caffe {

template <typename Dtype>
void ParitySplitLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->shape(0)%2, 0)<<"Batch size must be a multiple of 2";
  CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0))<<"The number of features must equal to the number of labels";
  CHECK_EQ(bottom[1]->shape(1), 1)<<"The dim of label must be 1";
}

template <typename Dtype>
void ParitySplitLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int n = bottom[0]->shape(0);
  int w = bottom[0]->shape(1);

  std::vector<int> feat_shape;
  feat_shape.push_back(n/2);
//  feat_shape.push_back(c);
//  feat_shape.push_back(h);
  feat_shape.push_back(w);

  // Output of feature map
  top[0]->Reshape(feat_shape);
  top[1]->Reshape(feat_shape);

  std::vector<int> label_shape;
  label_shape.push_back(n/2);
//  label_shape.push_back(1);
//  label_shape.push_back(1);
  label_shape.push_back(1);

  // Output of labels
  top[2]->Reshape(label_shape);
  top[3]->Reshape(label_shape);
}

template <typename Dtype>
void ParitySplitLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data_0 = top[0]->mutable_cpu_data();
  Dtype* top_data_1 = top[1]->mutable_cpu_data();
  int n_images = bottom[0]->shape(0);
  int n_step = bottom[0]->count(1);
  for (int n = 0; n < n_images; ++n) {
    if(n%2 == 0){
      caffe_copy(n_step, bottom_data, top_data_0);
      bottom_data += n_step;
      top_data_0 += n_step;
    }else{
      caffe_copy(n_step, bottom_data, top_data_1);
      bottom_data += n_step;
      top_data_1 += n_step;
    }
  }
  // Label
  const Dtype* label_data = bottom[1]->cpu_data();
  Dtype* top_label_0 = top[2]->mutable_cpu_data();
  Dtype* top_label_1 = top[3]->mutable_cpu_data();
  for (int n = 0; n < n_images; ++n) {
    if(n%2==0){
     *top_label_0 = label_data[n];
     top_label_0++;
    }else{
      *top_label_1 = label_data[n];
      top_label_1++;
    }

  }
}

template <typename Dtype>
void ParitySplitLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                              const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom){
  // Just copy diffs
  if(propagate_down[0]){
    const Dtype* top_diff_0 = top[0]->cpu_diff();
    const Dtype* top_diff_1 = top[1]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    int n_images = bottom[0]->shape(0);
    int n_step = bottom[0]->count(1);
    for (int n = 0; n < n_images; ++n) {
      if(n%2 == 0){
        caffe_copy(n_step, top_diff_0, bottom_diff);
        bottom_diff += n_step;
        top_diff_0 += n_step;
      }else{
        caffe_copy(n_step, top_diff_1, bottom_diff);
        bottom_diff += n_step;
        top_diff_1 += n_step;
      }
    }
  }


}

INSTANTIATE_CLASS(ParitySplitLayer);
REGISTER_LAYER_CLASS(ParitySplit);

}  // namespace caffe
