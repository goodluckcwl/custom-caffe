#include <vector>
#include <cfloat>
#include <caffe/caffe.hpp>

#include "caffe/layers/weight_euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void WeightEuclideanLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                                 const vector<Blob<Dtype> *> &top) {
    main_weight_ = this->layer_param_.weight_euclidean_loss_param().main_weight();
    contour_weight_ = this->layer_param_.weight_euclidean_loss_param().contour_weight();
    coarse_weight_ = this->layer_param_.weight_euclidean_loss_param().coarse_weight();
    normalized_ = this->layer_param_.weight_euclidean_loss_param().normalized();
    normalized_type_ = this->layer_param_.weight_euclidean_loss_param().normalized_type();
    //coarse index
    int coarse_index[13] = {18, 22, 23, 27, 37, 40, 43, 46, 28, 32, 36, 49, 55};

    l2_weight_.ReshapeLike(*bottom[0]);
    // set l2-weight
    int N = l2_weight_.shape(0);
    // Fill weights
    Dtype* p = l2_weight_.mutable_cpu_data();
    for (int n = 0; n < N; ++n){
        for (int j = 0; j < 2; ++j) {
            for (int i = 0; i < 68;++i) {
                //0-16
                if(contour_weight_ != 0 && i>=0 && i<=16){
                    *p = contour_weight_;
                }
                //17-68
                if(main_weight_ != 0 && i>=17 && i<=67){
                    *p = main_weight_;
                }
                //for coarse point
                if(coarse_weight_ != 0){
                    for (int k = 0; k < 13; ++k) {
                        if(i == coarse_index[k] - 1){
                            *p = coarse_weight_;
                            break;
                        }
                    }
                }
                p++;
            }
        }

    }
    //print
//    Dtype* s = l2_weight_.mutable_cpu_data();
//    for (int j = 0; j < 136; ++j) {
//        std::cout<<*(s++)<<' ';
//    }

}

template <typename Dtype>
void WeightEuclideanLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::Reshape(bottom, top);
    CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
    diff_.ReshapeLike(*bottom[0]);
    wdiff_.ReshapeLike(*bottom[0]);
    pupil_dis_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void WeightEuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    // Calculate the inter-pupil distance
    int left_ind[2] =  {36, 39};
    int right_ind[2] = {42, 45};
    const Dtype* src = bottom[1]->cpu_data();
    Dtype* dst = pupil_dis_.mutable_cpu_data();
    int N = pupil_dis_.shape(0);
    int M = pupil_dis_.shape(1);
    for (int n = 0; n < N; ++n) {

        for (int i = 0; i < M; ++i) {
            if (normalized_==1){
                // Inter pupil distance
                if(normalized_type_ == 0){
                    Dtype x1 = *(src + left_ind[0]);
                    Dtype y1 = *(src + M/2 + left_ind[1]);
                    Dtype x2 = *(src + right_ind[0]);
                    Dtype y2 = *(src + M/2 + right_ind[1]);
                    Dtype dis= sqrtf((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
                    *dst = 1/std::max(dis, Dtype(FLT_MIN));
                // Bbox
                }else{
                    Dtype x1 = 1;
                    Dtype x2 = 0;
                    Dtype y1 = 1;
                    Dtype y2 = 0;
                    for (int j = 0; j < M/2; ++j) {
                        Dtype pts_x = *(src + j);
                        Dtype pts_y = *(src + j + M/2);
                        x1 = pts_x < x1 ? pts_x : x1;
                        x2 = pts_x > x2 ? pts_x : x2;
                        y1 = pts_y < y1 ? pts_y : y1;
                        y2 = pts_y > y2 ? pts_y : y2;
                    }

                    Dtype dis = sqrtf((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
                    *dst = 1/std::max(dis, Dtype(FLT_MIN));
                }

            }
            else{
                *dst = 1;
            }

            dst++;
        }
        src += M;
    }

    int count = bottom[0]->count();
    caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());

    // w_i*(x_i - x_i^*)
    caffe_mul(count, l2_weight_.cpu_data(), diff_.cpu_data(), wdiff_.mutable_cpu_data());

    // 1/di*w_i*(x_i - x_i^*)
    caffe_mul(count, pupil_dis_.cpu_data(), wdiff_.cpu_data(), wdiff_.mutable_cpu_data());

    if (bottom.size() == 3) {
    caffe_mul(count, bottom[2]->cpu_data(), diff_.cpu_data(), diff_.mutable_cpu_data());
    }

    Dtype dot = caffe_cpu_dot(count, wdiff_.cpu_data(), diff_.cpu_data());
    Dtype loss = dot / bottom[0]->num() / Dtype(2);
    top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void WeightEuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();

      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          wdiff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(WeightEuclideanLossLayer);
#endif

INSTANTIATE_CLASS(WeightEuclideanLossLayer);
REGISTER_LAYER_CLASS(WeightEuclideanLoss);

}  // namespace caffe
