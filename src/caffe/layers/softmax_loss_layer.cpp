#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

    template<typename Dtype>
    void SoftmaxWithLossLayer<Dtype>::LayerSetUp(
            const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
        LossLayer<Dtype>::LayerSetUp(bottom, top);
        is_soft_classify_ = this->layer_param_.softmax_loss_param().is_soft_classify();
        is_output_landmark_loss_ = this->layer_param_.softmax_loss_param().is_output_landmark_loss();
        roi_ = this->layer_param_.softmax_loss_param().roi();
        type_ = this->layer_param_.softmax_loss_param().type();
        gaussion_sigma_ = this->layer_param_.softmax_loss_param().sigma();
        is_profile_ = this->layer_param_.softmax_loss_param().is_profile();
        profile_type_ = this->layer_param_.softmax_loss_param().profile_type();
        normalize_type_ =
                this->layer_param_.softmax_param().normalize_type();
        if (normalize_type_ == "Softmax") {
            LayerParameter softmax_param(this->layer_param_);
            softmax_param.set_type("Softmax");
            // By chenweiliang. 2018.01.13
            softmax_param.clear_loss_weight();

            softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
            softmax_bottom_vec_.clear();
            softmax_bottom_vec_.push_back(bottom[0]);
            softmax_top_vec_.clear();
            softmax_top_vec_.push_back(&prob_);
            softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
        } else if (normalize_type_ == "L2" || normalize_type_ == "L1") {
            LayerParameter normalize_param(this->layer_param_);
            normalize_param.set_type("Normalize");
            softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(normalize_param);
            softmax_bottom_vec_.clear();
            softmax_bottom_vec_.push_back(bottom[0]);
            softmax_top_vec_.clear();
            softmax_top_vec_.push_back(&prob_);
            softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
        } else {
            NOT_IMPLEMENTED;
        }

        has_ignore_label_ =
                this->layer_param_.loss_param().has_ignore_label();
        if (has_ignore_label_) {
            ignore_label_ = this->layer_param_.loss_param().ignore_label();
        }
        has_hard_ratio_ =
                this->layer_param_.softmax_param().has_hard_ratio();
        if (has_hard_ratio_) {
            hard_ratio_ = this->layer_param_.softmax_param().hard_ratio();
            CHECK_GE(hard_ratio_, 0);
            CHECK_LE(hard_ratio_, 1);
        }
        has_cutting_point_ =
                this->layer_param_.softmax_param().has_cutting_point();
        if (has_cutting_point_) {
            cutting_point_ = this->layer_param_.softmax_param().cutting_point();
            CHECK_GE(cutting_point_, 0);
            CHECK_LE(cutting_point_, 1);
        }
        has_hard_mining_label_ = this->layer_param_.softmax_param().has_hard_mining_label();
        if (has_hard_mining_label_) {
            hard_mining_label_ = this->layer_param_.softmax_param().hard_mining_label();
        }
        has_class_weight_ = (this->layer_param_.softmax_param().class_weight_size() != 0);
        softmax_axis_ =
                bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
        if (has_class_weight_) {
            class_weight_.Reshape({bottom[0]->shape(softmax_axis_)});
            CHECK_EQ(this->layer_param_.softmax_param().class_weight().size(), bottom[0]->shape(softmax_axis_));
            for (int i = 0; i < bottom[0]->shape(softmax_axis_); i++) {
                class_weight_.mutable_cpu_data()[i] = (Dtype) this->layer_param_.softmax_param().class_weight(i);
            }
        } else {
            if (bottom.size() == 3) {
                class_weight_.Reshape({bottom[0]->shape(softmax_axis_)});
                for (int i = 0; i < bottom[0]->shape(softmax_axis_); i++) {
                    class_weight_.mutable_cpu_data()[i] = (Dtype) 1.0;
                }
            }
        }
        if (!this->layer_param_.loss_param().has_normalization() &&
            this->layer_param_.loss_param().has_normalize()) {
            normalization_ = this->layer_param_.loss_param().normalize() ?
                             LossParameter_NormalizationMode_VALID :
                             LossParameter_NormalizationMode_BATCH_SIZE;
        } else {
            normalization_ = this->layer_param_.loss_param().normalization();
        }
    }

    template<typename Dtype>
    void SoftmaxWithLossLayer<Dtype>::Reshape(
            const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
        LossLayer<Dtype>::Reshape(bottom, top);
        softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
        softmax_axis_ =
                bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
        outer_num_ = bottom[0]->count(0, softmax_axis_);
        inner_num_ = bottom[0]->count(softmax_axis_ + 1);
        counts_.Reshape({outer_num_, inner_num_});
        loss_.Reshape({outer_num_, inner_num_});
        if (is_soft_classify_) {
            CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0))
                << "Number of images must match number of predictions";
            CHECK_EQ(bottom[1]->shape(1) / 2, bottom[0]->shape(1))
                << "Number of prob map must equal number of landmarks";
        } else {
            CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
                << "Number of labels must match number of predictions; "
                << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
                << "label count (number of labels) must be N*H*W, "
                << "with integer values in {0, 1, ..., C-1}.";
        }

        if (bottom.size() == 3 && !is_soft_classify_) {
            CHECK_EQ(outer_num_ * inner_num_, bottom[2]->count())
                << "Number of loss weights must match number of label.";
        }
        if (bottom.size() == 3 && is_soft_classify_) {
            CHECK_EQ(bottom[0]->shape(0), bottom[2]->count())
                << "Number of loss weights must match number of image.";
        }
        if (top.size() >= 2) {
            // By chenweiliang. 2018.11.13
            // softmax output
            if (is_output_landmark_loss_){
                int num = bottom[0]->shape(0);
                int channels = bottom[0]->shape(1);
                std::vector<int> tmp;
                tmp.push_back(num);
                tmp.push_back(channels);
                top[1]->Reshape(tmp);
            }else{
                top[1]->ReshapeLike(*bottom[0]);
            }

        }

        if (has_class_weight_) {
            CHECK_EQ(class_weight_.count(), bottom[0]->shape(1));
        }

        if (is_soft_classify_) {
            int roi_w = roi_ * 2 + 1;
            int roi_h = roi_ * 2 + 1;
            int num = bottom[0]->shape(0);
            int channels = bottom[0]->shape(1);
            prob_groundtruth_.Reshape(num, channels, roi_w, roi_h);
        }

    }

    template<typename Dtype>
    Dtype SoftmaxWithLossLayer<Dtype>::get_normalizer(
            LossParameter_NormalizationMode normalization_mode, Dtype valid_count) {
        Dtype normalizer;
        switch (normalization_mode) {
            case LossParameter_NormalizationMode_FULL:
                normalizer = Dtype(outer_num_ * inner_num_);
                break;
            case LossParameter_NormalizationMode_VALID:
                if (valid_count == -1) {
                    normalizer = Dtype(outer_num_ * inner_num_);
                } else {
                    normalizer = valid_count;
                }
                break;
            case LossParameter_NormalizationMode_BATCH_SIZE:
                normalizer = Dtype(outer_num_);
                break;
            case LossParameter_NormalizationMode_NONE:
                normalizer = Dtype(1);
                break;
            default:
                LOG(FATAL) << "Unknown normalization mode: "
                           << LossParameter_NormalizationMode_Name(normalization_mode);
        }
        // Some users will have no labels for some examples in order to 'turn off' a
        // particular loss in a multi-task setup. The max prevents NaNs in that case.
        return std::max(Dtype(1.0), normalizer);
    }

    template<typename Dtype>
    void SoftmaxWithLossLayer<Dtype>::Forward_cpu(
            const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
        // The forward pass computes the softmax prob values.
        softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
        const Dtype *prob_data = prob_.cpu_data();
        const Dtype *label = bottom[1]->cpu_data();
        int dim = prob_.count() / outer_num_;
        Dtype count = 0;
        Dtype loss = 0;

        if (bottom.size() == 2 && !is_soft_classify_) {
            for (int i = 0; i < outer_num_; ++i) {
                for (int j = 0; j < inner_num_; j++) {
                    const int label_value = static_cast<int>(label[i * inner_num_ + j]);
                    if (has_ignore_label_ && label_value == ignore_label_) {
                        continue;
                    }
                    DCHECK_GE(label_value, 0);
                    DCHECK_LT(label_value, prob_.shape(softmax_axis_));
                    loss -= log(std::max(prob_data[i * dim + label_value * inner_num_ + j],
                                         Dtype(FLT_MIN)));
                    count += 1;
                }
            }
        } else if ( (bottom.size() == 2 || bottom.size() == 3)
                    && is_soft_classify_) {
            int num_images = bottom[0]->shape(0);
            int channels = bottom[0]->shape(1);
            int dim0 = bottom[0]->count() / num_images;
            int dim1 = bottom[1]->count() / num_images;
            int h = sqrtf(bottom[0]->shape(2));
            int w = h;

            // Weight
            const Dtype* weights;
            if(bottom.size() == 3){
                weights = bottom[2]->cpu_data();
            }

            // Generate groundtruth probability map
            int roi_w = roi_ * 2 + 1;
            int roi_h = roi_ * 2 + 1;
            Dtype *prob_gt_data = prob_groundtruth_.mutable_cpu_data();
            int c_step = prob_groundtruth_.count(2);
            int n_step = prob_groundtruth_.count(1);
            for (int n = 0; n < num_images; ++n) {
                if(is_profile_){
                    // Determine left or right profile
                    Dtype nose_x = label[n * dim1 + 19] * w;
                    Dtype contour_x = label[n * dim1 + 3] * w;
                    if(nose_x>contour_x){
                        cur_profile_type_.push_back(2); // Right profile
                    }else{
                        cur_profile_type_.push_back(1); // Left profile
                    }
                }

                for (int c = 0; c < channels; ++c) {
                    Dtype *dst = prob_gt_data + n * n_step + c * c_step;

                    // For each landmark
                    Dtype pts_x = label[n * dim1 + c] * w;
                    Dtype pts_y = label[n * dim1 + c + channels] * h;
                    pts_x = pts_x - floorf(pts_x) + roi_;
                    pts_y = pts_y - floorf(pts_y) + roi_;
                    Dtype prob_normalizer = 0;
                    for (int y = 0; y < roi_h; ++y) {
                        for (int x = 0; x < roi_w; ++x) {
                            switch (type_) {
                                case SoftmaxWithLossParameter_KernelType_EXP:

                                    *dst = pow(0.5, std::max(fabs(x - pts_x), fabs(y - pts_y)));
                                    break;
                                case SoftmaxWithLossParameter_KernelType_GAUSION:
                                    // -1/(2*PI*sigma^2)*exp(-0.5*d^2/sigma^2)
                                    *dst = 1 / (sqrt(2 * M_PI) * gaussion_sigma_) *
                                           exp(-0.5 * ((x - pts_x) * (x - pts_x) + (y - pts_y) * (y - pts_y)) /
                                               (gaussion_sigma_ * gaussion_sigma_));
                                    break;
                                default:
                                    LOG(FATAL) << "Unknown kernel type.";
                            }

                            prob_normalizer += *dst;
                            dst++;
                        }
                    }
                    // Normalize
                    caffe_scal(c_step, Dtype(1.0 / prob_normalizer), prob_gt_data + n * n_step + c * c_step);
                    // Visual
                    //dst = prob_gt_data + n * n_step + c * c_step;
                    //for (int y = 0; y < roi_h; ++y) {
                    //    for (int x = 0; x < roi_h; ++x) {
                    //        std::cout<<*dst<<" ";
                    //        dst++;
                    //    }
                    //    std::cout<<""<<std::endl;
                }
            }

            Dtype *loss_per_channel;
            if(is_output_landmark_loss_){
                loss_per_channel = top[1]->mutable_cpu_data();
            }
            int loss_per_channel_idx = 0;
            for (int i = 0; i < num_images; ++i) {
                Dtype weight_value;
                if(bottom.size()==3){
                    weight_value = weights[i];
                }
                for (int j = 0; j < channels; ++j) {
                    // Center of the j-th Probability map
                    const Dtype cx = label[i * dim1 + j] * w;
                    const Dtype cy = label[i * dim1 + j + channels] * h;
                    DCHECK_GE(cx, 0);
                    DCHECK_GE(cy, 0);
                    DCHECK_LT(cy, h - 1);
                    DCHECK_LT(cx, w - 1);
                    // Region for computing cross entropy loss
                    // This is important. Must keep consistent with the groundtruth map.
                    int x1 = roundf(cx) - roi_ > 0 ? roundf(cx) - roi_ : 0;
                    int y1 = roundf(cy) - roi_ > 0 ? roundf(cy) - roi_ : 0;
                    int x2 = roundf(cx) + roi_ < w ? roundf(cx) + roi_ : w - 1;
                    int y2 = roundf(cy) + roi_ < h ? roundf(cy) + roi_ : h - 1;
                    int roi_w = roi_ * 2 + 1;
                    int xb = roundf(cx) - roi_;
                    int yb = roundf(cy) - roi_;
                    Dtype loss_channel = 0;
                    for (int y = y1; y <= y2; ++y) {
                        for (int x = x1; x <= x2; ++x) {
                            loss -= prob_gt_data[i * n_step + j * c_step + (y - yb) * roi_w + x - xb] *
                                    log(std::max(prob_data[i * dim0 + j * dim + y * w + x], Dtype(FLT_MIN)));
                            if(is_output_landmark_loss_){
                                loss_channel -= prob_gt_data[i * n_step + j * c_step + (y - yb) * roi_w + x - xb] *
                                                log(std::max(prob_data[i * dim0 + j * dim + y * w + x], Dtype(FLT_MIN)));
                            }
                        }
                    }
                    if(is_output_landmark_loss_){
                        loss_per_channel[loss_per_channel_idx] = loss_channel;
                        loss_per_channel_idx++;
                    }

                    // Weight
                    if(bottom.size() == 3){
                        loss *= weight_value;
                        count +=weight_value;
                    }else{
                        count += 1;
                    }

                }
            }

        }
  else if(bottom.size() == 3 && !is_soft_classify_) {
    const Dtype* weights = bottom[2]->cpu_data();
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; j++) {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        const Dtype weight_value = weights[i * inner_num_ + j] * (has_class_weight_? class_weight_.cpu_data()[label_value] : 1.0);
        if (weight_value == 0) continue;
        if (has_ignore_label_ && label_value == ignore_label_) {
          continue;
        }
        DCHECK_GE(label_value, 0);
        DCHECK_LT(label_value, prob_.shape(softmax_axis_));
        loss -= weight_value * log(std::max(prob_data[i * dim + label_value * inner_num_ + j],
          Dtype(FLT_MIN)));
        count += weight_value;
      }
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, count);
  if (top.size() == 2) {
      if(!is_output_landmark_loss_){
          top[1]->ShareData(prob_);
      }
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    std::cout<<std::endl;
    int dim = prob_.count() / outer_num_;
    Dtype count = 0;
    if (bottom.size() == 2 && !is_soft_classify_) {
      for (int i = 0; i < outer_num_; ++i) {
        for (int j = 0; j < inner_num_; ++j) {
          const int label_value = static_cast<int>(label[i * inner_num_ + j]);
          if (has_ignore_label_ && label_value == ignore_label_) {
            for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
              bottom_diff[i * dim + c * inner_num_ + j] = 0;
            }
          }
          else {
            bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
            count += 1;
          }
        }
      }
    }
    else if(propagate_down[0]
            && (bottom.size() == 2 || bottom.size() == 3)
            && is_soft_classify_){
        int num_images = bottom[0]->shape(0);
        int channels = bottom[0]->shape(1);
        int dim0 = bottom[0]->count()/num_images;
        int dim1 = bottom[1]->count()/num_images;
        int w = sqrtf(bottom[0]->shape(2));
        int h = w;

        // Weights
        const Dtype* weights;
        if(bottom.size() == 3){
            weights = bottom[2]->cpu_data();
        }


        const Dtype* prob_gt_data = prob_groundtruth_.cpu_data();
        int n_step = prob_groundtruth_.count(1);
        int c_step = prob_groundtruth_.count(2);
        for (int i = 0; i < num_images; ++i) {
            const Dtype weight_value = weights[i];
            for (int j = 0; j < channels; ++j) {
                // Center of the j-th Probability map.
                const Dtype cx = label[i * dim1 + j ] * w;
                const Dtype cy = label[i * dim1 + j + channels] * h;
                DCHECK_GE(cx, 0);
                DCHECK_GE(cy, 0);
                DCHECK_LT(cy, h - 1);
                DCHECK_LT(cx, w - 1);
                // Region for computing cross entropy loss
                int x1 = roundf(cx - roi_) > 0 ? roundf(cx) - roi_ : 0;
                int y1 = roundf(cy - roi_) > 0 ? roundf(cy) - roi_ : 0;
                int x2 = roundf(cx + roi_) < w ? roundf(cx) + roi_ : w - 1;
                int y2 = roundf(cy + roi_) < h ? roundf(cy) + roi_ : h - 1;
                int roi_w = roi_ * 2 + 1;
                int xb = roundf(cx) - roi_;
                int yb = roundf(cy) - roi_;
                const Dtype* prob_gt_channel_data = prob_gt_data + i * n_step + j * c_step;
                for (int y = y1; y <= y2; ++y) {
                    for (int x = x1; x <= x2; ++x){
                        bottom_diff[i * dim0 + j * dim + y * w + x] -=
                                prob_gt_channel_data[ (y - yb) * roi_w + x - xb];
                    }

                }

                // Weights
                if(bottom.size() == 3){
                    for (int y = 0; y < h; ++y) {
                        for (int x = 0; x < w; ++x) {
                            bottom_diff[i * dim0 + j * dim + y * w + x] *= weight_value;
                        }
                    }
                    count += weight_value;
                }else{
                    count += 1;
                }
            }
            // Set diff to zero if the profile type is different.
            if(is_profile_ && profile_type_ != cur_profile_type_[i]) {
                caffe_set(dim0, Dtype(0), bottom_diff + i * dim0);
            }
        }
    }
    else if (propagate_down[0] && bottom.size() == 3 && !is_soft_classify_) {
      const Dtype* weights = bottom[2]->cpu_data();
      for (int i = 0; i < outer_num_; ++i) {
        for (int j = 0; j < inner_num_; ++j) {
          const int label_value = static_cast<int>(label[i * inner_num_ + j]);
          const Dtype weight_value = weights[i * inner_num_ + j];
          if (has_ignore_label_ && label_value == ignore_label_) {
            for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
              bottom_diff[i * dim + c * inner_num_ + j] = 0;
            }
          }
          else {
            bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
            for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
              bottom_diff[i * dim + c * inner_num_ + j] *= weight_value * (has_class_weight_ ? class_weight_.cpu_data()[label_value] : 1.0);
            }
            if(weight_value != 0) count += weight_value;
          }
        }
      }
    }
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] /
                        get_normalizer(normalization_, count);
    caffe_scal(prob_.count(), loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithLossLayer);
REGISTER_LAYER_CLASS(SoftmaxWithLoss);

}  // namespace caffe
