#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include "thrust/functional.h"
#include "thrust/sort.h"

namespace caffe {
    template <typename Dtype>
    __global__ void ScaleGPU(const int nthreads, const Dtype alpha, Dtype* X){
        CUDA_KERNEL_LOOP(index, nthreads) {
            X[index] = alpha * X[index];
        }
    }

    template <typename Dtype>
    __global__ void SoftmaxLossProbComputeGPU(const int nthreads, Dtype* prob_gt_data,
        const int n_step, const int c_step, const Dtype* label, const int dim1, const int channels,
        const int w, const int h,
        const int roi_, SoftmaxWithLossParameter_KernelType type_, Dtype gaussion_sigma_){
        CUDA_KERNEL_LOOP(index, nthreads) {
            const int n = index / channels;
            const int c = index % channels;
            int roi_w = roi_ * 2 + 1;
            int roi_h = roi_ * 2 + 1;
            Dtype *dst = prob_gt_data + n * n_step + c * c_step;
            // For each landmar
            Dtype pts_x = label[n * dim1 + c] * w;
            Dtype pts_y = label[n * dim1 + c + channels] * h;
            pts_x = pts_x - floorf(pts_x) + roi_;
            pts_y = pts_y - floorf(pts_y) + roi_;

            Dtype prob_normalizer = 0;
            for (int y = 0; y < roi_h; ++y) {
                for (int x = 0; x < roi_w; ++x) {
                    switch (type_) {
                        case SoftmaxWithLossParameter_KernelType_EXP:

                            *dst = pow(0.5, MAX(fabs(x - pts_x), fabs(y - pts_y)));
                            break;
                        case SoftmaxWithLossParameter_KernelType_GAUSION:
                            // -1/(2*PI*sigma^2)*exp(-0.5*d^2/sigma^2)
                            *dst = 1 / (sqrt(2 * M_PI) * gaussion_sigma_) *
                                   exp(-0.5 * ((x - pts_x) * (x - pts_x) + (y - pts_y) * (y - pts_y)) /
                                       (gaussion_sigma_ * gaussion_sigma_));
                            break;
                        default:
                            *dst = pow(0.5, MAX(fabs(x - pts_x), fabs(y - pts_y)));
                    }

                    prob_normalizer += *dst;
                    dst++;
                }
            }
            // Normalize
            dst = prob_gt_data + n * n_step + c * c_step;
            for (int i = 0; i < c_step; ++i) {
                dst[i] = dst[i] / prob_normalizer;
            }


        }

    }

  template <typename Dtype>
  __global__ void SoftmaxLossForwardGPU(const int nthreads,
    Dtype* prob_data, const Dtype* label, Dtype* loss,
    const int num, const int dim, const int spatial_dim,
    const bool has_ignore_label_, const int ignore_label_,
    const bool has_hard_mining_label_, const int hard_mining_label_,
    const bool has_cutting_point_, Dtype cutting_point_, Dtype* counts) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      const int n = index / spatial_dim;
      const int s = index % spatial_dim;
      const int label_value = static_cast<int>(label[n * spatial_dim + s]);
      const int channels = dim / spatial_dim;
      if (has_cutting_point_ && prob_data[n * dim + label_value * spatial_dim + s] > cutting_point_
        && (!has_hard_mining_label_ || hard_mining_label_ == label_value)) {
        for (int c = 0; c < channels; ++c) {
          prob_data[n * dim + c * spatial_dim + s] = 0;
        }
        prob_data[n * dim + label_value * spatial_dim + s] = 1;
      }
      if ((has_ignore_label_ && label_value == ignore_label_)) {
        loss[index] = 0;
        counts[index] = 0;
      }
      else {
        loss[index] = -log(max(prob_data[n * dim + label_value * spatial_dim + s],
          Dtype(FLT_MIN)));
        counts[index] = 1;
      }
    }
  }

  template <typename Dtype>
  __global__ void SoftmaxLossCrossEntropyForwardGPU(const int nthreads, Dtype* prob_gt_data,
    const int n_step, const int c_step, const Dtype* prob_data, const int dim0, const int dim,
    const Dtype* label, const int dim1, const int channels, Dtype* loss_data,
    const int w, const int h, const int roi_, Dtype* counts, const Dtype* weights, bool input_weights){
    CUDA_KERNEL_LOOP(index, nthreads) {
        const int n = index / channels;
        const int c = index % channels;
        // Center of the j-th Probability map
        const Dtype cx = label[n * dim1 + c] * w;
        const Dtype cy = label[n * dim1 + c + channels] * h;

        // Region for computing cross entropy loss
        // This is important. Must keep consistent with the groundtruth map.
        int x1 = roundf(cx) - roi_ > 0 ? roundf(cx) - roi_ : 0;
        int y1 = roundf(cy) - roi_ > 0 ? roundf(cy) - roi_ : 0;
        int x2 = roundf(cx) + roi_ < w ? roundf(cx) + roi_ : w - 1;
        int y2 = roundf(cy) + roi_ < h ? roundf(cy) + roi_ : h - 1;
        int roi_w = roi_ * 2 + 1;
        int xb = roundf(cx) - roi_;
        int yb = roundf(cy) - roi_;


        // Weights for each image.
        Dtype weight_value;
        if(input_weights){
            weight_value = weights[n];
        }
        loss_data[index] = 0;

        for (int y = y1; y <= y2; ++y) {
            for (int x = x1; x <= x2; ++x) {
                loss_data[index] -= prob_gt_data[n * n_step + c * c_step + (y - yb) * roi_w + x - xb] *
                        log(MAX(prob_data[n * dim0 + c * dim + y * w + x], Dtype(FLT_MIN)));
            }
        }
        // Weight
        if(input_weights){
            loss_data[index] *= weight_value;
            counts[index] = weight_value;
        }else{
            counts[index] = 1;
        }

    }
  }

  template <typename Dtype>
  __global__ void SoftmaxLossForwardWithWeightsGPU(const int nthreads,
    Dtype* prob_data, const Dtype* label, Dtype* loss,
    const Dtype* weights, const Dtype* class_weights,
    const int num, const int dim, const int spatial_dim,
    const bool has_ignore_label_, const int ignore_label_,
    const bool has_hard_mining_label_, const int hard_mining_label_,
    const bool has_cutting_point_, Dtype cutting_point_, Dtype* counts) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      const int n = index / spatial_dim;
      const int s = index % spatial_dim;
      const int label_value = static_cast<int>(label[n * spatial_dim + s]);
      const Dtype weight_value = weights[n * spatial_dim + s] * class_weights[label_value];
      const int channels = dim / spatial_dim;
      if (has_cutting_point_ && prob_data[n * dim + label_value * spatial_dim + s] > cutting_point_
        && (!has_hard_mining_label_ || hard_mining_label_ == label_value)) {
        for (int c = 0; c < channels; ++c) {
          prob_data[n * dim + c * spatial_dim + s] = 0;
        }
        prob_data[n * dim + label_value * spatial_dim + s] = 1;
      }
      if ((weight_value == 0) || (has_ignore_label_ && label_value == ignore_label_)) {
        loss[index] = 0;
        counts[index] = 0;
      }
      else {
        loss[index] = -weight_value * log(max(prob_data[n * dim + label_value * spatial_dim + s],
          Dtype(FLT_MIN)));
        counts[index] = weight_value;
      }
    }
  }

  template <typename Dtype>
  void SoftmaxWithLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    if(is_soft_classify_){
      Forward_cpu(bottom, top);
      return ;
    }

    softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
    Dtype* prob_data = prob_.mutable_gpu_data();
    const Dtype* label = bottom[1]->gpu_data();
    const int dim = prob_.count() / outer_num_;
    const int nthreads = outer_num_ * inner_num_;
    // Since this memory is not used for anything until it is overwritten
    // on the backward pass, we use it here to avoid having to allocate new GPU
    // memory to accumulate intermediate results in the kernel.
    Dtype* loss_data_test = loss_.mutable_cpu_data();

    Dtype* loss_data = loss_.mutable_gpu_data();
    // Similarly, this memory is never used elsewhere, and thus we can use it
    // to avoid having to allocate additional GPU memory.
    Dtype* counts = counts_.mutable_gpu_data();
    if (bottom.size() == 2 && !is_soft_classify_) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      SoftmaxLossForwardGPU<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS >> >(nthreads, prob_data, label, loss_data,
          outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_,
          has_hard_mining_label_, hard_mining_label_,
          has_cutting_point_, cutting_point_, counts);
    }
    else if( (bottom.size() == 2 || bottom.size() == 3) && is_soft_classify_) {
        int num_images = bottom[0]->shape(0);
        int channels = bottom[0]->shape(1);
        int dim0 = bottom[0]->count() / num_images;
        int dim1 = bottom[1]->count() / num_images;
        int h = sqrtf(bottom[0]->shape(2));
        int w = h;

        //
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
        }

        // Weights
        const Dtype* weights;
        bool input_weights = false;
        if(bottom.size() == 3){
            weights = bottom[2]->gpu_data();
            input_weights = true;
        }

        // Generate groundtruth probability map
        Dtype *prob_gt_data = prob_groundtruth_.mutable_gpu_data();
        int c_step = prob_groundtruth_.count(2);
        int n_step = prob_groundtruth_.count(1);
        // For Each channels
        const int nthreads1 = num_images * channels;
        // NOLINT_NEXT_LINE(whitespace/operators)
        SoftmaxLossProbComputeGPU<Dtype> << <CAFFE_GET_BLOCKS(nthreads1),
                CAFFE_CUDA_NUM_THREADS >> >(nthreads1, prob_gt_data, n_step, c_step,
                label, dim1, channels, w, h, roi_, type_, gaussion_sigma_);

        // Visual
        //prob_gt_data = prob_groundtruth_.mutable_cpu_data();
        //for (int y = 0; y < roi_h; ++y) {
        //    for (int x = 0; x < roi_h; ++x) {
        //        std::cout << prob_gt_data[y * roi_w + x] << " ";
        //   }
        //    std::cout << "" << std::endl;
        //}
        // Compute loss
        SoftmaxLossCrossEntropyForwardGPU<Dtype> << <CAFFE_GET_BLOCKS(nthreads1),
                CAFFE_CUDA_NUM_THREADS >> >(nthreads1, prob_gt_data, n_step, c_step, prob_data,dim0,dim,
                        label, dim1, channels, loss_data, w, h, roi_, counts, weights, input_weights);
    }
    else if (bottom.size() == 3 && !is_soft_classify_) {
      const Dtype* weights = bottom[2]->gpu_data();
      // NOLINT_NEXT_LINE(whitespace/operators)
      SoftmaxLossForwardWithWeightsGPU<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS >> >(nthreads, prob_data, label, loss_data,
          weights, class_weight_.gpu_data(),
          outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_,
          has_hard_mining_label_, hard_mining_label_,
          has_cutting_point_, cutting_point_, counts);
    }
    Dtype loss;
    caffe_gpu_asum(nthreads, loss_data, &loss);
    Dtype valid_count = -1;
    // Only launch another CUDA kernel if we actually need the count of valid
    // outputs.
    if ((normalization_ == LossParameter_NormalizationMode_VALID && has_ignore_label_)
        || (bottom.size() == 3)) {
      caffe_gpu_asum(nthreads, counts, &valid_count);
    }
    top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_,
                                                          valid_count);
    if (top.size() == 2) {
      if(!is_output_landmark_loss_){
          top[1]->ShareData(prob_);
      }
    }
  }

    template <typename Dtype>
    __global__ void SoftmaxLossCrossEntropyBackwardGPU(const int nthreads, const Dtype* prob_gt_data,
                                                       const int n_step, const int c_step, Dtype* bottom_diff, const int dim0, const int dim,
                                                       const Dtype*label, const int dim1,
                                                       const int channels, const int w, const int h, const int roi_, Dtype* counts,
                                                       const Dtype* weights, bool input_weights){
        CUDA_KERNEL_LOOP(index, nthreads) {
            const int n = index / channels;
            const int c = index % channels;

            // Weights
            Dtype weight_value;
            if(input_weights){
                weight_value = weights[n];
            }

            // Center of the j-th Probability map.
            const Dtype cx = label[n * dim1 + c ] * w;
            const Dtype cy = label[n * dim1 + c + channels] * h;

            // Region for computing cross entropy loss
            // This is important. Must keep consistent with the groundtruth map.
            int x1 = roundf(cx - roi_) > 0 ? roundf(cx) - roi_ : 0;
            int y1 = roundf(cy - roi_) > 0 ? roundf(cy) - roi_ : 0;
            int x2 = roundf(cx + roi_) < w ? roundf(cx) + roi_ : w - 1;
            int y2 = roundf(cy + roi_) < h ? roundf(cy) + roi_ : h - 1;
            int roi_w = roi_ * 2 + 1;
            int xb = roundf(cx) - roi_;
            int yb = roundf(cy) - roi_;
            const Dtype* prob_gt_channel_data = prob_gt_data + n * n_step + c * c_step;
            for (int y = y1; y <= y2; ++y) {
                for (int x = x1; x <= x2; ++x){
                    bottom_diff[n * dim0 + c * dim + y * w + x] -=
                            prob_gt_channel_data[ (y - yb) * roi_w + x - xb];
                }
            }
            // Weights
            if(input_weights){
                for (int y = 0; y < h; ++y) {
                    for (int x = 0; x < w; ++x) {
                        bottom_diff[n * dim0 + c * dim + y * w + x] *= weight_value;
                    }
                }
                counts[index] = weight_value;
            }else{
                counts[index] = 1;
            }

        }
    }

  template <typename Dtype>
  __global__ void SoftmaxLossBackwardGPU(const int nthreads, const Dtype* top,
                                         const Dtype* label, Dtype* bottom_diff, const int num, const int dim,
                                         const int spatial_dim, const bool has_ignore_label_,
                                         const int ignore_label_, Dtype* counts) {
    const int channels = dim / spatial_dim;

    CUDA_KERNEL_LOOP(index, nthreads) {
      const int n = index / spatial_dim;
      const int s = index % spatial_dim;
      const int label_value = static_cast<int>(label[n * spatial_dim + s]);

      if (has_ignore_label_ && label_value == ignore_label_) {
        for (int c = 0; c < channels; ++c) {
          bottom_diff[n * dim + c * spatial_dim + s] = 0;
        }
        counts[index] = 0;
      }
      else {
        bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
        counts[index] = 1;
      }
    }
  }

  template <typename Dtype>
  __global__ void SoftmaxLossBackwardWithWeightsGPU(const int nthreads, const Dtype* top,
                                                    const Dtype* weights, const Dtype* class_weight,
                                                    const Dtype* label, Dtype* bottom_diff, const int num, const int dim,
                                                    const int spatial_dim, const bool has_ignore_label_,
                                                    const int ignore_label_, Dtype* counts) {
    const int channels = dim / spatial_dim;

    CUDA_KERNEL_LOOP(index, nthreads) {
      const int n = index / spatial_dim;
      const int s = index % spatial_dim;
      const int label_value = static_cast<int>(label[n * spatial_dim + s]);
      const Dtype weight_value = weights[n * spatial_dim + s];
      if ((has_ignore_label_ && label_value == ignore_label_) || (weight_value == 0)) {
        for (int c = 0; c < channels; ++c) {
          bottom_diff[n * dim + c * spatial_dim + s] = 0;
        }
        counts[index] = 0;
      }
      else {
        bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
        for (int c = 0; c < channels; ++c) {
          bottom_diff[n * dim + c * spatial_dim + s] *= weight_value * class_weight[c];
        }
        counts[index] = weight_value;
      }
    }
  }

  template <typename Dtype>
  __global__ void Threshold(const int n, const Dtype* loss, Dtype threshold, Dtype* out) {
    CUDA_KERNEL_LOOP(index, n) {
      out[index] = loss[index] < threshold ? 0 : out[index];
    }
  }

  template <typename Dtype>
  __global__ void ThresholdWithLabel(const int n, const Dtype* loss, Dtype threshold, 
    const Dtype* label, Dtype hard_mining_label, Dtype* out) {
    CUDA_KERNEL_LOOP(index, n) {
      out[index] = (label[index] == hard_mining_label &&loss[index] < threshold) ? 0 : out[index];
    }
  }

  template <typename Dtype>
  void SoftmaxWithLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                                 const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[1]) {
      LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to label inputs.";
    }
    if (propagate_down[0]) {
      /*if(is_soft_classify_){
        Backward_cpu(top, propagate_down, bottom);
        return ;
      }*/

      if (has_hard_ratio_ && bottom.size() == 3) {
        caffe_copy(outer_num_ * inner_num_, loss_.cpu_data(), loss_.mutable_cpu_diff());
        std::sort(loss_.mutable_cpu_diff(), loss_.mutable_cpu_diff() + outer_num_ * inner_num_);//thrust::sort
        Dtype loss_threshold = loss_.cpu_diff()[(int)(outer_num_ * inner_num_ * (1 - hard_ratio_))];
        if (has_hard_mining_label_) {
          // NOLINT_NEXT_LINE(whitespace/operators)
          ThresholdWithLabel<Dtype> << <CAFFE_GET_BLOCKS(outer_num_ * inner_num_), CAFFE_CUDA_NUM_THREADS >> >(
            outer_num_ * inner_num_, loss_.gpu_data(), loss_threshold, 
            bottom[1]->gpu_data(), hard_mining_label_, bottom[2]->mutable_gpu_data());
        }
        else {
          // NOLINT_NEXT_LINE(whitespace/operators)
          Threshold<Dtype> << <CAFFE_GET_BLOCKS(outer_num_ * inner_num_), CAFFE_CUDA_NUM_THREADS >> >(
            outer_num_ * inner_num_, loss_.gpu_data(), loss_threshold, bottom[2]->mutable_gpu_data());
        }
      }

      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
      const Dtype* prob_data = prob_.gpu_data();
      const Dtype* top_data = top[0]->gpu_data();
      caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
      const Dtype* label = bottom[1]->gpu_data();
      const int dim = prob_.count() / outer_num_;
      const int nthreads = outer_num_ * inner_num_;
      // Since this memory is never used for anything else,
      // we use to to avoid allocating new GPU memory.
      Dtype* counts = counts_.mutable_gpu_data();
      if (bottom.size() == 2 && !is_soft_classify_) {
        // NOLINT_NEXT_LINE(whitespace/operators)
        SoftmaxLossBackwardGPU<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
          CAFFE_CUDA_NUM_THREADS >> >(nthreads, top_data, label, bottom_diff,
          outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts);
      }
      else if( (bottom.size() == 2 || bottom.size() == 3) && is_soft_classify_){
          int num_images = bottom[0]->shape(0);
          int channels = bottom[0]->shape(1);
          int dim0 = bottom[0]->count()/num_images;
          int dim1 = bottom[1]->count()/num_images;
          int w = sqrtf(bottom[0]->shape(2));
          int h = w;
          const Dtype* prob_gt_data = prob_groundtruth_.gpu_data();
          int n_step = prob_groundtruth_.count(1);
          int c_step = prob_groundtruth_.count(2);

          // Weights of each image
          const Dtype* weights;
          bool input_weights = false;
          if(bottom.size() == 3){
              weights = bottom[2]->gpu_data();
              input_weights = true;
          }

          // NOLINT_NEXT_LINE(whitespace/operators)
          SoftmaxLossCrossEntropyBackwardGPU<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
                  CAFFE_CUDA_NUM_THREADS >> >(nthreads, prob_gt_data, n_step, c_step,
                          bottom_diff, dim0, dim, label, dim1, channels,
                  w, h, roi_, counts, weights, input_weights);

          // For profile landmark detection.
          for (int n = 0; n < num_images; ++n) {
              // Set diff to zero if the profile type is different.
              if(is_profile_ && profile_type_ != cur_profile_type_[n]) {
                  caffe_gpu_set(dim0, Dtype(0), bottom_diff + n * dim0);
              }
          }

      }
      else if (bottom.size() == 3 && !is_soft_classify_) {
        const Dtype* weights = bottom[2]->gpu_data();
        // NOLINT_NEXT_LINE(whitespace/operators)
        SoftmaxLossBackwardWithWeightsGPU<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
          CAFFE_CUDA_NUM_THREADS >> >(nthreads, top_data, 
          weights, class_weight_.gpu_data(), label, bottom_diff,
          outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts);
      }

      Dtype valid_count = -1;
      // Only launch another CUDA kernel if we actually need the count of valid
      // outputs.
      if ((normalization_ == LossParameter_NormalizationMode_VALID && has_ignore_label_)
          || (bottom.size() == 3)) {
        caffe_gpu_asum(nthreads, counts, &valid_count);
      }
      const Dtype loss_weight = top[0]->cpu_diff()[0] /
        get_normalizer(normalization_, valid_count);
      caffe_gpu_scal(prob_.count(), loss_weight, bottom_diff);
    }
  }

  INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxWithLossLayer);

}  // namespace caffe
