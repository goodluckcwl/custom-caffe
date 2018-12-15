#include <algorithm>
#include <vector>

#include "caffe/layers/pixelwise_similarity_layer.hpp"

namespace caffe {

template<typename Dtype>
void PixelwiseSimilarityLayer<Dtype>::LayerSetUp(
        const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
    CHECK(this->layer_param_.pixelwise_similarity_param().has_roi())
    << "roi should be specified.";
    roi_ = this->layer_param_.pixelwise_similarity_param().roi();
    CHECK(roi_ > 0)<<"roi must > 0";
    CHECK(bottom[0]->shape(0) % 2 == 0)
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
    std::vector<int> sim_shape;
    top_shape.push_back(n/2);
    top_shape.push_back(height*width);
    top_shape.push_back(1);
    top_shape.push_back(roi_*roi_);

    top[0]->Reshape(top_shape);

    kernel_row_.Reshape(n/2, height*width, 1, channels);
    im_row_.Reshape(n/2, height*width, roi_*roi_, channels);
}

template <typename Dtype>
void PixelwiseSimilarityLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    int n_images = bottom[0]->shape(0);
    int channels = bottom[0]->shape(1);
    int height = bottom[0]->shape(2);
    int width = bottom[0]->shape(3);

    // image to rows; kernel to rows.
    Dtype* data_kernel_row = kernel_row_.mutable_cpu_data();
    Dtype* data_row = im_row_.mutable_cpu_data();
    const int n_step = width*height*channels;
    const int im_channel_size = bottom[0]->shape(2) * bottom[0]->shape(3);
    // For a pair of (image1, image2), we refer image1 as image and image2 as kernel.
    for (int n = 0; n < n_images/2; ++n) {
        const Dtype* data_im = bottom_data + n_step * 2 * n;
        const Dtype* data_kernel = bottom_data + n_step * ( 2 * n + 1);
        // For each pixel in the kernel, we need to calculate its cosine with the image.
        for (int kernel_y = 0; kernel_y < height; ++kernel_y) {
            for(int kernel_x = 0; kernel_x < width; ++kernel_x){
                int x_begin = kernel_x - ceil((roi_ - 1)/2.0);
                int x_end = kernel_x + floor((roi_ - 1)/2.0);
                int y_begin = kernel_y - ceil((roi_ - 1)/2.0);
                int y_end = kernel_y + floor((roi_ - 1)/2.0);

                // Get values in the search region of image
                for (int im_y = y_begin; im_y <= y_end; ++im_y) {
                    for (int im_x = x_begin; im_x <= x_end; ++im_x) {
                        // Features are extracted along the channel axis, which are saved as rows of output image
                        for (int input_channels = 0; input_channels < channels; ++input_channels){
                            // Pading zeros
                            if(im_y < 0 || im_y >= height ||
                                    im_x <0 || im_x >= width){
                                *(data_row++) = 0;
                            }else{
                                *(data_row++) = *(data_im + input_channels * im_channel_size + width * im_y + im_x);
                            }
                        }
                    }
                }

                // Get values of the kernel along the channel axis
                for (int input_channels = 0; input_channels < channels; ++input_channels) {
                    *(data_kernel_row++) = *(data_kernel + input_channels * im_channel_size + width * kernel_y + kernel_x);
                }
            }
        }
    }


    // Calculate the consine similarity scores
    for(int n = 0; n < n_images/2; ++n){
        const Dtype* matA = im_row_.cpu_data() + n * im_row_.count(1);
        const Dtype* vecX = kernel_row_.cpu_data() + n * kernel_row_.count(1);
        for(int output_y = 0; output_y < height; ++output_y){
            for(int output_x = 0; output_x < width; ++output_x){
                //top_data[n * width * height + output_y * width + output_x]
                int channel_ind = output_y * width + output_x;
                const int M = im_row_.shape(2);
                const int N = im_row_.shape(3);
                caffe_cpu_gemv(CblasNoTrans, M, N, (Dtype)1.0,
                               matA + channel_ind * im_row_.count(2),
                               vecX + channel_ind * kernel_row_.count(2),
                               (Dtype)0,
                               top_data + n * top[0]->count(1) + channel_ind * top[0]->count(2));
            }
        }
    }


}

template <typename Dtype>
void PixelwiseSimilarityLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
      std::cout<<"Enter backwared_cpu"<<std::endl;
      const Dtype* top_diff = top[0]->cpu_diff();
      Dtype* kernel_row_diff = kernel_row_.mutable_cpu_diff();
      Dtype* im_row_diff = im_row_.mutable_cpu_diff();
      int n_images = bottom[0]->shape(0);
      int channels = bottom[0]->shape(1);
      int height = bottom[0]->shape(2);
      int width = bottom[0]->shape(3);
      // gradient w.r.t. kernel. Note that we will accumulate diffs.
      for (int n = 0; n < n_images/2; ++n) {
          const Dtype* matA = im_row_.cpu_data() + n * im_row_.count(1);
          for(int output_y = 0; output_y < height; ++output_y) {
              for (int output_x = 0; output_x < width; ++output_x) {
                  int channel_ind = output_y * width + output_x;
                  int M = im_row_.shape(2); // rows of A
                  int N = im_row_.shape(3); // cols of A
                  // y = alpha*Ax+belta*y
                  caffe_cpu_gemv(CblasTrans, M, N,
                                 (Dtype)1.0, // alpha
                                 matA + channel_ind * im_row_.count(2), // A^T
                                 top_diff + n*top[0]->count(1) + channel_ind * top[0]->count(2), // x
                                 (Dtype)0.0,// belta
                                 kernel_row_diff + n * kernel_row_.count(1) + channel_ind * kernel_row_.count(2) // y
                  );
              }
          }
      }
      // gradient w.r.t. image. Note that we will accumulate diffs.
      for (int n = 0; n < n_images/2; ++n) {
          const Dtype* vecX = kernel_row_.cpu_data() + n * kernel_row_.count(1);
          for(int output_y = 0; output_y < height; ++output_y) {
              for (int output_x = 0; output_x < width; ++output_x) {
                  int channel_ind = output_y * width + output_x;
                  int M = top[0]->shape(3); // rows of A. A: roi*roi * 1
                  int N = kernel_row_.shape(3); // cols of B^T. B: channels*1
                  int K = top[0]->shape(2); // cols of A
                  // C=alpha*A*B+beta*C
                  caffe_cpu_gemm(CblasNoTrans,CblasTrans,
                                 M, N, K,
                                 (Dtype)1.0, // alpha
                                 top_diff + n*top[0]->count(1) + channel_ind * top[0]->count(2), // A
                                 vecX + channel_ind * kernel_row_.count(2), // B
                                 (Dtype)0.0,// belta
                                 im_row_diff + n * im_row_.count(1) + channel_ind * im_row_.count(2)// C
                  );
              }
          }
      }

      // row2im
      // Put grads to images and kernels
      const int n_step = width*height*channels;
      const int im_channel_size = bottom[0]->shape(2) * bottom[0]->shape(3);
      // Clear diffs
      Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
      caffe_set(bottom[0]->count(0), Dtype(0.0), bottom_diff);
      for (int n = 0; n < n_images/2; ++n) {
          Dtype* im_diff = bottom_diff + n_step * 2 * n;
          Dtype* kernel_diff = bottom_diff + n_step * ( 2 * n + 1);

          for (int kernel_y = 0; kernel_y < height; ++kernel_y) {
              for (int kernel_x = 0; kernel_x < width; ++kernel_x) {
                  int x_begin = kernel_x - ceil((roi_ - 1)/2.0);
                  int x_end = kernel_x + floor((roi_ - 1)/2.0);
                  int y_begin = kernel_y - ceil((roi_ - 1)/2.0);
                  int y_end = kernel_y + floor((roi_ - 1)/2.0);

                  // Get values in the search region of image
                  int c = 0;
                  for (int im_y = y_begin; im_y <= y_end; ++im_y) {
                      for (int im_x = x_begin; im_x <= x_end; ++im_x) {
                          // Features are extracted along the channel axis, which are saved as rows of output image
                          for (int input_channels = 0; input_channels < channels; ++input_channels){
                              // Pading zeros
                              if(im_y < 0 || im_y >= height ||
                                 im_x <0 || im_x >= width){
                                  // Do nothing
                                  im_row_diff++;
                                  c++;
                              }else{
                                  *(im_diff + input_channels * im_channel_size + width * im_y + im_x) += *(im_row_diff++);
                                  c++;
                              }
                          }
                          //std::cout<<"im_y:"<<im_y<<" im_x:"<<im_x<<std::endl;
                      }
                  }

                  // Get values of the kernel along the channel axis
                  for (int input_channels = 0; input_channels < channels; ++input_channels) {
                      *(kernel_diff + input_channels * im_channel_size + width * kernel_y + kernel_x) += *(kernel_row_diff++) ;
                  }

              }

          }
      }
  }
}


#ifdef CPU_ONLY
STUB_GPU(PixelwiseSimilarityLayer);
#endif

INSTANTIATE_CLASS(PixelwiseSimilarityLayer);
REGISTER_LAYER_CLASS(PixelwiseSimilarity);

}  // namespace caffe
