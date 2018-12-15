#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/pixelwise_similarity_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void im2row_gpu_kernel(const int n_kernels, const Dtype* data_im,
                                  const int channels, const int height, const int width, const int roi,
                                  const Dtype* data_kernel,
                                  Dtype* data_row, Dtype* data_kernel_row) {
  CUDA_KERNEL_LOOP(index, n_kernels) {
    const int kernel_x = index % width;
    const int kernel_y = (index / width) % height;
    const int channel_idx = index / (width * height);

    const int x_begin = kernel_x - ceil((roi - 1)/2.0);
    const int x_end = kernel_x + floor((roi - 1)/2.0);
    const int y_begin = kernel_y - ceil((roi - 1)/2.0);
    const int y_end = kernel_y + floor((roi - 1)/2.0);

    // Get the data on the current channel
    const Dtype* data_im_ptr = data_im;
    data_im_ptr += channel_idx * height * width;
    const Dtype* data_kernel_ptr = data_kernel;
    data_kernel_ptr += channel_idx * height * width;

    // data_row:(n/2, w*h, roi*roi, channels)
    Dtype* data_row_ptr = data_row;
    data_row_ptr += (kernel_y * width + kernel_x) * roi * roi * channels;
    // data_kernel:(n/2, w*h, 1, channels)
    Dtype* data_kernel_row_ptr = data_kernel_row;
    data_kernel_row_ptr += (kernel_y * width + kernel_x) * channels;

    // Copy to im_row
    int roi_idx = 0;
    for (int im_y = y_begin; im_y <= y_end; ++im_y) {
      for (int im_x = x_begin; im_x <= x_end; ++im_x) {
          if(im_y < 0 || im_y >= height ||
             im_x <0 || im_x >= width){
              data_row_ptr[roi_idx * channels + channel_idx] =  0;
          }else{
              data_row_ptr[roi_idx * channels + channel_idx] =  data_im_ptr[im_y * width + im_x];
          }

        roi_idx++;
      }
    }
    // Copy to kernel_row
    data_kernel_row_ptr[channel_idx] = data_kernel_ptr[kernel_y * width + kernel_x];
  }
}

template <typename Dtype>
__global__ void row2kernel_gpu_kernel(const int n_kernels, Dtype *kernel_diff,
                                      const int channels, const int height, const int width, const int roi,
                                      const Dtype *kernel_row_diff
){
    CUDA_KERNEL_LOOP(index, n_kernels) {
        const int w = index % width;
        const int h = index / width;

        // kernel_row: (n/2, w*h, 1, channels)
        const Dtype *kernel_row_diff_ptr = kernel_row_diff;
        kernel_row_diff_ptr += (h * width + w) * channels;
        // kernel: (n/2, channels, h, w)
        Dtype *kernel_diff_ptr = kernel_diff;
        kernel_diff_ptr += h * width + w;

        const int channel_size = height * width;
        for (int i = 0; i < channels; ++i) {
            kernel_diff_ptr[i * channel_size] = kernel_row_diff_ptr[i];
        }
    }

}

template <typename Dtype>
__global__ void row2im_gpu_kernel(const int n_kernels, Dtype *im_diff,
                                  const int channels, const int height, const int width, const int roi,
                                  const Dtype *im_row_diff
){
    CUDA_KERNEL_LOOP(index, n_kernels) {
        const int w = index % width;
        const int h = (index / width) % height;
        const int c = index / (width * height);

        const int roi_left = ceil((roi - 1)/2.0);
        const int roi_right = floor((roi - 1)/2.0);

        // Find the kernel that covers this pixel
        const int x_begin = MAX(w - roi_right, 0);
        const int x_end = MIN(w + roi_left, width-1);
        const int y_begin = MAX(h - roi_right, 0);
        const int y_end = MIN(h + roi_left, height-1);
//        printf("xbegin:%d,xend:%d\n", x_begin, x_end);
        // im_row: (n/2, w*h, roi*roi, channels)
        Dtype val = 0;
        const int channel_size = roi*roi*channels;
        // For each kernel
        for (int kernel_y = y_begin; kernel_y <= y_end; ++kernel_y) {
            for (int kernel_x = x_begin; kernel_x <= x_end ; ++kernel_x) {

                // The left top of the roi region.
                const int x1 = kernel_x - roi_left;
                const int y1 = kernel_y - roi_left;
                const int roi_idx = (h - y1) * roi + (w - x1);
                int channel_idx = kernel_y * width + kernel_x;
                val += im_row_diff[channel_idx * channel_size + roi_idx * channels + c];
            }
        }
        im_diff[index] = val;
    }

}

template <typename Dtype>
void PixelwiseSimilarityLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if(false){
      this->Forward_cpu(bottom, top);
      return;
  }

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int n_images = bottom[0]->shape(0);
  int channels = bottom[0]->shape(1);
  int height = bottom[0]->shape(2);
  int width = bottom[0]->shape(3);

  // image to rows; kernel to rows. We launch channels*height*width gpu kernels. Each
  // kernel responsible for copying a single-channel grid.
  int num_kernels = channels * height * width;
  Dtype* data_kernel_row = kernel_row_.mutable_gpu_data();
  Dtype* data_row = im_row_.mutable_gpu_data();
  // For each (image, kernel) pair.
  const int n_step = width*height*channels;
  for (int n = 0; n < n_images/2; ++n) {
    const Dtype* data_im = bottom_data + n_step * 2 * n;
    const Dtype* data_kernel = bottom_data + n_step * ( 2 * n + 1);
    im2row_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
            CAFFE_CUDA_NUM_THREADS>>>(
                    num_kernels, data_im, channels, height, width, roi_,
                            data_kernel,
                            data_row + n * im_row_.count(1),
                            data_kernel_row + n * kernel_row_.count(1));
  }

    // Calculate the consine similarity scores
    for(int n = 0; n < n_images/2; ++n){
        const Dtype* matA = im_row_.gpu_data() + n * im_row_.count(1);
        const Dtype* vecX = kernel_row_.gpu_data() + n * kernel_row_.count(1);
        for(int output_y = 0; output_y < height; ++output_y){
            for(int output_x = 0; output_x < width; ++output_x){
                //top_data[n * width * height + output_y * width + output_x]
                int channel_ind = output_y * width + output_x;
                const int M = im_row_.shape(2);
                const int N = im_row_.shape(3);
                caffe_gpu_gemv(CblasNoTrans, M, N, (Dtype)1.0,
                               matA + channel_ind * im_row_.count(2),
                               vecX + channel_ind * kernel_row_.count(2),
                               (Dtype)0,
                               top_data + n * top[0]->count(1) + channel_ind * top[0]->count(2));
            }
        }
    }



  CUDA_POST_KERNEL_CHECK;
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}


template <typename Dtype>
void PixelwiseSimilarityLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {

      if(false){
          this->Backward_cpu(top, propagate_down, bottom);

          return;
      }
      const Dtype* top_diff = top[0]->gpu_diff();
      Dtype* kernel_row_diff = kernel_row_.mutable_gpu_diff();
      Dtype* im_row_diff = im_row_.mutable_gpu_diff();
      int n_images = bottom[0]->shape(0);
      int channels = bottom[0]->shape(1);
      int height = bottom[0]->shape(2);
      int width = bottom[0]->shape(3);
      // gradient w.r.t. kernel. Note that we will accumulate diffs.
      for (int n = 0; n < n_images/2; ++n) {
          const Dtype* matA = im_row_.gpu_data() + n * im_row_.count(1);
          for(int output_y = 0; output_y < height; ++output_y) {
              for (int output_x = 0; output_x < width; ++output_x) {
                  int channel_ind = output_y * width + output_x;
                  int M = im_row_.shape(2); // rows of A
                  int N = im_row_.shape(3); // cols of A
                  // y = alpha*Ax+belta*y
                  caffe_gpu_gemv(CblasTrans, M, N,
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
          const Dtype* vecX = kernel_row_.gpu_data() + n * kernel_row_.count(1);
          for(int output_y = 0; output_y < height; ++output_y) {
              for (int output_x = 0; output_x < width; ++output_x) {
                  int channel_ind = output_y * width + output_x;
                  int M = top[0]->shape(3); // rows of A. A: roi*roi * 1
                  int N = kernel_row_.shape(3); // cols of B^T. B: channels*1
                  int K = top[0]->shape(2); // cols of A
                  // C=alpha*A*B+beta*C
                  caffe_gpu_gemm(CblasNoTrans,CblasTrans,
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
      // For row2im, to avoid involving atomic operations, we will launch one kernel per
      // bottom dimension, and then in the kernel add up the top dimensions.
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
      caffe_gpu_set(bottom[0]->count(0), Dtype(0.0), bottom_diff);
      const int n_step = width*height*channels;
      for (int n = 0; n < n_images/2; ++n) {
          Dtype* im_diff = bottom_diff + n_step * 2 * n;
          Dtype* kernel_diff = bottom_diff + n_step * ( 2 * n + 1);
          const int num_kernels = channels * height * width;
          row2im_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                  CAFFE_CUDA_NUM_THREADS>>>(num_kernels, im_diff, channels, height, width, roi_,
                   //Input
                  im_row_diff + n * im_row_.count(1)
                  );
          // row2kernel
          // We launch width*height kernel.
          const int num_kernels2 = width * height;
          row2kernel_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels2),
                  CAFFE_CUDA_NUM_THREADS>>>(num_kernels2, kernel_diff, channels, height, width, roi_,
                          // Input
                          kernel_row_diff + n * kernel_row_.count(1)
                  );
      }



      CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(PixelwiseSimilarityLayer);


}  // namespace caffe
