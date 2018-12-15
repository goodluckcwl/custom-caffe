#include <vector>

#include "caffe/layers/mapping_by_similarity_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void mapping_forward_gpu_kernel(const int n_kernels, const Dtype* data_im1, const Dtype* data_im2,
                                        const int channels, const int height, const int width,
                                        const Dtype* data_sim, Dtype alpha,
                                           Dtype* output_im1, Dtype* output_im2){
    CUDA_KERNEL_LOOP(index, n_kernels) {
        const int w = index % width;
        const int h = index / width;

        const Dtype* data_im1_ptr = data_im1;
        data_im1_ptr += h * width + w;
        const Dtype* data_im2_ptr = data_im2;
        data_im2_ptr += h * width + w;
        Dtype* output_im1_ptr = output_im1;
        output_im1_ptr += h * width + w;
        Dtype* output_im2_ptr = output_im2;
        output_im2_ptr += h * width + w;

        Dtype beta = *(data_sim + h * width + w);
        const int channel_size = width * height;
        for (int c = 0; c < channels; ++c) {
            Dtype x1 = *(data_im1_ptr + c * channel_size);
            Dtype x2 = *(data_im2_ptr + c * channel_size);
            output_im1_ptr[c * channel_size] = (x1 + alpha*(1-beta) * x2)/(1 + alpha - alpha*beta);
            output_im2_ptr[c * channel_size] = (x2 + alpha*(1-beta) * x1)/(1 + alpha - alpha*beta);

        }
    }
}

template <typename Dtype>
__global__ void image_backward_gpu_kernel(const int n_kernels, const Dtype* top_diff1, const Dtype* top_diff2,
                                        const int channels, const int height, const int width,
                                        const Dtype* data_sim, Dtype alpha,
                                          Dtype* bottom_diff1, Dtype* bottom_diff2){
    CUDA_KERNEL_LOOP(index, n_kernels) {
        const int w = index % width;
        const int h = index / width;

        const Dtype* top_diff1_ptr = top_diff1;
        top_diff1_ptr += h * width + w;
        const Dtype* top_diff2_ptr = top_diff2;
        top_diff2_ptr += h * width + w;

        Dtype* bottom_diff1_ptr = bottom_diff1;
        bottom_diff1_ptr += h * width + w;
        Dtype* bottom_diff2_ptr = bottom_diff2;
        bottom_diff2_ptr += h * width + w;

        const int channel_size = width * height;
        Dtype beta = *(data_sim + h * width + w);
        for (int c = 0; c < channels; ++c) {
            Dtype d1_diff = top_diff1_ptr[c * channel_size];
            Dtype d2_diff = top_diff2_ptr[c * channel_size];
            Dtype factor = 1+alpha-alpha*beta;
            bottom_diff1_ptr[c * channel_size] += 1/factor * d1_diff + alpha*(1-beta)/factor*d2_diff;
            bottom_diff2_ptr[c * channel_size] += 1/factor * d2_diff + alpha*(1-beta)/factor*d1_diff;
        }
    }
}

template <typename Dtype>
__global__ void similarity_backward_gpu_kernel(const int n_kernels, const Dtype* bottom_data1, const Dtype* bottom_data2,
                                            const int channels, const int height, const int width,
                                            const Dtype* top_diff1, const Dtype* top_diff2,
                                            const Dtype* data_sim,
                                               Dtype alpha,
                                               Dtype* diff_sim){
    CUDA_KERNEL_LOOP(index, n_kernels) {
        const int w = index % width;
        const int h = index / width;

        const Dtype* bottom_data1_ptr = bottom_data1;
        bottom_data1_ptr += h * width + w;
        const Dtype* bottom_data2_ptr = bottom_data2;
        bottom_data2_ptr += h * width + w;

        const Dtype* top_diff1_ptr = top_diff1;
        top_diff1_ptr += h * width + w;
        const Dtype* top_diff2_ptr = top_diff2;
        top_diff2_ptr += h * width + w;

        Dtype* diff_sim_ptr = diff_sim;
        diff_sim_ptr += h * width + w;

        const int channel_size = width * height;
        Dtype beta = data_sim[h * width + w];
        for (int c = 0; c < channels; ++c) {
            Dtype x1 = bottom_data1_ptr[c * channel_size];
            Dtype x2 = bottom_data2_ptr[c * channel_size];
            Dtype factor = 1 + alpha - alpha * beta;
            Dtype factor1 =(alpha*x1 - alpha*x2)/(factor*factor);
            Dtype factor2 =(alpha*x2 - alpha*x1)/(factor*factor);
            // Accumulate diffs
            Dtype d1_diff = top_diff1_ptr[c * channel_size];
            Dtype d2_diff = top_diff2_ptr[c * channel_size];
            *diff_sim_ptr += factor1 * d1_diff + factor2 * d2_diff;
        }
    }
}

template <typename Dtype>
void MappingBySimilarityLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    if(false){
        this->Forward_cpu(bottom, top);
    }
    const Dtype *bottom_data = bottom[0]->gpu_data();
    const Dtype *sim_data = bottom[1]->gpu_data();
    Dtype *top_data = top[0]->mutable_gpu_data();
    int n_images = bottom[0]->shape(0);
    int channels = bottom[0]->shape(1);
    int height = bottom[0]->shape(2);
    int width = bottom[0]->shape(3);

    int channel_size = bottom[0]->count(2);
    // We lanch width * height kernels.
    const int num_kernels = width * height;
    for (int n = 0; n < n_images/2; ++n) {
        const Dtype *s1 = bottom_data + 2*n * bottom[0]->count(1);
        const Dtype *s2 = bottom_data + (2*n+1) * bottom[0]->count(1);
        Dtype *d1 = top_data + 2 * n * top[0]->count(1);
        Dtype *d2 = top_data + (2*n+1) * top[0]->count(1);
        mapping_forward_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                CAFFE_CUDA_NUM_THREADS>>>(
                        num_kernels, s1, s2, channels, height, width,
                        sim_data + n * bottom[1]->count(1),
                        //alpha
                        alpha_,
                        //Output
                        d1,d2);
    }

}

template <typename Dtype>
void MappingBySimilarityLayer<Dtype>::Backward_gpu(const std::vector<caffe::Blob<Dtype> *> &top,
                                                   const std::vector<bool> &propagate_down,
                                                   const std::vector<caffe::Blob<Dtype> *> &bottom) {
    if (propagate_down[0]) {
        if(false){
            this->Backward_cpu(top, propagate_down, bottom);
        }
        // gradient w.r.t. image. Note that we will accumulate diffs.
        Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
        const Dtype *top_diff = top[0]->gpu_diff();
        const Dtype *sim_data = bottom[1]->gpu_data();
        int n_images = bottom[0]->shape(0);
        int channels = bottom[0]->shape(1);
        int height = bottom[0]->shape(2);
        int width = bottom[0]->shape(3);

        // Clear grad
        caffe_gpu_set(bottom[0]->count(0), Dtype(0.0), bottom_diff);
        int channel_size = bottom[0]->count(2);
        // We launch width*height kernels.
        const int num_kernels = width * height;
        for (int n = 0; n < n_images/2; ++n) {
            Dtype *diff1 = bottom_diff + 2*n * bottom[0]->count(1) ;
            Dtype *diff2 = bottom_diff + (2*n+1) * bottom[0]->count(1);
            const Dtype *d1 = top_diff + 2*n * top[0]->count(1) ;
            const Dtype *d2 = top_diff + (2*n+1) * top[0]->count(1);
            image_backward_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                    CAFFE_CUDA_NUM_THREADS>>>(
                            num_kernels, d1, d2, channels, height, width,
                                    sim_data + n * bottom[1]->count(1),
                                    alpha_,
                                    //Output
                                    diff1,diff2);
        }
    }
    if (propagate_down[1]) {
        Dtype* sim_diff = bottom[1]->mutable_gpu_diff();
        const Dtype *sim_data = bottom[1]->gpu_data();

        const Dtype* bottom_data = bottom[0]->gpu_data();
        const Dtype *top_diff = top[0]->gpu_diff();

        int n_images = bottom[0]->shape(0);
        int channels = bottom[0]->shape(1);
        int height = bottom[0]->shape(2);
        int width = bottom[0]->shape(3);
        int channel_size = bottom[0]->count(2);

        // Clear grads. Because we need to accumulate diffs.
        // Note that the similarity map has only one channel.
        caffe_gpu_set(bottom[1]->count(0), Dtype(0.0), sim_diff);
        // We launch width*height kernels.
        const int num_kernels = width * height;
        for (int n = 0; n < n_images/2; ++n) {
            const Dtype *s1 = bottom_data + 2*n * bottom[0]->count(1) ;
            const Dtype *s2 = bottom_data + (2*n+1) * bottom[0]->count(1) ;
            Dtype *diff = sim_diff + n * bottom[1]->count(1) ;
            const Dtype *d1 = top_diff + 2*n * top[0]->count(1) ;
            const Dtype *d2 = top_diff + (2*n+1) * top[0]->count(1);
            similarity_backward_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                    CAFFE_CUDA_NUM_THREADS>>>(
                            num_kernels, s1, s2, channels, height, width,
                                    d1, d2,
                                    sim_data + n * bottom[1]->count(1),
                                    alpha_,
                                    //Output
                                    diff);
        }

    }
}


INSTANTIATE_LAYER_GPU_FUNCS(MappingBySimilarityLayer);

}  // namespace caffe
