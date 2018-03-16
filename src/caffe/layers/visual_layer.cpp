#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <vector>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <string>


#include "caffe/layer.hpp"
#include "caffe/layers/visual_layer.hpp"
#include "caffe/net.hpp"
#include <sys/stat.h>
#include <sys/types.h>


namespace caffe {

template <typename Dtype>
void VisualLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
   visual_interval_ = this->layer_param_.visual_param().visual_interval();
   save_dir_ = this->layer_param_.visual_param().save_dir();
    im_width_ = this->layer_param_.visual_param().width();
    iter_ = 0;
   CHECK_EQ(bottom.size(), 1) << "Wrong number of bottom blobs.";
   mkdir(save_dir_.c_str(), S_IRWXU);
}

template <typename Dtype>
void VisualLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    top[0]->Reshape(1,1,1,1);
}


template <typename Dtype>
void VisualLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    if(iter_ % visual_interval_ == 0){
        const Dtype* bottom_data = bottom[0]->cpu_data();
        int channels = bottom[0]->shape(1);
        int height = bottom[0]->shape(2);
        int width = bottom[0]->shape(3);
        im_height_ = int(ceil(channels*1.0/im_width_));
        int im_rows = im_height_ * height;
        int im_cols = im_width_ * width;
        cv::Mat cv_img = cv::Mat(im_rows, im_cols, CV_8UC1, cv::Scalar(0));
        for (int c = 0; c < channels; ++c) {
            const Dtype* data = bottom_data + c * height * width;
            Dtype max_value = -1000;
            Dtype min_value = 1000;
            // Find max value
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    if( data[y * width + x] > max_value){
                        max_value = data[y * width + x];
                    }
                    if( data[y * width + x] < min_value){
                        min_value = data[y * width + x];
                    }
                }
            }

            int ind_x = c % im_width_;
            int ind_y = c / im_width_;
            // Normalize to 0-255
            uchar* im_data = (uchar*)cv_img.data + ind_y * im_cols * height +ind_x * width;
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    im_data[y * im_cols + x] = (uchar(data[y * width + x] - min_value) / (max_value - min_value) * 255);
                }
            }
        }
        stringstream ss;
        ss<<iter_;
        std::string filename =  save_dir_ + "/" +ss.str() + ".jpg";
        if(cv::imwrite(filename, cv_img)){
            std::cout<<"Write " <<filename<<std::endl;
        }
    }
    iter_++;
}

template <typename Dtype>
void VisualLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(VisualLayer);
#endif

INSTANTIATE_CLASS(VisualLayer);
REGISTER_LAYER_CLASS(Visual);

}  // namespace caffe
