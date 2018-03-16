#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/image_float_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
ImageFloatDataLayer<Dtype>::~ImageFloatDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void ImageFloatDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  string root_folder = this->layer_param_.image_data_param().root_folder();
  balance_ = this->layer_param_.image_data_param().balance_class();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string line;
  size_t pos;
  std::vector<Dtype> label;
  lines_.clear();
  hard_sample_ = this->layer_param_.image_float_data_param().hard_sample();
  label_dim_ = this->layer_param_.image_float_data_param().label_dim();
  with_heatmap_ = this->layer_param_.image_float_data_param().with_heatmap();
  heatmap_dir_ = this->layer_param_.image_float_data_param().heatmap_dir();

  while (std::getline(infile, line)) {
    label.clear();
    pos = line.find_first_of(' ');
    string im_name = line.substr(0, pos);
    line = line.substr(pos);
    pos = line.find_first_of(' ');
    while(pos != string::npos){


        label.push_back(atof(line.substr(pos + 1).c_str()));
        line = line.substr(pos + 1);
        pos = line.find_first_of(' ');
    }
    if(hard_sample_){
        CHECK_EQ(label_dim_ + 1, label.size());
    }else{
        CHECK_EQ(label_dim_, label.size());
    }



    lines_.push_back(std::make_pair(im_name, label));
  }

  CHECK(!lines_.empty()) << "File is empty";

  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  } else {
    if (this->phase_ == TRAIN && Caffe::solver_rank() > 0 &&
        this->layer_param_.image_data_param().rand_skip() == 0) {
      LOG(WARNING) << "Shuffling or skipping recommended for multi-GPU";
    }
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    new_height, new_width, is_color);
  // Read Heatmap
  if(with_heatmap_){
    std::string im_name = lines_[lines_id_].first;
    //pos = name.find_first_of('/');
    //std::string im_name = name.substr(0, pos) + '_' + name.substr(pos+1);
    cv::Mat cv_heatmap = ReadImageToCVMat(heatmap_dir_ + im_name,
                                          new_height, new_width, false);
    if(!cv_heatmap.data){
          cv_heatmap = cv::Mat(cv_img.rows, cv_img.cols, CV_8UC1);
    }
    //cv::Mat cv_tmp = cv::Mat(cv_img.rows, cv_img.cols, CV_32SC(4));
    std::vector<cv::Mat> channels;
    cv::split(cv_img, channels);
    channels.push_back(cv_heatmap);
    cv::merge(channels, cv_img);
  }

  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  vector<int> label_shape;
  label_shape.push_back( batch_size);
  label_shape.push_back(label_dim_);
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->label_.Reshape(label_shape);
  }
    // Weight
    vector<int> weight_shape(1, batch_size);
    if (top.size() == 3 ) {
        top[2]->Reshape(weight_shape);
        for (int i = 0; i < this->prefetch_.size(); ++i) {
            this->prefetch_[i]->weight_.Reshape(weight_shape);
        }
        this->output_weights_ = true;
    }
}

template <typename Dtype>
void ImageFloatDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void ImageFloatDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const bool is_color = image_data_param.is_color();
  string root_folder = image_data_param.root_folder();


  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
      new_height, new_width, is_color);
  // Read Heatmap
  if(with_heatmap_){
    std::string im_name = lines_[lines_id_].first;
    //int pos = name.find_first_of('/');
    //std::string im_name = name.substr(0, pos) + '_' + name.substr(pos+1);
    cv::Mat cv_heatmap = ReadImageToCVMat(heatmap_dir_ + im_name,
                                          new_height, new_width, false);
    if(!cv_heatmap.data){
        cv_heatmap = cv::Mat(cv_img.rows, cv_img.cols, CV_8UC1);
    }
    //cv::Mat cv_tmp = cv::Mat(cv_img.rows, cv_img.cols, CV_32SC(4));
    std::vector<cv::Mat> channels;
    cv::split(cv_img, channels);
    channels.push_back(cv_heatmap);
    cv::merge(channels, cv_img);
  }
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();
  Dtype* prefetch_weight ;
  if (this->output_weights_) {
    prefetch_weight = batch->weight_.mutable_cpu_data();
  }

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    bool valid_sample = false;
    while (!valid_sample) {
      std::pair<std::string, std::vector<Dtype> > this_line;

      if (balance_) {

      }
      else {
        CHECK_GT(lines_size, lines_id_);
        this_line = lines_[lines_id_];
      }

      cv::Mat cv_img = ReadImageToCVMat(root_folder + this_line.first,
                                        new_height, new_width, is_color);
      // Read Heatmap
      cv::Mat cv_heatmap;
      if(with_heatmap_){
        std::string im_name = lines_[lines_id_].first;
        //int pos = name.find_first_of('/');
        //std::string im_name = name.substr(0, pos) + '_' + name.substr(pos+1);

        cv_heatmap = ReadImageToCVMat(heatmap_dir_  + im_name,
                                              new_height, new_width, false);
        if(cv_heatmap.data){
            //cv::Mat cv_tmp = cv::Mat(cv_img.rows, cv_img.cols, CV_32SC(4));
            std::vector<cv::Mat> channels;
            cv::split(cv_img, channels);
            channels.push_back(cv_heatmap);
            cv::merge(channels, cv_img);
        }

      }
      if (!cv_img.data || (!cv_heatmap.data && with_heatmap_)) {
        LOG(INFO) << "Could not load " << this_line.first;
        valid_sample = false;
      }
      else {
        valid_sample = true;
      }
      read_time += timer.MicroSeconds();
      timer.Start();
      // Apply transformations (mirror, crop...) to the image
      int offset = batch->data_.offset(item_id);
      this->transformed_data_.set_cpu_data(prefetch_data + offset);
      if(cv_img.channels() == this->transformed_data_.channels()){
          // Load label
          vector<Dtype> label_tmp;
          if(hard_sample_){
              std::copy(this_line.second.begin(), this_line.second.end() - 1, std::back_inserter(label_tmp)); // Use one dim for weight
              // Set Weight
              prefetch_weight[item_id] = *(this_line.second.end() - 1);
          }else{
              std::copy(this_line.second.begin(), this_line.second.end(), std::back_inserter(label_tmp));
          }
          CHECK_EQ(label_tmp.size(), label_dim_);
          //caffe_copy(label_dim, label_tmp.data(), prefetch_label+item_id * label_dim);
        this->data_transformer_->Transform(cv_img, &(this->transformed_data_), label_tmp, prefetch_label+item_id * label_dim_);
        trans_time += timer.MicroSeconds();

      }else{
        // since we have not fetched an image sucessfully.
        valid_sample = false;
        // Try another image

      }



      // go to the next iter
      if (balance_) {
        class_id_++;
        if (class_id_ >= num_samples_.size()) {
          // We have reached the end. Restart from the first.
          DLOG(INFO) << "Restarting data prefetching from start.";
          class_id_ = 0;
        }
      }
      else {
        lines_id_++;
        if (lines_id_ >= lines_size) {
          // We have reached the end. Restart from the first.
          DLOG(INFO) << "Restarting data prefetching from start.";
          lines_id_ = 0;
          if (this->layer_param_.image_data_param().shuffle()) {
            ShuffleImages();
          }
        }
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ImageFloatDataLayer);
REGISTER_LAYER_CLASS(ImageFloatData);

}  // namespace caffe
#endif  // USE_OPENCV
