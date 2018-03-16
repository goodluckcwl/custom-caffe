#ifndef CAFFE_IMAGE_FLOAT_DATA_LAYER_HPP_
#define CAFFE_IMAGE_FLOAT_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class ImageFloatDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit ImageFloatDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~ImageFloatDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ImageFloatData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline int MaxTopBlobs() const { return 3; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);

  vector<std::pair<std::string, std::vector<Dtype> > > lines_;

  int lines_id_;
  vector<int> num_samples_;
  vector<Dtype> class_weights_;
  int label_dim_;
  bool balance_;
  bool hard_sample_;
  bool with_heatmap_; // Indicate heatmap input
  std::string heatmap_dir_; //
  vector<vector<std::pair<std::string, int> > > filename_by_class_;
  int class_id_;
};


}  // namespace caffe

#endif  // CAFFE_IMAGE_FLOAT_DATA_LAYER_HPP_
