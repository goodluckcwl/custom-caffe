#ifndef CAFFE_MAPPING_BY_SIMILARITY_LAYER_HPP_
#define CAFFE_MAPPING_BY_SIMILARITY_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

/**
 * @brief MappingBySimilarity @f$ y_1 = (x_1 + (1-\alpha)*x_2)/(2-\alpha),
 * y_2 = (x_2 + (1-\alpha)*x_1)/(2-\alpha)  @f$.
 *
 */
template <typename Dtype>
class MappingBySimilarityLayer : public Layer<Dtype> {
 public:

  explicit MappingBySimilarityLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "MappingBySimilarity"; }
    virtual inline int ExactNumBottomBlobs() const { return 2; }
    virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
   /**
    *
    * @param bottom bottom[0] is the input image blob of size (n, c, h, w), which means
    * it contains n/2 pairs of images exactly.
    * bottom[1] is the similarity map of size (n/2, 1, h, w).
    * @param top. The output image blob of size (n, c, h, w).
    */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);


  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
private:
    Dtype alpha_;
};

}  // namespace caffe

#endif  // CAFFE_MAPPING_BY_SIMILARITY_LAYER_HPP_
