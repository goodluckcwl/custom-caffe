#ifndef CAFFE_PARITY_SPLIT_LAYER_HPP_
#define CAFFE_PARITY_SPLIT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief 根据人脸识别的需要，把属于同一个pair的feature分成两组，同时生成label输出
 *
 * NOTE: does not implement Backwards operation.
 */
template <typename Dtype>
class ParitySplitLayer : public Layer<Dtype> {
public:
    /**
     * @param
     */
    explicit ParitySplitLayer(const LayerParameter &param)
            : Layer<Dtype>(param) {}

    virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                            const vector<Blob<Dtype> *> &top);

    virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                         const vector<Blob<Dtype> *> &top);

    virtual inline const char *type() const { return "ParitySplit"; }

    virtual inline int ExactNumBottomBlobs() const { return 2; }

    virtual inline int ExactNumTopBlobs() const { return 4; }

protected:
    /**
     * @param bottom input Blob vector (length 1)
     *   -# @f$ (N \times C \times H \times W) @f$
     *      the inputs @f$ x @f$
     * @param top output Blob vector (length 1)
     *   -# @f$ (N \times 1 \times K) @f$ or, if out_max_val
     *      @f$ (N \times 2 \times K) @f$ unless axis set than e.g.
     *      @f$ (N \times K \times H \times W) @f$ if axis == 1
     *      the computed outputs @f$
     *       y_n = \arg\max\limits_i x_{ni}
     *      @f$ (for @f$ K = 1 @f$).
     */
    virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                             const vector<Blob<Dtype> *> &top);

    /// @brief Not implemented (non-differentiable function)
    virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
                              const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom);
};
}  // namespace caffe

#endif  // CAFFE_PARITY_SPLIT_LAYER_HPP_
