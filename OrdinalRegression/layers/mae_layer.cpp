#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/mae_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <math.h>
namespace caffe {

template <typename Dtype>
void MAELayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  has_ignore_label_ = this->layer_param_.mae_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.mae_param().ignore_label();
  }
}

template <typename Dtype>
void MAELayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  label_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.accuracy_param().axis());
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_ + 1);
  vector<int> top_shape(0);  // MAE is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void MAELayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype abs_diff = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const int dim = bottom[0]->count() / outer_num_;
  const int num_labels = bottom[0]->shape(label_axis_);
  int count = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j) {
      const int label_value =
          static_cast<int>(bottom_label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, num_labels);
      // get the predict value
      int predict_val = 0;
      for (int k = 0; k < num_labels; k = k + 2) {
           if(bottom_data[i * dim + k * inner_num_ + j] < bottom_data[i * dim + (k + 1) * inner_num_ + j]) predict_val++;
      }
      
      abs_diff += abs(label_value - predict_val);
      ++count;
    }
  }

  // LOG(INFO) << "Accuracy: " << accuracy;
  top[0]->mutable_cpu_data()[0] = abs_diff / count;
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(MAELayer);
REGISTER_LAYER_CLASS(MAE);

}  // namespace caffe
