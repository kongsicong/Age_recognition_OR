Ordinal Regression
==================

Ordinal Regression with Multiple Output CNN for Age Estimation

## References

[Ordinal Regression with Multiple Output CNN for Age Estimation](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Niu_Ordinal_Regression_With_CVPR_2016_paper.pdf)

添加mae_layer, ordianl_regression_loss_layer的步骤：
1，将mae_layer.hpp, ordinal_regression_loss_layer.hpp 放置在caffe/include/caffe/layers/下
2，将mae_layer.cpp, ordinal_regression_loss_layer.cpp, ordinal_regressioin_loss_layer.cu 放置在caffe/src/caffe/layers/下
3，修改caffe/src/caffe/proto/caffe.proto，具体添加代码见layers/caffe.proto中的说明
4，将test_ordinal_regression_loss_layer.cpp 放置在caffe/src/caffe/test/下（可以不用放置，该文件只是测试自定义层是否正确的代码）
