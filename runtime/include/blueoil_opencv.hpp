#ifndef RUNTIME_INCLUDE_BLUEOIL_OPENCV_HPP_
#define RUNTIME_INCLUDE_BLUEOIL_OPENCV_HPP_

#ifdef OPENCV_FOUND

#include <opencv2/opencv.hpp>

#include "blueoil.hpp"

namespace blueoil {
namespace opencv {

Tensor Tensor_fromCVMat(cv::Mat img);
cv::Mat Tensor_toCVMat(Tensor &tensor);

}  // namespace opencv
}  // namespace blueoil

#endif // OPENCV_FOUND

#endif  // RUNTIME_INCLUDE_BLUEOIL_OPENCV_HPP_
