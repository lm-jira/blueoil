#include <cmath>
#include <dlfcn.h>
#include <string>
#include <vector>
#include <iostream>
#include <numeric>
#include <algorithm>

#include "blueoil.hpp"
#include "blueoil_image.hpp"
#include "blueoil_data_processor.hpp"

namespace blueoil {
namespace data_processor {

// TODO(wakisaka): imple resize.
Tensor Resize(const Tensor& image, const std::pair<int, int>& size) {
  const int width = size.first;
  const int height = size.second;
  return blueoil::image::Resize(image, width, height,
				blueoil::image::RESIZE_FILTER_NEAREST_NEIGHBOR);
}

Tensor DivideBy255(const Tensor& image) {
  Tensor out(image);

  auto div255 = [](float i) { return i/255; };
  std::transform(image.begin(), image.end(), out.begin(), div255);

  return out;
}

Tensor PerImageStandardization(const Tensor& image) {
  Tensor out(image);

  double sum = 0.0;
  double sum2 = 0.0;

  for (auto it : image) {
    sum += it;
    sum2 += it * it;
  }

  float mean = sum / image.size();
  float var = sum2 / image.size() - mean * mean;
  double sd = std::sqrt(var);
  float adjusted_sd = std::max(sd, 1.0 / std::sqrt(image.size()));
  auto standardization = [mean, adjusted_sd](float i) { return (i - mean) / adjusted_sd; };
  std::transform(image.begin(), image.end(), out.begin(), standardization);
  return out;
}

// TODO(wakisaka): impl
Tensor FormatYoloV2(const Tensor& input,
                    const std::vector<std::pair<float, float>>& anchors,
                    const int& boxes_per_cell,
                    const std::string& data_format,
                    const std::pair<int, int>& image_size,
                    const int& num_classes) {
  Tensor t(input);

  return t;
}
// TODO(wakisaka): optimize
Tensor FormatYoloV2(const Tensor& input, const FormatYoloV2Parameters& params) {
  return FormatYoloV2(input,
                      params.anchors,
                      params.boxes_per_cell,
                      params.data_format,
                      params.image_size,
                      params.num_classes);
}

// TODO(wakisaka): impl
Tensor ExcludeLowScoreBox(const Tensor& input, const float& threshold) {
  Tensor t(input);

  return t;
}
// TODO(wakisaka): impl
Tensor NMS(const Tensor& input,
           const std::vector<std::string>& classes,
           const float& iou_threshold,
           const int& max_output_size,
           const bool& per_class) {
  Tensor t(input);

  return t;
}
// TODO(wakisaka): optimize
Tensor NMS(const Tensor& input, const NMSParameters& params){
  return NMS(input,
             params.classes,
             params.iou_threshold,
             params.max_output_size,
             params.per_class);
};
}  // namespace data_processor
}  // namespace blueoil
