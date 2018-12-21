#include <cassert>
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

static std::vector<float> softmax(const float* xs, int num) {
  std::vector<float> r(num);

  float max_val = 0.0;
  for (int i = 0; i < num; i++) {
    max_val = std::max(xs[i], max_val);
  }

  float exp_sum = 0.0;
  for (int i = 0; i < num; i++) {
    exp_sum = exp(xs[i] - max_val);
  }

  for (int i = 0; i < num; i++) {
    r[i] = exp(xs[i] - max_val) / exp_sum;
  }

  return std::move(r);
}

static float sigmoid(float x) {
  if (x > 0) {
    return 1.0 / (1.0 + exp(-x));
  } else {
    return exp(x) / (1.0 + exp(x));
  }
}

static float CalcIoU(const box_util::Box& a, const box_util::Box& b) {
  float left = std::max(a.x, b.x);
  float top = std::max(a.y, b.y);

  float right = std::min(a.x + a.w, b.x + b.w);
  float bottom = std::min(a.y + a.h, b.y + b.h);

  float inner_area = (right - left) * (top - bottom);
  float a_area = a.w * a.h;
  float b_area = b.w * b.h;

  float epsilon = 1e-10;

  float r = inner_area / (a_area + b_area - inner_area + epsilon);

  if (std::isnan(r)) {
    return 0.0;
  }
  if (r > 1.0) {
    return 1.0;
  } else if (r < 0.0) {
    return 0.0;
  }
  return r;
}

static box_util::Box ConvertBboxCoordinate(float x, float y, float w, float h, float k,
                                           const std::pair<float, float>& anchor,
                                           int nth_y, int nth_x,
                                           int num_cell_y, int num_cell_x) {
  box_util::Box r;
  float anchor_w = anchor.first;
  float anchor_h = anchor.second;

  float cy = y + static_cast<float>(nth_y) / num_cell_y;
  float cx = x + static_cast<float>(nth_x) / num_cell_x;

  r.h = exp(h) * static_cast<float>(anchor_h) / num_cell_y;
  r.w = exp(w) * static_cast<float>(anchor_w) / num_cell_x;

  r.y = cy - (r.h / 2);
  r.x = cx - (r.w / 2);

  return r;
}

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


// convert yolov2's detection result to more easy format
// output coordinates are not translated into original image coordinates,
// since we can't know original image size.
Tensor FormatYoloV2(const Tensor& input,
                    const std::vector<std::pair<float, float>>& anchors,
                    const int& boxes_per_cell,
                    const std::string& data_format,
                    const std::pair<int, int>& image_size,
                    const int& num_classes) {
  //input shape must be NHWC, N == 1

  auto shape = input.shape();
  int num_cell_y = shape[1];
  int num_cell_x = shape[2];

  assert(shape[0] == 1);
  assert(shape.size() == 4);
  assert(input.size() % (num_cell_y * num_cell_x * anchors.size()) == 0);
  assert(anchors.size() == boxes_per_cell);

  std::vector<int> output_shape = {1, num_cell_y * num_cell_x * boxes_per_cell * num_classes, 6};
  Tensor result(output_shape);

  int r_i = 0;
  for (int i = 0; i < num_cell_y; i++) {
    for (int j = 0; j < num_cell_x; j++) {
      const float* predictions = input.dataAsArray({0, i, j, 0});
      for (size_t k = 0; k < anchors.size(); k++) {
        // is it ok to use softmax when num_classes == 1?
        std::vector<float> probs = softmax(predictions, num_classes);
        float conf = sigmoid(predictions[num_classes]);
        float x = sigmoid(predictions[num_classes+1]);
        float y = sigmoid(predictions[num_classes+2]);
        float w = predictions[num_classes+3];
        float h = predictions[num_classes+4];

        box_util::Box bbox_im = ConvertBboxCoordinate(x, y, w, h, k, anchors[k], i, j, num_cell_y, num_cell_x);

        for (int c_i = 0; c_i < num_classes; c_i++) {
          float prob = probs[c_i];
          float score = prob * conf;
          auto p = result.dataAsArray({0, r_i, 0});
          p[0] = bbox_im.x;
          p[1] = bbox_im.y;
          p[2] = bbox_im.w;
          p[3] = bbox_im.h;
          p[4] = c_i;
          p[5] = score;
          r_i++;
        }

        predictions = predictions + (num_classes + 5);
      }
    }
  }

  return result;
}

Tensor FormatYoloV2(const Tensor& input, const FormatYoloV2Parameters& params) {
  return FormatYoloV2(input,
                      params.anchors,
                      params.boxes_per_cell,
                      params.data_format,
                      params.image_size,
                      params.num_classes);
}

Tensor ExcludeLowScoreBox(const Tensor& input, const float& threshold) {
  Tensor result(input);

  auto shape = input.shape();
  int num_predictions = shape[1];

  for (int i = 0; i < num_predictions; i++) {
    float* predictions = result.dataAsArray({0, i, 0});
    float score = predictions[5];
    if (score < threshold) {
      predictions[5] = -1.0;
    }
  }

  return result;
}

Tensor NMS(const Tensor& input,
           const std::vector<std::string>& classes,
           const float& iou_threshold,
           const int& max_output_size,
           const bool& per_class) {
  auto shape = input.shape();
  int num_predictions = shape[1];

  std::vector<int> ids;
  for (int i = 0; i < num_predictions; i++) {
    ids.push_back(i);
  }

  std::sort(ids.begin(), ids.end(),
            [input](const int& a, const int& b) -> bool
            {
              const float* prediction_a = input.dataAsArray({0, a, 0});
              float score_a = prediction_a[5];
              const float* prediction_b = input.dataAsArray({0, b, 0});
              float score_b = prediction_b[5];
              return score_a > score_b;
            });

  Tensor tmp(input);

  for (int i = 0; i < num_predictions; i++) {
    float* prediction_a = tmp.dataAsArray({0, ids[i], 0});
    float score = prediction_a[5];
    if (score < 0.0) {
      break;
    } else if (score == 0.0) {
      continue;
    }
    box_util::Box box_a = box_util::Box(prediction_a[0], prediction_a[1], prediction_a[2], prediction_a[3]);

    for (int j = i+1; j < num_predictions; j++) {
      float* prediction_b = tmp.dataAsArray({0, ids[j], 0});
      box_util::Box box_b = box_util::Box(prediction_b[0], prediction_b[1], prediction_b[2], prediction_b[3]);
      float iou = CalcIoU(box_a, box_b);

      if (iou > iou_threshold) {
        prediction_b[5] = 0.0;
      }
    }
  }

  Tensor result(input.shape());
  int j = 0;
  for (int i = 0; i < num_predictions; i++) {
    float* prediction = tmp.dataAsArray({0, ids[i], 0});
    float score = prediction[5];
    if (score > 0.0) {
      float* store_location = result.dataAsArray({0, j, 0});
      store_location[0] = prediction[0];
      store_location[1] = prediction[1];
      store_location[2] = prediction[2];
      store_location[3] = prediction[3];
      store_location[4] = prediction[4];
      store_location[5] = prediction[5];
      j++;
    }
  }
  return result;
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
