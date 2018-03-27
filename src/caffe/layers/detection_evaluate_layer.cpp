#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "caffe/layers/detection_evaluate_layer.hpp"
#include "caffe/util/bbox_util.hpp"

namespace caffe {

template <typename Dtype>
void DetectionEvaluateLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const DetectionEvaluateParameter& detection_evaluate_param =
      this->layer_param_.detection_evaluate_param();
  CHECK(detection_evaluate_param.has_num_classes())
      << "Must provide num_classes.";
  num_classes_ = detection_evaluate_param.num_classes();
  overlap_threshold_ = detection_evaluate_param.overlap_threshold();
  CHECK_GT(overlap_threshold_, 0.) << "overlap_threshold must be non negative.";
  evaluate_difficult_gt_ = detection_evaluate_param.evaluate_difficult_gt();
  CHECK(detection_evaluate_param.has_name_size_file())
      <<"size files is needed for now!";
  string name_size_file = detection_evaluate_param.name_size_file();
  std::ifstream infile(name_size_file.c_str());
  CHECK(infile.good())
      << "Failed to open name size file: " << name_size_file;
  // The file is in the following format:
  //    name height width
  //    ...
  string name;
  int height, width;
  while (infile >> name >> height >> width) {
    sizes_.push_back(std::make_pair(height, width));
  }
  infile.close();
  LOG(INFO) << "size file has "<<sizes_.size()<<" items.";
  count_ = 0;
}

template <typename Dtype>
void DetectionEvaluateLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_LE(count_, sizes_.size());
  CHECK_EQ(bottom[0]->channels(), 7);
  CHECK_EQ(bottom[1]->channels(), 8);
  CHECK_EQ(bottom[2]->channels(), 2);

  int num_pos_classes = num_classes_ - 1;
  int num_valid_det = 0;
  const Dtype* det_data = bottom[0]->cpu_data();
  for (int i = 0; i < bottom[0]->num(); ++i) {
    // invalid detections with label == -1
    if (det_data[5] != -1) {
      ++num_valid_det;
    }
    det_data += 7;
  }
  // Each row is a 5 dimension vector, which stores
  // [image_id, label, confidence, true_pos, false_pos]
  top[0]->Reshape(num_pos_classes+num_valid_det,5,1,1);
}

template <typename Dtype>
void DetectionEvaluateLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* det_data = bottom[0]->cpu_data();
  const Dtype* gt_data = bottom[1]->cpu_data();
  // input size (height, width)
  const Dtype * size_data = bottom[2]->cpu_data();

  // Retrieve all detection results.
  map<int, LabelBBox> all_detections;
  GetDetectionResults(det_data, bottom[0]->num(), bottom[0]->channels(), 
          &all_detections);

  // Retrieve all ground truth (including difficult ones).
  map<int, LabelBBox> all_gt_bboxes;
  GetGroundTruth(gt_data, bottom[1]->num(), bottom[1]->channels(), true, 
          &all_gt_bboxes);

  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(0.), top_data);
  int num_det = 0;

  // Insert number of ground truth for each label.
  map<int, int> num_pos;
  for (map<int, LabelBBox>::iterator it = all_gt_bboxes.begin();
       it != all_gt_bboxes.end(); ++it) {
    for (LabelBBox::iterator iit = it->second.begin(); iit != it->second.end();
         ++iit) {
      int count = 0;
      if (evaluate_difficult_gt_) {
        count = iit->second.size();
      } else {
        // Get number of non difficult ground truth.
        for (int i = 0; i < iit->second.size(); ++i) {
          if (!iit->second[i].difficult) {
            ++count;
          }
        }
      }
      if (num_pos.find(iit->first) == num_pos.end()) {
        num_pos[iit->first] = count;
      } else {
        num_pos[iit->first] += count;
      }
    }
  }
  for (int c = 1; c < num_classes_; ++c) {
    top_data[num_det * 5] = -1;
    top_data[num_det * 5 + 1] = c;
    if (num_pos.find(c) == num_pos.end()) {
      top_data[num_det * 5 + 2] = 0;
    } else {
      top_data[num_det * 5 + 2] = num_pos.find(c)->second;
    }
    top_data[num_det * 5 + 3] = -1;
    top_data[num_det * 5 + 4] = -1;
    ++num_det;
  }

  // Insert detection evaluate status.
  for (map<int, LabelBBox>::iterator it = all_detections.begin(); 
        it != all_detections.end(); ++it) {
    int image_id = it->first;
    LabelBBox& detections = it->second;
    if (all_gt_bboxes.find(image_id) == all_gt_bboxes.end()) {
      // No ground truth for current image. All detections become false_pos.
      for (LabelBBox::iterator iit = detections.begin();
           iit != detections.end(); ++iit) {
        int label = iit->first;
        if (label == -1) {
          continue; // ignore fake detections
        }
        const vector<BBox>& bboxes = iit->second;
        for (int i = 0; i < bboxes.size(); ++i) {
          top_data[num_det * 5] = image_id;
          top_data[num_det * 5 + 1] = label;
          top_data[num_det * 5 + 2] = bboxes[i].score;
          top_data[num_det * 5 + 3] = 0;
          top_data[num_det * 5 + 4] = 1;
          ++num_det;
        }
      }
    } else {
      LabelBBox& label_bboxes = all_gt_bboxes.find(image_id)->second;
      for (LabelBBox::iterator iit = detections.begin();
           iit != detections.end(); ++iit) {
        int label = iit->first;
        if (label == -1) {
          continue; // ignore fake detections
        }
        vector<BBox>& bboxes = iit->second;
        if (label_bboxes.find(label) == label_bboxes.end()) {
          // No ground truth for current label. All detections become false_pos.
          for (int i = 0; i < bboxes.size(); ++i) {
            top_data[num_det * 5] = image_id;
            top_data[num_det * 5 + 1] = label;
            top_data[num_det * 5 + 2] = bboxes[i].score;
            top_data[num_det * 5 + 3] = 0;
            top_data[num_det * 5 + 4] = 1;
            ++num_det;
          }
        } else {
          CHECK_GT(bottom[2]->num(),image_id);
          const float h_ratio = sizes_[count_].first / size_data[2*image_id];
          const float w_ratio = sizes_[count_].second / size_data[2*image_id+1];
          vector<BBox>& gt_bboxes = label_bboxes.find(label)->second;
          // scale ground truth and detections to actual values
          AffineBBoxes(gt_bboxes,w_ratio,h_ratio,0,0);
          AffineBBoxes(bboxes,w_ratio,h_ratio,0,0);
          
          vector<bool> visited(gt_bboxes.size(), false);
          // Sort detections in descend order based on scores.
          std::sort(bboxes.begin(), bboxes.end(), SortBBoxDescend);
          for (int i = 0; i < bboxes.size(); ++i) {
            top_data[num_det * 5] = image_id;
            top_data[num_det * 5 + 1] = label;
            top_data[num_det * 5 + 2] = bboxes[i].score;
            // Compare with each ground truth bbox.
            float overlap_max = -1;
            int jmax = -1;
            for (int j = 0; j < gt_bboxes.size(); ++j) {
              float overlap = JaccardOverlap(bboxes[i], gt_bboxes[j],"IOU");
              if (overlap > overlap_max) {
                overlap_max = overlap;
                jmax = j;
              }
            }
            if (overlap_max >= overlap_threshold_) {
              if (evaluate_difficult_gt_ ||
                  (!evaluate_difficult_gt_ && !gt_bboxes[jmax].difficult)) {
                if (!visited[jmax]) {
                  // true positive.
                  top_data[num_det * 5 + 3] = 1;
                  top_data[num_det * 5 + 4] = 0;
                  visited[jmax] = true;
                } else {
                  // false positive (multiple detection).
                  top_data[num_det * 5 + 3] = 0;
                  top_data[num_det * 5 + 4] = 1;
                }
              }
            } else {
              // false positive.
              top_data[num_det * 5 + 3] = 0;
              top_data[num_det * 5 + 4] = 1;
            }
            ++num_det;
          }
        }
      }
    }
    if (sizes_.size() > 0) {
      ++count_;
      if (count_ == sizes_.size()) {
        // reset count after a full iterations through the DB.
        count_ = 0;
      }
    }
  }
  CHECK_EQ(num_det,top[0]->num());
}

INSTANTIATE_CLASS(DetectionEvaluateLayer);
REGISTER_LAYER_CLASS(DetectionEvaluate);

}  // namespace caffe
