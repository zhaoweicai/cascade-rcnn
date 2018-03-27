// ------------------------------------------------------------------
// Cascade-RCNN
// Copyright (c) 2018 The Regents of the University of California
// see cascade-rcnn/LICENSE for details
// Written by Zhaowei Cai [zwcai-at-ucsd.edu]
// Please email me if you find bugs, or have suggestions or questions!
// ------------------------------------------------------------------

#ifndef CAFFE_UTIL_BBOX_UTIL_H_
#define CAFFE_UTIL_BBOX_UTIL_H_

#include <algorithm>
#include <string>
#include <utility>
#include <vector>
#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit
#include <map>

#include "glog/logging.h"

#include "caffe/common.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

namespace caffe {

struct BBox {
  BBox() {
    xmin = 0; ymin = 0; xmax = -1; ymax = -1;
    label = -1; difficult = false; score = 0; size = 0; ignore = false;
  }
  float xmin;
  float ymin;
  float xmax;
  float ymax;
  int label;
  bool difficult;
  float score;
  float size;
  bool ignore;
};

typedef map<int, vector<BBox> > LabelBBox;

bool SortBBoxAscend(const BBox& bbox1, const BBox& bbox2);

bool SortBBoxDescend(const BBox& bbox1, const BBox& bbox2);

template <typename T>
bool SortScorePairAscend(const pair<float, T>& pair1,
                         const pair<float, T>& pair2);

template <typename T>
bool SortScorePairDescend(const pair<float, T>& pair1,
                          const pair<float, T>& pair2);

BBox InitBBox(float xmin, float ymin, float xmax, float ymax);

float BBoxSize(const BBox bbox);

template <typename Dtype>
Dtype BBoxSize(const vector<Dtype> bbox);

float JaccardOverlap(const BBox bbox1, const BBox bbox2, const string iou_mode);

template <typename Dtype>
Dtype JaccardOverlap(const vector<Dtype> bbox1, const vector<Dtype> bbox2, 
        const string iou_mode);

template <typename Dtype>
Dtype JaccardOverlap(const Dtype x1, const Dtype y1, const Dtype w1, const Dtype h1,
          const Dtype x2, const Dtype y2, const Dtype w2, const Dtype h2, const string mode);

void ClipBBox(BBox& bbox, const float max_width, const float max_height);

template <typename Dtype>
void ClipBBox(vector<Dtype>& bbox, const Dtype max_width, const Dtype max_height);

void AffineBBox(BBox& bbox, const float w_scale, const float h_scale, 
        const float w_shift, const float h_shift);

void AffineBBoxes(vector<BBox>& bboxes, const float w_scale, const float h_scale, 
        const float w_shift, const float h_shift);

void FlipBBox(BBox& bbox, const float width);

void FlipBBoxes(vector<BBox>& bboxes, const float width);
        
vector<BBox> ApplyNMS(const vector<BBox>bboxes, const float iou_thr, 
        const bool greedy, const string iou_mode) ;

template <typename Dtype>
void EncodeBBoxes(const Dtype* bbox_data, const Dtype* label, const int num, 
        const int width, const int height, const Dtype* means, const Dtype* stds, 
        const float fw, const float fh, const float stride, Dtype* encode_data);

template <typename Dtype>
void EncodeGroupBBoxes(const Dtype* bbox_data, const Dtype* label, const int num, 
        const int group_num, const int width, const int height, const Dtype* means, 
        const Dtype* stds, const Dtype* fws, const Dtype* fhs, const float stride, 
        Dtype* encode_data);

template <typename Dtype>
void DecodeBBoxes(const Dtype* bbox_data, const int num, const int bottom_dim, 
        const int width, const int height, const Dtype* means, const Dtype* stds, 
        const Dtype* bounds, const float fw, const float fh, 
        const float stride, const int top_dim, Dtype* pred_data);

template <typename Dtype>
void DecodeGroupBBoxes(const Dtype* bbox_data, const int num, const int group_num, 
        const int bottom_dim, const int width, const int height, const Dtype* means, 
        const Dtype* stds, const Dtype* bounds, const Dtype* fws, const Dtype* fhs, 
        const float stride, const int top_dim, Dtype* pred_data);

template <typename Dtype>
void DecodeBBoxesWithPrior(const Dtype* bbox_data, const vector<BBox> prior_bboxes,  
        const int bbox_dim, const Dtype* means, const Dtype* stds, 
        Dtype* pred_data);

template <typename Dtype>
void BoundBBoxPreds(Dtype* bbox_data, const int num, const int bottom_dim,
        const int sp_dim, const Dtype* bounds);

template <typename Dtype>
void GetDetectionResults(const Dtype* det_data, const int num, const int dim,
      map<int, LabelBBox>* all_detections);

template <typename Dtype>
void GetGroundTruth(const Dtype* gt_data, const int num, const int dim,
      const bool use_difficult_gt, map<int, LabelBBox>* all_gt_bboxes);

void CumSum(const vector<pair<float, int> >& pairs, vector<int>* cumsum);

void ComputeAP(const vector<pair<float, int> >& tp, const int num_pos,
               const vector<pair<float, int> >& fp, const string ap_version,
               vector<float>* prec, vector<float>* rec, float* ap);

#ifndef CPU_ONLY  // GPU

template <typename Dtype>
void EncodeBBoxesGPU(const int nthreads, const Dtype* bbox_data, const Dtype* label,
        const int width, const int height, const Dtype* means, const Dtype* stds, 
        const float fw, const float fh, const float stride, Dtype* encode_data);

template <typename Dtype>
void EncodeGroupBBoxesGPU(const int nthreads, const Dtype* bbox_data, 
        const Dtype* label, const int group_num, const int width, const int height, 
        const Dtype* means, const Dtype* stds, const Dtype* fws, const Dtype* fhs, 
        const float stride, Dtype* encode_data);

template <typename Dtype>
void DecodeBBoxesGPU(const int nthreads, const Dtype* bbox_data,
        const int bottom_dim, const int width, const int height, const Dtype* means, 
        const Dtype* stds, const Dtype* bounds, const float fw, const float fh, 
        const float stride, const int top_dim, Dtype* pred_data);

template <typename Dtype>
void DecodeGroupBBoxesGPU(const int nthreads, const Dtype* bbox_data, 
        const int group_num, const int bottom_dim, const int width, const int height, 
        const Dtype* means, const Dtype* stds, const Dtype* bounds, const Dtype* fws, 
        const Dtype* fhs, const float stride, const int top_dim, Dtype* pred_data);

template <typename Dtype>
void BoundBBoxPredsGPU(const int nthreads, Dtype* bbox_data, 
        const int bottom_dim, const int sp_dim, const Dtype* bounds);

#endif  // !CPU_ONLY

}  // namespace caffe

#endif  // CAFFE_UTIL_BBOX_UTIL_H_
