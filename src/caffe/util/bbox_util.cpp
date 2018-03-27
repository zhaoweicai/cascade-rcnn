// ------------------------------------------------------------------
// Cascade-RCNN
// Copyright (c) 2018 The Regents of the University of California
// see cascade-rcnn/LICENSE for details
// Written by Zhaowei Cai [zwcai-at-ucsd.edu]
// Please email me if you find bugs, or have suggestions or questions!
// ------------------------------------------------------------------

#include <algorithm>
#include <string>
#include <utility>
#include <vector>
#include <map>

#include "caffe/util/bbox_util.hpp"

namespace caffe {

bool SortBBoxAscend(const BBox& bbox1, const BBox& bbox2) {
  return bbox1.score < bbox2.score;
}

bool SortBBoxDescend(const BBox& bbox1, const BBox& bbox2) {
  return bbox1.score > bbox2.score;
}

template <typename T>
bool SortScorePairAscend(const pair<float, T>& pair1,
                         const pair<float, T>& pair2) {
  return pair1.first < pair2.first;
}

template bool SortScorePairAscend(const pair<float, int>& pair1,
                                  const pair<float, int>& pair2);
template bool SortScorePairAscend(const pair<float, pair<int, int> >& pair1,
                                  const pair<float, pair<int, int> >& pair2);

template <typename T>
bool SortScorePairDescend(const pair<float, T>& pair1,
                          const pair<float, T>& pair2) {
  return pair1.first > pair2.first;
}

template bool SortScorePairDescend(const pair<float, int>& pair1,
                                   const pair<float, int>& pair2);
template bool SortScorePairDescend(const pair<float, pair<int, int> >& pair1,
                                   const pair<float, pair<int, int> >& pair2);

BBox InitBBox(float xmin, float ymin, float xmax, float ymax) {
  BBox bbox;
  bbox.xmin = xmin; bbox.ymin = ymin;
  bbox.xmax = xmax; bbox.ymax = ymax;
  return bbox;
}

float BBoxSize(const BBox bbox) {
  if (bbox.xmax<bbox.xmin || bbox.ymax<bbox.ymin) {
    // If bbox is invalid (e.g. width <= 0 or height <= 0), return 0.
    return float(0.);
  } else {
    return (bbox.xmax-bbox.xmin+1) * (bbox.ymax-bbox.ymin+1);
  }
}

template <typename Dtype>
Dtype BBoxSize(const vector<Dtype> bbox) {
  // bbox = [xmin,ymin,xmax,ymax]
  if (bbox[2]<bbox[0] || bbox[3]<bbox[1]) {
    // If bbox is invalid (e.g. width <= 0 or height <= 0), return 0.
    return Dtype(0.);
  } else {
    return (bbox[2]-bbox[0]+1) * (bbox[3]-bbox[1]+1);
  }
}

template float BBoxSize(const vector<float> bbox);
template double BBoxSize(const vector<double> bbox);


float JaccardOverlap(const BBox bbox1, const BBox bbox2, const string iou_mode) {
  if (bbox1.xmax<bbox1.xmin || bbox1.ymax<bbox1.ymin 
          || bbox2.xmax<bbox2.xmin || bbox2.ymax<bbox2.ymin) {
    return float(0.);
  }
  const float inter_xmin = std::max(bbox1.xmin, bbox2.xmin);
  const float inter_ymin = std::max(bbox1.ymin, bbox2.ymin);
  const float inter_xmax = std::min(bbox1.xmax, bbox2.xmax);
  const float inter_ymax = std::min(bbox1.ymax, bbox2.ymax);
  float inter_size;
  if((inter_xmin>inter_xmax)||(inter_ymin>=inter_ymax)) {
    inter_size = float(0.);
  } else {
    inter_size = (inter_xmax-inter_xmin+1)*(inter_ymax-inter_ymin+1);
  }
  float union_size;
  float bbox1_size = BBoxSize(bbox1);
  float bbox2_size = BBoxSize(bbox2);
  
  if (iou_mode == "IOMU") {
    union_size = std::min(bbox1_size,bbox2_size);
  } else if (iou_mode == "IOFU") {
    union_size = bbox1_size;
  } else {
    union_size = bbox1_size+bbox2_size-inter_size;
  }
  
  if (union_size <= 0) {
    return float(0.);
  } else {
    return inter_size/union_size;
  }
} 

template <typename Dtype>
Dtype JaccardOverlap(const vector<Dtype> bbox1, const vector<Dtype> bbox2, 
        const string iou_mode) {
  // bbox = [xmin,ymin,xmax,ymax]
  if (bbox1[2]<bbox1[0] || bbox1[3]<bbox1[1] 
          || bbox2[2]<bbox2[0] || bbox2[3]<bbox2[1]) {
    return Dtype(0.);
  }
  const Dtype inter_xmin = std::max(bbox1[0], bbox2[0]);
  const Dtype inter_ymin = std::max(bbox1[1], bbox2[1]);
  const Dtype inter_xmax = std::min(bbox1[2], bbox2[2]);
  const Dtype inter_ymax = std::min(bbox1[3], bbox2[3]);
  Dtype inter_size;
  if((inter_xmin>inter_xmax)||(inter_ymin>=inter_ymax)) {
    inter_size = Dtype(0);
  } else {
    inter_size = (inter_xmax-inter_xmin+1)*(inter_ymax-inter_ymin+1);
  }
  Dtype union_size;
  Dtype bbox1_size = BBoxSize(bbox1);
  Dtype bbox2_size = BBoxSize(bbox2);
  
  if (iou_mode == "IOMU") {
    union_size = std::min(bbox1_size,bbox2_size);
  } else if (iou_mode == "IOFU") {
    union_size = bbox1_size;
  } else {
    union_size = bbox1_size+bbox2_size-inter_size;
  }
  
  if (union_size <= 0) {
    return Dtype(0.);
  } else {
    return inter_size/union_size;
  }
}  

template
float JaccardOverlap(const vector<float> bbox1, const vector<float> bbox2, 
        const string iou_mode);

template
double JaccardOverlap(const vector<double> bbox1, const vector<double> bbox2, 
        const string iou_mode);

template <typename Dtype>
Dtype JaccardOverlap(const Dtype x1, const Dtype y1, const Dtype w1, const Dtype h1,
          const Dtype x2, const Dtype y2, const Dtype w2, const Dtype h2, const string mode) {
  if (w1<=0 || h1<=0 || w2<=0 || h2<=0) {
    return Dtype(0);
  }
  Dtype tlx = std::max(x1, x2);
  Dtype tly = std::max(y1, y2);
  Dtype brx = std::min(x1+w1-1, x2+w2-1);
  Dtype bry = std::min(y1+h1-1, y2+h2-1);
  Dtype over;
  if((tlx>brx)||(tly>bry)) over = Dtype(0);
  else over = (brx-tlx+1)*(bry-tly+1);
  Dtype u;
  if (mode == "IOMU") {
    u = std::min(w1*h1,w2*h2);
  } else if (mode == "IOFU") {
    u = w1*h1;
  } else {
    u = w1*h1+w2*h2-over;
  }
  
  return over/u;
}  

template
float JaccardOverlap(const float x1, const float y1, const float w1, const float h1,
          const float x2, const float y2, const float w2, const float h2, const string mode);

template
double JaccardOverlap(const double x1, const double y1, const double w1, const double h1,
          const double x2, const double y2, const double w2, const double h2, const string mode);


void ClipBBox(BBox& bbox, const float max_width, const float max_height) {
  bbox.xmin = std::max(float(0.),bbox.xmin); 
  bbox.ymin = std::max(float(0.),bbox.ymin);
  bbox.xmax = std::min(max_width,bbox.xmax);
  bbox.ymax = std::min(max_height,bbox.ymax);
}

template <typename Dtype>
void ClipBBox(vector<Dtype>& bbox, const Dtype max_width, const Dtype max_height) {
  bbox[0] = std::max(Dtype(0.),bbox[0]); 
  bbox[1] = std::max(Dtype(0.),bbox[1]);
  bbox[2] = std::min(max_width,bbox[2]);
  bbox[3] = std::min(max_height,bbox[3]);
}

template
void ClipBBox(vector<float>& bbox, const float img_width, const float img_height);
        
template
void ClipBBox(vector<double>& bbox, const double img_width, const double img_height);

void AffineBBox(BBox& bbox, const float w_scale, const float h_scale, 
        const float w_shift, const float h_shift) {
  // rescale
  bbox.xmin = w_scale*bbox.xmin; bbox.xmax = w_scale*bbox.xmax;
  bbox.ymin = h_scale*bbox.ymin; bbox.ymax = h_scale*bbox.ymax;
  // shift
  bbox.xmin = bbox.xmin+w_shift; bbox.xmax = bbox.xmax+w_shift;
  bbox.ymin = bbox.ymin+h_shift; bbox.ymax = bbox.ymax+h_shift;
}

void AffineBBoxes(vector<BBox>& bboxes, const float w_scale, const float h_scale, 
        const float w_shift, const float h_shift) {
  for (int i = 0; i < bboxes.size(); i++) {
    AffineBBox(bboxes[i], w_scale, h_scale, w_shift, h_shift);
  }
}

void FlipBBox(BBox& bbox, const float width) {
  const float xmin = bbox.xmin, xmax = bbox.xmax;
  bbox.xmin = width - xmax; 
  bbox.xmax = width - xmin;
  CHECK_GE(bbox.xmax,bbox.xmin);
}

void FlipBBoxes(vector<BBox>& bboxes, const float width) {
  for (int i = 0; i < bboxes.size(); i++) {
    FlipBBox(bboxes[i], width);
  }
}
        
vector<BBox> ApplyNMS(const vector<BBox>bboxes, const float iou_thr, 
        const bool greedy, const string iou_mode) {
  vector<BBox> nms_bboxes;
  const int n = bboxes.size();
  if (n <= 0) return nms_bboxes;
  // for each i suppress all j st j>i and area-overlap>overlap
  vector<bool> kp(n, true); 
  for (int i = 0; i < n; i++){ 
    if(greedy && !kp[i]) continue; 
    for (int j = i+1; j < n; j++) {
      if(!kp[j]) continue; 
      float o = JaccardOverlap(bboxes[i], bboxes[j], iou_mode); 
      if(o>iou_thr) kp[j]=false;
    }
  }
  for (int i = 0; i < n; i++) {
    if (kp[i]) {
      nms_bboxes.push_back(bboxes[i]);
    }
  }
  return nms_bboxes;
}

template <typename Dtype>
vector<vector<Dtype> > ApplyNMS(const vector<vector<Dtype> >bboxes, 
  const float iou_thr, const bool greedy, const string iou_mode) {
  //bbox = [xmin ymin xmax ymax];
  vector<vector<Dtype> > nms_bboxes;
  const int n = bboxes.size();
  if (n <= 0) return nms_bboxes;
  // for each i suppress all j st j>i and area-overlap>overlap
  vector<bool> kp(n, true); 
  for (int i = 0; i < n; i++){ 
    if(greedy && !kp[i]) continue; 
    for (int j = i+1; j < n; j++) {
      if(!kp[j]) continue; 
      Dtype o = JaccardOverlap(bboxes[i], bboxes[j], iou_mode); 
      if(o>iou_thr) kp[j]=false;
    }
  }
  for (int i = 0; i < n; i++) {
    if (kp[i]) {
      nms_bboxes.push_back(bboxes[i]);
    }
  }
  return nms_bboxes;
}

template
vector<vector<float> > ApplyNMS(const vector<vector<float> >bboxes, 
  const float iou_thr, const bool greedy, const string iou_mode);
  
template
vector<vector<double> > ApplyNMS(const vector<vector<double> >bboxes, 
  const float iou_thr, const bool greedy, const string iou_mode);
  
template <typename Dtype>
void EncodeBBoxes(const Dtype* bbox_data, const Dtype* label, const int num, const int width,
        const int height, const Dtype* means, const Dtype* stds, 
        const float fw, const float fh, const float stride, Dtype* encode_data) {
  const int spatial_dim = width*height;
  const int bbox_dim = 4*spatial_dim;
  for (int i = 0; i < num; i++) {
    for (int s = 0; s < spatial_dim; s++) {
      const int label_value = static_cast<int>(label[i*spatial_dim+s]);
      const int idx = i*bbox_dim + s;
      if (label_value > 0) {
        const int h = s / width, w = s % width;  
        Dtype bbx, bby, bbw, bbh; 
        bbx = (bbox_data[idx]-(w+Dtype(0.5))*stride) / fw;
        bby = (bbox_data[idx+spatial_dim]-(h+Dtype(0.5))*stride) / fh;
        bbw = log(std::max(bbox_data[idx+2*spatial_dim],Dtype(2)) / fw);
        bbh = log(std::max(bbox_data[idx+3*spatial_dim],Dtype(2)) / fh);      
        // bbox normalization
        bbx -= means[0]; bby -= means[1]; bbw -= means[2]; bbh -= means[3];
        bbx /= stds[0]; bby /= stds[1]; bbw /= stds[2]; bbh /= stds[3];
      
        encode_data[idx] = bbx;
        encode_data[idx+spatial_dim] = bby;
        encode_data[idx+2*spatial_dim] = bbw;
        encode_data[idx+3*spatial_dim] = bbh;
      } else {
        encode_data[idx] = Dtype(0); 
        encode_data[idx+spatial_dim] = Dtype(0);
        encode_data[idx+2*spatial_dim] = Dtype(0); 
        encode_data[idx+3*spatial_dim] = Dtype(0);
      }
    }
  }
}

template 
void EncodeBBoxes(const float* bbox_data, const float* label, const int num, 
        const int width, const int height, const float* means, const float* stds, 
        const float fw, const float fh, const float stride, float* encode_data);

template 
void EncodeBBoxes(const double* bbox_data, const double* label, const int num, 
        const int width, const int height, const double* means, const double* stds, 
        const float fw, const float fh, const float stride, double* encode_data);

template <typename Dtype>
void EncodeGroupBBoxes(const Dtype* bbox_data, const Dtype* label, const int num, 
        const int group_num, const int width, const int height, const Dtype* means, 
        const Dtype* stds, const Dtype* fws, const Dtype* fhs, const float stride, 
        Dtype* encode_data) {
  const int spatial_dim = width*height;
  const int bbox_dim = 4*spatial_dim;
  for (int i = 0; i < num; i++) {
    for (int k = 0; k < group_num; k++) {
      for (int s = 0; s < spatial_dim; s++) {
        const int label_value = static_cast<int>(label[(i*group_num+k)*spatial_dim+s]);
        const int idx = (i*group_num+k)*bbox_dim + s;
        if (label_value > 0) {
          const int h = s / width, w = s % width;  
          Dtype bbx, bby, bbw, bbh; 
          bbx = (bbox_data[idx]-(w+Dtype(0.5))*stride) / fws[k];
          bby = (bbox_data[idx+spatial_dim]-(h+Dtype(0.5))*stride) / fhs[k];
          bbw = log(std::max(bbox_data[idx+2*spatial_dim],Dtype(2)) / fws[k]);
          bbh = log(std::max(bbox_data[idx+3*spatial_dim],Dtype(2)) / fhs[k]);      
          // bbox normalization
          bbx -= means[0]; bby -= means[1]; bbw -= means[2]; bbh -= means[3];
          bbx /= stds[0]; bby /= stds[1]; bbw /= stds[2]; bbh /= stds[3];
      
          encode_data[idx] = bbx;
          encode_data[idx+spatial_dim] = bby;
          encode_data[idx+2*spatial_dim] = bbw;
          encode_data[idx+3*spatial_dim] = bbh;
        } else {
          encode_data[idx] = Dtype(0); 
          encode_data[idx+spatial_dim] = Dtype(0);
          encode_data[idx+2*spatial_dim] = Dtype(0); 
          encode_data[idx+3*spatial_dim] = Dtype(0);
        }
      }
    }
  }
}

template
void EncodeGroupBBoxes(const float* bbox_data, const float* label, const int num, 
        const int group_num, const int width, const int height, const float* means, 
        const float* stds, const float* fws, const float* fhs, const float stride, 
        float* encode_data);

template
void EncodeGroupBBoxes(const double* bbox_data, const double* label, const int num, 
        const int group_num, const int width, const int height, const double* means, 
        const double* stds, const double* fws, const double* fhs, const float stride, 
        double* encode_data);

template <typename Dtype>
void DecodeBBoxes(const Dtype* bbox_data, const int num, const int bottom_dim, 
        const int width, const int height, const Dtype* means, const Dtype* stds, 
        const Dtype* bounds, const float fw, const float fh, 
        const float stride, const int top_dim, Dtype* pred_data) {
  const int spatial_dim = width*height;
  for (int i = 0; i < num; i++) {
    for (int s = 0; s < spatial_dim; s++) {
      const int idx = i*bottom_dim+s;
      const int h = s / width, w = s % width; 
      Dtype bbx, bby, bbw, bbh;
      // bbox de-normalization
      bbx = bbox_data[idx]*stds[0]+means[0];
      bby = bbox_data[idx+spatial_dim]*stds[1]+means[1];
      bbw = bbox_data[idx+2*spatial_dim]*stds[2]+means[2];
      bbh = bbox_data[idx+3*spatial_dim]*stds[3]+means[3];
          
      // bbox bounding
      bbx = std::max(bounds[0],bbx); bbx = std::min(bounds[1],bbx); 
      bby = std::max(bounds[0],bby); bby = std::min(bounds[1],bby);
      bbw = std::max(bounds[2],bbw); bbw = std::min(bounds[3],bbw); 
      bbh = std::max(bounds[2],bbh); bbh = std::min(bounds[3],bbh);
      
      bbx = bbx*fw + (w+Dtype(0.5))*stride;
      bby = bby*fh + (h+Dtype(0.5))*stride;         
      bbw = fw*exp(bbw); bbh = fh*exp(bbh);
      bbx = bbx - bbw/Dtype(2); bby = bby - bbh/Dtype(2);
        
      const int top_idx = i*top_dim+s;
      pred_data[top_idx] = bbx; pred_data[top_idx+spatial_dim] = bby; 
      pred_data[top_idx+2*spatial_dim] = bbx+bbw-1; 
      pred_data[top_idx+3*spatial_dim] = bby+bbh-1; 
    }
  }
}

template 
void DecodeBBoxes(const float* bbox_data, const int num, const int bottom_dim, 
        const int width, const int height, const float* means, const float* stds, 
        const float* bounds, const float fw, const float fh, 
        const float stride, const int top_dim, float* pred_data);

template 
void DecodeBBoxes(const double* bbox_data, const int num, const int bottom_dim, 
        const int width, const int height, const double* means, const double* stds, 
        const double* bounds, const float fw, const float fh, 
        const float stride, const int top_dim, double* pred_data);

template <typename Dtype>
void DecodeGroupBBoxes(const Dtype* bbox_data, const int num, const int group_num, 
        const int bottom_dim, const int width, const int height, const Dtype* means, 
        const Dtype* stds, const Dtype* bounds, const Dtype* fws, const Dtype* fhs, 
        const float stride, const int top_dim, Dtype* pred_data) {
  const int spatial_dim = width*height;
  for (int i = 0; i < num*group_num; i++) {
    const int k = i % group_num;
    for (int s = 0; s < spatial_dim; s++) {
      const int idx = i*bottom_dim+s;
      const int h = s / width, w = s % width; 
      Dtype bbx, bby, bbw, bbh;
      // bbox de-normalization
      bbx = bbox_data[idx]*stds[0]+means[0];
      bby = bbox_data[idx+spatial_dim]*stds[1]+means[1];
      bbw = bbox_data[idx+2*spatial_dim]*stds[2]+means[2];
      bbh = bbox_data[idx+3*spatial_dim]*stds[3]+means[3];
          
      // bbox bounding
      bbx = std::max(bounds[0],bbx); bbx = std::min(bounds[1],bbx); 
      bby = std::max(bounds[0],bby); bby = std::min(bounds[1],bby);
      bbw = std::max(bounds[2],bbw); bbw = std::min(bounds[3],bbw); 
      bbh = std::max(bounds[2],bbh); bbh = std::min(bounds[3],bbh);
      
      bbx = bbx*fws[k] + (w+Dtype(0.5))*stride;
      bby = bby*fhs[k] + (h+Dtype(0.5))*stride;         
      bbw = fws[k]*exp(bbw); bbh = fhs[k]*exp(bbh);
      bbx = bbx - bbw/Dtype(2); bby = bby - bbh/Dtype(2);
        
      const int top_idx = i*top_dim+s;
      pred_data[top_idx] = bbx; pred_data[top_idx+spatial_dim] = bby; 
      pred_data[top_idx+2*spatial_dim] = bbx+bbw-1; 
      pred_data[top_idx+3*spatial_dim] = bby+bbh-1; 
    }
  }
}

template 
void DecodeGroupBBoxes(const float* bbox_data, const int num, const int group_num, 
        const int bottom_dim, const int width, const int height, const float* means, 
        const float* stds, const float* bounds, const float* fws, const float* fhs, 
        const float stride, const int top_dim, float* pred_data);

template 
void DecodeGroupBBoxes(const double* bbox_data, const int num, const int group_num, 
        const int bottom_dim, const int width, const int height, const double* means, 
        const double* stds, const double* bounds, const double* fws, const double* fhs, 
        const float stride, const int top_dim, double* pred_data);

template <typename Dtype>
void DecodeBBoxesWithPrior(const Dtype* bbox_data, const vector<BBox> prior_bboxes,  
        const int bbox_dim, const Dtype* means, const Dtype* stds, 
        Dtype* pred_data) {
  const int num = prior_bboxes.size();
  const int cls_num = bbox_dim/4;
  for (int i = 0; i < num; i++) {
    Dtype pw, ph, cx, cy;
    pw = prior_bboxes[i].xmax-prior_bboxes[i].xmin+1; 
    ph = prior_bboxes[i].ymax-prior_bboxes[i].ymin+1;
    cx = 0.5*(prior_bboxes[i].xmax+prior_bboxes[i].xmin); 
    cy = 0.5*(prior_bboxes[i].ymax+prior_bboxes[i].ymin);
    for (int c = 0; c < cls_num; c++) {
      Dtype bx, by, bw, bh;
      // bbox de-normalization
      bx = bbox_data[i*bbox_dim+4*c]*stds[0]+means[0];
      by = bbox_data[i*bbox_dim+4*c+1]*stds[1]+means[1];
      bw = bbox_data[i*bbox_dim+4*c+2]*stds[2]+means[2];
      bh = bbox_data[i*bbox_dim+4*c+3]*stds[3]+means[3];

      Dtype tx, ty, tw, th;
      tx = bx*pw+cx; ty = by*ph+cy;
      tw = pw*exp(bw); th = ph*exp(bh);
      tx -= (tw-1)/2; ty -= (th-1)/2;
      pred_data[i*bbox_dim+4*c] = tx; 
      pred_data[i*bbox_dim+4*c+1] = ty;
      pred_data[i*bbox_dim+4*c+2] = tx+tw-1; 
      pred_data[i*bbox_dim+4*c+3] = ty+th-1;
    }
  }
}

template
void DecodeBBoxesWithPrior(const float* bbox_data, const vector<BBox> prior_bboxes,  
        const int bbox_dim, const float* means, const float* stds, 
        float* pred_data);

template
void DecodeBBoxesWithPrior(const double* bbox_data, const vector<BBox> prior_bboxes,  
        const int bbox_dim, const double* means, const double* stds, 
        double* pred_data);

template <typename Dtype>
void BoundBBoxPreds(Dtype* bbox_data, const int num, const int bottom_dim,
        const int sp_dim, const Dtype* bounds) {
  for (int i = 0; i < num; i++) {
    for (int j = 0; j < sp_dim; j++) {
      const int idx = i*bottom_dim+j;
      // x
      bbox_data[idx] = std::max(bounds[0],bbox_data[idx]);
      bbox_data[idx] = std::min(bounds[1],bbox_data[idx]);
      // y
      bbox_data[idx+sp_dim] = std::max(bounds[0],bbox_data[idx+sp_dim]);
      bbox_data[idx+sp_dim] = std::min(bounds[1],bbox_data[idx+sp_dim]);
      // w
      bbox_data[idx+2*sp_dim] = std::max(bounds[2],bbox_data[idx+2*sp_dim]);
      bbox_data[idx+2*sp_dim] = std::min(bounds[3],bbox_data[idx+2*sp_dim]);
      // h
      bbox_data[idx+3*sp_dim] = std::max(bounds[2],bbox_data[idx+3*sp_dim]);
      bbox_data[idx+3*sp_dim] = std::min(bounds[3],bbox_data[idx+3*sp_dim]);
    }
  }
}

template 
void BoundBBoxPreds(float* bbox_data, const int num, const int bottom_dim,
        const int sp_dim, const float* bounds);

template 
void BoundBBoxPreds(double* bbox_data, const int num, const int bottom_dim,
        const int sp_dim, const double* bounds);

template <typename Dtype>
void GetDetectionResults(const Dtype* det_data, const int num, const int dim,
      map<int, LabelBBox>* all_detections) {
  all_detections->clear();
  // [image_id, xmin, ymin, xmax, ymax, label, confidence]
  for (int i = 0; i < num; ++i) {
    int start_idx = i * dim;
    int item_id = det_data[start_idx];
    if (item_id == -1) {
      continue; // no detection in this item
    }
    int label = det_data[start_idx + 5];
    CHECK_NE(label, 0) << "Found background label in the detection results.";
    BBox bbox;
    bbox.xmin = det_data[start_idx + 1];
    bbox.ymin = det_data[start_idx + 2];
    bbox.xmax = det_data[start_idx + 3];
    bbox.ymax = det_data[start_idx + 4];
    bbox.score = det_data[start_idx + 6];
    bbox.size = BBoxSize(bbox);
    (*all_detections)[item_id][label].push_back(bbox);
  }
}

template 
void GetDetectionResults(const float* det_data,const int num, const int dim,
      map<int, LabelBBox>* all_detections);
template 
void GetDetectionResults(const double* det_data, const int num, const int dim,
      map<int, LabelBBox>* all_detections);


template <typename Dtype>
void GetGroundTruth(const Dtype* gt_data, const int num, const int dim,
      const bool use_difficult_gt, map<int, LabelBBox>* all_gt_bboxes) {
  all_gt_bboxes->clear();
  // [img_id, xmin, ymin, xmax, ymax, label, ignored, difficult]
  for (int i = 0; i < num; ++i) {
    int start_idx = i * dim;
    int item_id = gt_data[start_idx];
    if (item_id == -1) {
      continue; // no ground truth in this item
    }
    int label = gt_data[start_idx + 5];
    CHECK_GT(label, 0) << "Found background label in the dataset.";
    bool difficult = static_cast<bool>(gt_data[start_idx + 7]);
    if (!use_difficult_gt && difficult) {
      // Skip reading difficult ground truth.
      continue;
    }
    BBox bbox;
    bbox.xmin = gt_data[start_idx + 1];
    bbox.ymin = gt_data[start_idx + 2];
    bbox.xmax = gt_data[start_idx + 3];
    bbox.ymax = gt_data[start_idx + 4];
    bbox.difficult = difficult;
    bbox.size = BBoxSize(bbox);
    (*all_gt_bboxes)[item_id][label].push_back(bbox);
  }
}

template 
void GetGroundTruth(const float* gt_data, const int num, const int dim,
      const bool use_difficult_gt, map<int, LabelBBox>* all_gt_bboxes);
template 
void GetGroundTruth(const double* gt_data, const int num, const int dim,
      const bool use_difficult_gt, map<int, LabelBBox>* all_gt_bboxes);


void CumSum(const vector<pair<float, int> >& pairs, vector<int>* cumsum) {
  // Sort the pairs based on first item of the pair.
  vector<pair<float, int> > sort_pairs = pairs;
  std::stable_sort(sort_pairs.begin(), sort_pairs.end(),
                   SortScorePairDescend<int>);

  cumsum->clear();
  for (int i = 0; i < sort_pairs.size(); ++i) {
    if (i == 0) {
      cumsum->push_back(sort_pairs[i].second);
    } else {
      cumsum->push_back(cumsum->back() + sort_pairs[i].second);
    }
  }
}

void ComputeAP(const vector<pair<float, int> >& tp, const int num_pos,
               const vector<pair<float, int> >& fp, const string ap_version,
               vector<float>* prec, vector<float>* rec, float* ap) {
  const float eps = 1e-6;
  CHECK_EQ(tp.size(), fp.size()) << "tp must have same size as fp.";
  const int num = tp.size();
  // Make sure that tp and fp have complement value.
  for (int i = 0; i < num; ++i) {
    CHECK_LE(fabs(tp[i].first - fp[i].first), eps);
    CHECK_EQ(tp[i].second, 1 - fp[i].second);
  }
  prec->clear();
  rec->clear();
  *ap = 0;
  if (tp.size() == 0 || num_pos == 0) {
    return;
  }

  // Compute cumsum of tp.
  vector<int> tp_cumsum;
  CumSum(tp, &tp_cumsum);
  CHECK_EQ(tp_cumsum.size(), num);

  // Compute cumsum of fp.
  vector<int> fp_cumsum;
  CumSum(fp, &fp_cumsum);
  CHECK_EQ(fp_cumsum.size(), num);

  // Compute precision.
  for (int i = 0; i < num; ++i) {
    prec->push_back(static_cast<float>(tp_cumsum[i]) /
                    (tp_cumsum[i] + fp_cumsum[i]));
  }

  // Compute recall.
  for (int i = 0; i < num; ++i) {
    CHECK_LE(tp_cumsum[i], num_pos);
    rec->push_back(static_cast<float>(tp_cumsum[i]) / num_pos);
  }

  if (ap_version == "11point") {
    // VOC2007 style for computing AP.
    vector<float> max_precs(11, 0.);
    int start_idx = num - 1;
    for (int j = 10; j >= 0; --j) {
      for (int i = start_idx; i >= 0 ; --i) {
        if ((*rec)[i] < j / 10.) {
          start_idx = i;
          if (j > 0) {
            max_precs[j-1] = max_precs[j];
          }
          break;
        } else {
          if (max_precs[j] < (*prec)[i]) {
            max_precs[j] = (*prec)[i];
          }
        }
      }
    }
    for (int j = 10; j >= 0; --j) {
      *ap += max_precs[j] / 11;
    }
  } else if (ap_version == "101point") {
    // COCO style for computing AP.
    vector<float> max_precs(101, 0.);
    int start_idx = num - 1;
    for (int j = 100; j >= 0; --j) {
      for (int i = start_idx; i >= 0 ; --i) {
        if ((*rec)[i] < j / 100.) {
          start_idx = i;
          if (j > 0) {
            max_precs[j-1] = max_precs[j];
          }
          break;
        } else {
          if (max_precs[j] < (*prec)[i]) {
            max_precs[j] = (*prec)[i];
          }
        }
      }
    }
    for (int j = 100; j >= 0; --j) {
      *ap += max_precs[j] / 101;
    }
  } else if (ap_version == "MaxIntegral") {
    // VOC2012 or ILSVRC style for computing AP.
    float cur_rec = rec->back();
    float cur_prec = prec->back();
    for (int i = num - 2; i >= 0; --i) {
      cur_prec = std::max<float>((*prec)[i], cur_prec);
      if (fabs(cur_rec - (*rec)[i]) > eps) {
        *ap += cur_prec * fabs(cur_rec - (*rec)[i]);
      }
      cur_rec = (*rec)[i];
    }
    *ap += cur_rec * cur_prec;
  } else if (ap_version == "Integral") {
    // Natural integral.
    float prev_rec = 0.;
    for (int i = 0; i < num; ++i) {
      if (fabs((*rec)[i] - prev_rec) > eps) {
        *ap += (*prec)[i] * fabs((*rec)[i] - prev_rec);
      }
      prev_rec = (*rec)[i];
    }
  } else {
    LOG(FATAL) << "Unknown ap_version: " << ap_version;
  }
}

}  // namespace caffe
