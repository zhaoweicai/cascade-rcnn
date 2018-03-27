% Cascade-RCNN
% Copyright (c) 2018 The Regents of the University of California
% see cascade-rcnn/LICENSE for details
% Written by Zhaowei Cai [zwcai-at-ucsd.edu]
% Please email me if you find bugs, or have suggestions or questions!

clear all; close all;
addpath('../../matlab/'); addpath('../../utils/');
cocoDir = '/your/path/to/coco'; % your COCO directory
addpath(genpath([cocoDir '/MatlabAPI']));

%% experimental setup %%
% network
root_dir = './res101-15s-800-fpn-cascade-pretrained/';
binary_file = [root_dir 'cascadercnn_coco_iter_280000.caffemodel'];
assert(exist(binary_file, 'file') ~= 0); 
definition_file = [root_dir 'deploy.prototxt'];
assert(exist(definition_file, 'file') ~= 0);
use_gpu = true;
if ~use_gpu
  caffe.set_mode_cpu();
else
  caffe.set_mode_gpu(); gpu_id = 0;
  caffe.set_device(gpu_id);
end
net = caffe.Net(definition_file, binary_file, 'test');

% dataset
annTypes = { 'instances', 'captions', 'person_keypoints' };
dataType='val2017'; annType=annTypes{1}; % specify dataType/annType
annFile=sprintf([cocoDir '/annotations/%s_%s.json'],annType,dataType);
coco=CocoApi(annFile); 
cats = coco.loadCats(coco.getCatIds()); obj_names = {cats.name};
IDs = [cats.id]; num_cls=length(IDs); obj_ids = 2:num_cls+1;
imgIds=sort(coco.getImgIds()); nImg = length(imgIds);

% architecture
if(~isempty(strfind(root_dir, 'cascade'))), CASCADE = 1;
else CASCADE = 0; end
if(~isempty(strfind(root_dir, 'fpn'))), FPN = 1;
else FPN = 0; end
if (~CASCADE)
  % baseline model
  if (FPN)
    proposal_blob_names = {'proposals_to_all'};
  else
    proposal_blob_names = {'proposals'};
  end
  bbox_blob_names = {'output_bbox_1st'};
  cls_prob_blob_names = {'cls_prob_1st'};
  output_names = {'1st'};
else
  % cascade-rcnn model
  if (FPN)
    proposal_blob_names = {'proposals_to_all','proposals_to_all_2nd',...
        'proposals_to_all_3rd','proposals_to_all_2nd','proposals_to_all_3rd'};
  else
    proposal_blob_names = {'proposals','proposals_2nd','proposals_3rd',...
        'proposals_2nd','proposals_3rd'};
  end
  bbox_blob_names = {'output_bbox_1st','output_bbox_2nd','output_bbox_3rd',...
      'output_bbox_2nd','output_bbox_3rd'};
  cls_prob_blob_names = {'cls_prob_1st','cls_prob_2nd','cls_prob_3rd',...
      'cls_prob_2nd_avg','cls_prob_3rd_avg'};
  output_names = {'1st','2nd','3rd','2nd_avg','3rd_avg'};
end
num_outputs = numel(proposal_blob_names);
assert(num_outputs==numel(bbox_blob_names));
assert(num_outputs==numel(cls_prob_blob_names));
assert(num_outputs==numel(output_names));

% detection configuration
detect_final_boxes = cell(nImg,num_outputs);
det_thr = 0.001; % threoshold
max_per_img = 100; % max number of detections
pNms.type = 'maxg'; pNms.overlap = 0.5; pNms.ovrDnm = 'union'; % NMS

% saveing ID and evaluation log
comp_id = 'tmp'; % specify a unique ID if you want to archive the results
log_id = '280k';
eval_log = sprintf('%scoco_eval_%s_%s.txt',root_dir,log_id,dataType); 

% image pre-processing
if (FPN), shortSize = 800; longSize = 1312;
else shortSize = 600; longSize = 1000; end
mu0 = ones(1,1,3); mu0(:,:,1:3) = [104 117 123];

% detection showing
show = 0; show_thr = 0.1; usedtime=0; 
if (show)
  fig=figure(1); set(fig,'Position',[50 100 600 600]);
  h.axes = axes('position',[0,0,1,1]);
end

%% running %%
for kk = 1:nImg
  imgId = imgIds(kk);
  imgInfo = coco.loadImgs(imgId);
  img = imread(sprintf('%s/images/%s/%s',cocoDir,dataType,imgInfo.file_name));
  orgImg = img;
  if (size(img,3)==1), img = repmat(img,[1 1 3]); end
  [orgH,orgW,~] = size(img);
  
  rzRatio = shortSize/min(orgH,orgW);
  imgH = min(rzRatio*orgH,longSize); imgW = min(rzRatio*orgW,longSize);
  imgH = round(imgH/32)*32; imgW = round(imgW/32)*32; % must be the multiple of 32
  hwRatios = [imgH imgW]./[orgH orgW];
  img = imresize(img,[imgH imgW]); 
  mu = repmat(mu0,[imgH,imgW,1]);
  img = single(img(:,:,[3 2 1]));
  img = bsxfun(@minus,img,mu);
  img = permute(img, [2 1 3]);

  % network forward
  tic; outputs = net.forward({img}); pertime=toc;
  usedtime=usedtime+pertime; avgtime=usedtime/kk;
    
  for nn = 1:num_outputs
    if (show)
      imshow(orgImg,'parent',h.axes); axis(h.axes,'image','off');
    end
    detect_boxes = cell(num_cls,1); 
    tmp = squeeze(net.blobs(bbox_blob_names{nn}).get_data()); 
    tmp = tmp'; tmp = tmp(:,2:end);
    tmp(:,[1,3]) = tmp(:,[1,3])./hwRatios(2);
    tmp(:,[2,4]) = tmp(:,[2,4])./hwRatios(1);
    % clipping bbs to image boarders
    tmp(:,[1,2]) = max(0,tmp(:,[1,2]));
    tmp(:,3) = min(tmp(:,3),orgW); tmp(:,4) = min(tmp(:,4),orgH);
    tmp(:,[3,4]) = tmp(:,[3,4])-tmp(:,[1,2])+1;
    output_bboxs = double(tmp);  
    
    tmp = squeeze(net.blobs(cls_prob_blob_names{nn}).get_data()); 
    cls_prob = tmp'; 
    
    tmp = squeeze(net.blobs(proposal_blob_names{nn}).get_data());
    tmp = tmp'; tmp = tmp(:,2:end); 
    tmp(:,[3,4]) = tmp(:,[3,4])-tmp(:,[1,2])+1;
    proposals = tmp;
    
    keep_id = find(proposals(:,3)~=0 & proposals(:,4)~=0);
    proposals = proposals(keep_id,:); 
    output_bboxs = output_bboxs(keep_id,:); cls_prob = cls_prob(keep_id,:);

    for i = 1:num_cls
      id = obj_ids(i);        
      prob = cls_prob(:,id);         
      bbset = double([output_bboxs prob]);
      if (det_thr>0)
        keep_id = find(prob>=det_thr); bbset = bbset(keep_id,:);
      end
      bbset=bbNms(bbset,pNms);
      detect_boxes{i} = [ones(size(bbset,1),1)*IDs(i) bbset(:,1:5)];
        
      if (show) 
        bbs_show = zeros(0,6);
        if (size(bbset,1)>0) 
          show_id = find(bbset(:,5)>=show_thr);
          bbs_show = bbset(show_id,:);
        end
        for j = 1:size(bbs_show,1)
          rectangle('Position',bbs_show(j,1:4),'EdgeColor','y','LineWidth',2);
          show_text = sprintf('%s=%.2f',obj_names{i},bbs_show(j,5));
          x = bbs_show(j,1)+0.5*bbs_show(j,3);
          text(x,bbs_show(j,2),show_text,'color','r', 'BackgroundColor','k',...
              'HorizontalAlignment','center', 'VerticalAlignment','bottom',...
              'FontWeight','bold', 'FontSize',8);
        end  
      end
    end
    detect_boxes=cell2mat(detect_boxes);
    if (max_per_img>0 && size(detect_boxes,1)>max_per_img)
      rank_scores = sort(detect_boxes(:,6),'descend');
      keep_id = find(detect_boxes(:,6)>=rank_scores(max_per_img));
      detect_boxes = detect_boxes(keep_id,:);
    end
    detect_final_boxes{kk,nn} = [ones(size(detect_boxes,1),1)*imgId detect_boxes]; 
  end
  if (mod(kk,100)==0), fprintf('idx %i/%i, avgtime=%.4fs\n',kk,nImg,avgtime); end
end

%% evaluation %%
save_dir = 'detections/';
if (~exist(save_dir)), mkdir(save_dir); end
diary(eval_log); diary('on');
for nn = 1:num_outputs
  % saving
  resFile = sprintf('%s%s_%s_%s_results.json',save_dir,comp_id,dataType,output_names{nn});
  detect_boxes=cell2mat(detect_final_boxes(:,nn));
  num_objs = size(detect_boxes,1);
  res = cell(num_objs,1);
  for j = 1:num_objs
    res{j}.image_id = detect_boxes(j,1);
    res{j}.category_id = detect_boxes(j,2);
    res{j}.bbox = round(detect_boxes(j,3:6)*10)/10;
    res{j}.score = round(detect_boxes(j,7)*1000)/1000;
  end
  res_str = gason(res);
  fid = fopen(resFile,'w');
  fprintf(fid,res_str); fclose(fid);
  % eval
  fprintf('====================evaluating %s====================\n',output_names{nn});
  type = {'segm','bbox'}; type = type{2};
  cocoDt=coco.loadRes(resFile);
  cocoEval=CocoEval(coco,cocoDt);
  cocoEval.params.imgIds=imgIds;
  cocoEval.params.useSegm=strcmp(type,'segm');
  cocoEval.evaluate();
  cocoEval.accumulate();
  cocoEval.summarize();
  fprintf('\n\n');
end
diary('off');

caffe.reset_all();
