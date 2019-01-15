# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick, Sean Bell and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
np.set_printoptions(threshold=np.nan)
import numpy.random as npr
from model.config import cfg
from model.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps


def proposal_target_layer(rpn_rois, rpn_scores, gt_boxes, _num_classes):
  """
  Assign object detection proposals to ground-truth targets. Produces proposal
  classification labels and bounding-box regression targets.
  """

  # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
  # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
  all_rois = rpn_rois
  all_scores = rpn_scores

  # Include ground-truth boxes in the set of candidate rois
  if cfg.TRAIN.USE_GT:
    zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
    all_rois = np.vstack(
      (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
    )
    # not sure if it a wise appending, but anyway i am not using it
    all_scores = np.vstack((all_scores, zeros))

  num_images = 1
  rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
  fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

  # Sample rois with classification labels and bounding box regression
  # targets
  labels, rois, roi_scores, bbox_targets, bbox_inside_weights, choose_box = _sample_rois(
    all_rois, all_scores, gt_boxes, fg_rois_per_image,
    rois_per_image, _num_classes)

  rois = rois.reshape(-1, 5)
  roi_scores = roi_scores.reshape(-1)
  labels = labels.reshape(-1, 1)
  bbox_targets = bbox_targets.reshape(-1, _num_classes * 4)
  bbox_inside_weights = bbox_inside_weights.reshape(-1, _num_classes * 4)
  bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)

  return rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, choose_box


def _get_bbox_regression_labels(bbox_target_data, num_classes):
  """Bounding-box regression targets (bbox_target_data) are stored in a
  compact form N x (class, tx, ty, tw, th)

  This function expands those targets into the 4-of-4*K representation used
  by the network (i.e. only one class has non-zero targets).

  Returns:
      bbox_target (ndarray): N x 4K blob of regression targets
      bbox_inside_weights (ndarray): N x 4K blob of loss weights
  """

  clss = bbox_target_data[:, 0]
  bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
  bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
  inds = np.where(clss > 0)[0]
  for ind in inds:
    cls = clss[ind]
    start = int(4 * cls)
    end = start + 4
    bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
    bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
  return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
  """Compute bounding-box regression targets for an image."""

  assert ex_rois.shape[0] == gt_rois.shape[0]
  assert ex_rois.shape[1] == 4
  assert gt_rois.shape[1] == 4

  targets = bbox_transform(ex_rois, gt_rois)
  if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
    # Optionally normalize targets by a precomputed mean and stdev
    targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
               / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
  return np.hstack(
    (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)


def _sample_rois(all_rois, all_scores, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
  """Generate a random sample of RoIs comprising foreground and background
  examples.
  """
  # overlaps: (rois x gt_boxes)
  choose_box = 0
  overlaps = bbox_overlaps(
    np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
    np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float)) #计算all_rois和gt_boxes之间的重叠部分 overlaps[n, k] n是all_rois.shape[0],K是gt_boxes.shape[0]
  gt_assignment = overlaps.argmax(axis=1)  #找到每一个rois 对应的groundtruth位置
  max_overlaps = overlaps.max(axis=1) #找到每一个rois 对应的groundtruth其中anchors的score
  labels = gt_boxes[gt_assignment, 4]#gt_labels的第五个位置的信息赋给labels class信息
  # Select foreground RoIs as those with >= FG_THRESH overlap
  fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0] #找到前景 max_overlaps>=0.5
  # Guard against the case when an image has fewer than fg_rois_per_image  #预防找到的前景少于fg_rois_per_image的情况
  # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
  bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                     (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0] # 找到0.1<overlaps<0.5的位置当作背景
  # TODO: 改动de起始点
  fg_rois_per_image_temp = fg_rois_per_image
  # Small modification to the original version where we ensure a fixed number of regions are sampled
  if fg_inds.size > 0 and bg_inds.size > 0: #如果前景与背景都存在
    fg_rois_per_image = min(fg_rois_per_image, fg_inds.size) #找到真正的fg_rois_per_image的数量
    fg_inds = npr.choice(fg_inds, size=int(fg_rois_per_image), replace=False) #随机生成fg_rois_per_image数量的fg_inds
    choose_box = fg_inds.size
    # fg_rois = all_rois[fg_inds]
    num_temp = np.int((fg_rois_per_image_temp - fg_inds.size) // fg_inds.size) # 每个框增值的数量
    res_temp = np.int((fg_rois_per_image_temp - fg_inds.size) % fg_inds.size) # 最后一个框多增值的数量
    import copy
    cp_fg_inds = copy.deepcopy(fg_inds)
    
    for idx in cp_fg_inds:
      box_group_temp = []
      for i1 in range(3):
        for i2 in range(3):
          for i3 in range(3):
            for i4 in range(3):

              if i1 == 1 and i2 == 1 and i3 == 1 and i4 == 1:
                continue
              # print("----------------------------------------")
              # print(all_rois[idx].copy())
              # print(i1,i2,i3,i4)
              box_temp = all_rois[idx].copy()
              box_add_temp = [0, (i1-1) * 0.1 * (box_temp[3] - box_temp[1]),
                              (i2-1) * 0.1 * (box_temp[4] - box_temp[2]),
                              (i3-1) * 0.1 * (box_temp[3] - box_temp[1]),
                              (i4 - 1) * 0.1 * (box_temp[4] - box_temp[2]),]
              box_temp += box_add_temp
              box_group_temp.append(box_temp)
              # print(box_temp)
              # print("----------------------------------------")
      box_group_temp = np.array(box_group_temp)
      # print("=====================\n",box_group_temp)
      # print('88888888888888888888888888888888888888888888888888888888')
      # print(np.shape(all_rois))
      # print(box_group_temp.shape)
      # print("jieguo",res_temp)
      # 选框
      # overlaps: (rois x gt_boxes)
      overlaps1 = bbox_overlaps(
        np.ascontiguousarray(box_group_temp[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4],
                             dtype=np.float))  # 计算all_rois和gt_boxes之间的重叠部分 overlaps[n, k] n是all_rois.shape[0],K是gt_boxes.shape[0]
      gt_assignment1 = overlaps1.argmax(axis=1)  # 找到每一个rois 对应的groundtruth位置
      max_overlaps1 = overlaps1.max(axis=1)  # 找到每一个rois 对应的groundtruth其中anchors的score
      order_temp = np.argsort(-max_overlaps1) # 降序排序
      box_group_temp = box_group_temp[order_temp] # box
      gt_assignment1 = gt_assignment1[order_temp] # 对应groundtruth编号
      # print('#######',gt_assignment1)
      # print(num_temp)
      if idx != cp_fg_inds[-1]:
        box_group_temp = box_group_temp[:num_temp]
        gt_assignment1 = gt_assignment1[:num_temp]
        # fg_rois_per_image += num_temp # 根据新增的正样本数量
      else:
        box_group_temp = box_group_temp[:num_temp + res_temp]
        gt_assignment1 = gt_assignment1[:num_temp + res_temp]
        # fg_rois_per_image += num_temp + res_temp # 根据新增的正样本数量
      # box_group_temp = tf.convert_to_tensor(box_group_temp)

      # print(all_rois.shape)
      # print(box_group_temp.shape)
      for i in range(len(box_group_temp)):
        fg_inds = np.append(fg_inds, len(all_rois) + i) # 将新的box的序号添加到fg_inds
        labels = np.append(labels, labels[idx])
        all_scores = np.append(all_scores, 0.5) # 防止返回值越界
       
      # labels += gt_boxes[gt_assignment1, 4]  # gt_labels的第五个位置的信息赋给labels class信息
      all_rois = np.concatenate((all_rois, box_group_temp), axis=0)  # 新的box添加到原来的rois后面
      gt_assignment = np.concatenate((gt_assignment, gt_assignment1), axis=0)
      # Select foreground RoIs as those with >= FG_THRESH overlap
      # fg_inds1 = np.where(max_overlaps1 >= cfg.TRAIN.FG_THRESH)[0]  # 找到前景 max_overlaps>=0.5
    if fg_inds.size != fg_rois_per_image_temp:
      lack = fg_rois_per_image_temp - fg_inds.size
    else:
      lack = 0

    # print(all_rois.shape)
    # print(all_scores.shape)
    bg_rois_per_image = rois_per_image - fg_rois_per_image_temp + lack # 每张图片上的rois-fg_rois就是bg_rois_per_image
    to_replace = bg_inds.size < bg_rois_per_image # 如果bg_inds.size少于bg_rois_per_image 也就是说出现更多的前景
    bg_inds = npr.choice(bg_inds, size=int(bg_rois_per_image), replace=to_replace) #随机选取bg_rois_per_image数量的bg_inds
  elif fg_inds.size > 0:
    to_replace = fg_inds.size < rois_per_image
    fg_inds = npr.choice(fg_inds, size=int(rois_per_image), replace=to_replace)
    fg_rois_per_image = rois_per_image
  elif bg_inds.size > 0:
    to_replace = bg_inds.size < rois_per_image
    bg_inds = npr.choice(bg_inds, size=int(rois_per_image), replace=to_replace)
    fg_rois_per_image = 0
  else:
    import pdb
    pdb.set_trace()

  # The indices that we're selecting (both fg and bg)
  # print(len(fg_inds))
  keep_inds = np.append(fg_inds, bg_inds)
  # print(keep_inds)
  # Select sampled values from various arrays:
  # print("=====================\n",labels,len(labels))
  labels = labels[keep_inds] #去掉<0.1的值
  # print("=====================\n",labels,len(labels))
  # print(len(labels))
  # print(labels)
  # Clamp labels for the background RoIs to 0
  labels[int(fg_rois_per_image_temp):] = 0 #将background的标签设置为0
  # print(labels)
  rois = all_rois[keep_inds] #找到在0.1~1之间的rois
  roi_scores = all_scores[keep_inds] #找到在0.1~1之间的roi_scores

  bbox_target_data = _compute_targets(
    rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels) #返回pre 与gt 之间的tx值 并经过normalize处理 形式是labels+4tx
  # print(gt_assignment[keep_inds])
  bbox_targets, bbox_inside_weights = \
    _get_bbox_regression_labels(bbox_target_data, num_classes)
  rois = np.float32(rois)
  roi_scores = np.float32(roi_scores)
  # with open("log.txt","a") as f:
  #   f.write("label_list:\n"+str(labels) +
  #           "\n" + "fg_inds:\n" + str(fg_inds) + "\n" 
  #           "roi_scores:\n" + str(roi_scores)+"\n" +
  #           "bbox_targets:\n" + str(bbox_targets) + "\n" +
  #           "rois:\n" + str(rois))
  return labels, rois, roi_scores, bbox_targets, bbox_inside_weights, choose_box #返回每一个anchor对应的类别信息，rois设定为0.1~1之间的roi,bbox_targets的信息是在其对应坐标上的，bbox_inside_weights的也是

