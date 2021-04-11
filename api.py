import os
from pathlib import Path

import numpy as np
import cv2
from faster_rcnn_wrapper import FasterRCNNSlim
from _tf_compat_import import compat_tensorflow as tf
import argparse
from nms_wrapper import NMSType, NMSWrapper

def detect(sess, rcnn_cls, image):
    # pre-processing image for Faster-RCNN
    img_origin = image.astype(np.float32, copy=True)
    img_origin -= np.array([[[102.9801, 115.9465, 112.7717]]])

    img_shape = img_origin.shape
    img_size_min = np.min(img_shape[:2])
    img_size_max = np.max(img_shape[:2])

    img_scale = 600 / img_size_min
    if np.round(img_scale * img_size_max) > 1000:
        img_scale = 1000 / img_size_max
    img = cv2.resize(img_origin, None, None, img_scale, img_scale, cv2.INTER_LINEAR)
    img_info = np.array([img.shape[0], img.shape[1], img_scale], dtype=np.float32)
    img = np.expand_dims(img, 0)

    # test image
    _, scores, bbox_pred, rois = rcnn_cls.test_image(sess, img, img_info)

    # bbox transform
    boxes = rois[:, 1:] / img_scale

    boxes = boxes.astype(bbox_pred.dtype, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1
    heights = boxes[:, 3] - boxes[:, 1] + 1
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights
    dx = bbox_pred[:, 0::4]
    dy = bbox_pred[:, 1::4]
    dw = bbox_pred[:, 2::4]
    dh = bbox_pred[:, 3::4]
    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]
    pred_boxes = np.zeros_like(bbox_pred, dtype=bbox_pred.dtype)
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h
    # clipping edge
    pred_boxes[:, 0::4] = np.maximum(pred_boxes[:, 0::4], 0)
    pred_boxes[:, 1::4] = np.maximum(pred_boxes[:, 1::4], 0)
    pred_boxes[:, 2::4] = np.minimum(pred_boxes[:, 2::4], img_shape[1] - 1)
    pred_boxes[:, 3::4] = np.minimum(pred_boxes[:, 3::4], img_shape[0] - 1)
    return scores, pred_boxes

class FaceDetector():
    def __init__(self, model_path=None, nms_type='CPU_NMS', nms_threshold=0.3, threshold=0.8):
        """

        Parameters:
            nms_type: Type of NMS. Options ("PY_NMS" | "CPU_NMS" | "GPU_NMS")
            nms_threshold: Threshold for Non Max Suppression. (float)
            threshold: Threshold for class regression. (float)
            model_path: Path to the model. If none is provided, it uses the one inside this directory.
            quiet: Disable tensorflow display messages. (boolean)
        """

        # nms_type check
        if nms_type == 'PY_NMS':
            nms_type = NMSType.PY_NMS
        elif nms_type == 'CPU_NMS':
            nms_type = NMSType.CPU_NMS
        elif nms_type == 'GPU_NMS':
            nms_type = NMSType.GPU_NMS
        else:
            raise ValueError('Incorrect NMS Type, not supported yet')

        self.nms = NMSWrapper(nms_type)
        self.nms_threshold = nms_threshold
        self.cls_threshold = threshold

        cfg = tf.compat.v1.ConfigProto()
        cfg.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=cfg)

        self.net = FasterRCNNSlim()
        saver = tf.compat.v1.train.Saver()

        if model_path is None:
            model_path = str(Path(__file__).resolve().parent / 'model/res101_faster_rcnn_iter_60000.ckpt')

        saver.restore(self.sess, model_path)

    def detect_faceboxes(self, image):
        """
            Returns an array of faceboxes for each face detected.
            Facebox: (x1, y1, width, height) (int32)

            Parameters:
                Image: Numpy RGB Image
        """
        scores, boxes = detect(self.sess, self.net, image)
        boxes = boxes[:, 4:8]
        scores = scores[:, 1]
        keep = self.nms(np.hstack([boxes, scores[:, np.newaxis]]).astype(np.float32), self.nms_threshold)
        boxes = boxes[keep, :]
        scores = scores[keep]
        inds = np.where(scores >= self.cls_threshold)[0]
        scores = scores[inds]
        boxes = boxes[inds, :]

        faceboxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            facebox = tuple(np.array([x1, y1, x2 - x1, y2 - y1]).astype(np.int32))
            faceboxes.append(facebox)

        return faceboxes
