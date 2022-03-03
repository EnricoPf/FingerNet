import os
import cv2
import sys
import time
import glob

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import onnxruntime as ort

from data.config import cfg_re50
from utils.bbox import draw_pred
from utils.processing import preprocessing, postprocessing

from openvino.inference_engine import IECore

sys.path.remove(os.path.dirname(os.path.abspath(__file__)))

# PostProcessing Index Mapping:
        # 0 -  x from p1 of bbox
        # 1 -  y from p1 of bbox
        # 2 -  x from p2 of bbox
        # 3 -  y from p2 of bbox
        # 4 -  Confidence Score
        # 5 -  x from Left Eye
        # 6 -  y from Left Eye
        # 7 -  x from Right Eye
        # 8 -  y from Right Eye
        # 9 -  x from Nose
        # 10 - y from Nose
        # 11 - x from Left Mouth Corner
        # 12 - y from Left Mouth Corner
        # 13 - x from Right Mouth Corner
        # 14 - y from Right Mouth Corner

def detect_openvino(im, reshape=False, XML_PATH = "openvino_dynamic/retinaface_dynamic.xml", BIN_PATH = "openvino_dynamic/retinaface_dynamic.bin", **kwargs):
    if 'model_path' in kwargs.keys():
        if os.path.isdir(kwargs.get('model_path')):
            try:
                XML_PATH = glob.glob(os.path.join(kwargs.get('model_path'), '*.xml'))[0]
                BIN_PATH = glob.glob(os.path.join(kwargs.get('model_path'), '*.bin'))[0]
            except:
                return None

        # Assumes its the prefixed path without extension (XML and BIN file has the same prefix)
        else:
            XML_PATH = kwargs.get('model_path') + '.xml'
            BIN_PATH = kwargs.get('model_path') + '.bin'
    
    im, scale = preprocessing(im, reshape)
    # Initializing OpenVINO model
    ie = IECore()

    network = ie.read_network(model=XML_PATH,weights=BIN_PATH)
    input_blob = next(iter(network.input_info))
    network.reshape({input_blob: im.shape})

    model = ie.load_network(network, device_name="CPU")
    out =model.infer(inputs={input_blob: im})

    return postprocessing(out, cfg_re50, (im.shape[2], im.shape[3]), 0.7, det_scale=scale)

def det_openvino(im_file, reshape=False, XML_PATH = "openvino_dynamic/retinaface_dynamic.xml", BIN_PATH = "openvino_dynamic/retinaface_dynamic.bin", **kwargs):
    
    im = cv2.imread(im_file)

    if 'model_path' in kwargs.keys():
        if os.path.isdir(kwargs.get('model_path')):
            try:
                XML_PATH = glob.glob(os.path.join(kwargs.get('model_path'), '*.xml'))[0]
                BIN_PATH = glob.glob(os.path.join(kwargs.get('model_path'), '*.bin'))[0]
            except:
                return None, None
        # Assumes its the prefixed path without extension (XML and BIN file has the same prefix)
        else:
            XML_PATH = kwargs.get('model_path') + '.xml'
            BIN_PATH = kwargs.get('model_path') + '.bin'
    
    im, scale = preprocessing(im, reshape)
    # Initializing OpenVINO model
    ie = IECore()

    network = ie.read_network(model=XML_PATH,weights=BIN_PATH)
    input_blob = next(iter(network.input_info))
    network.reshape({input_blob: im.shape})

    model = ie.load_network(network, device_name="CPU")
    out =model.infer(inputs={input_blob: im})

    result = postprocessing(out, cfg_re50, (im.shape[2], im.shape[3]), 0.7, det_scale=scale)

    return result[:,:5] , result[:,5:]

def detect_onnx(im, reshape=False, model_path='retinaface_dynamic.onnx', **kwargs):
    im, scale = preprocessing(im, reshape)


    # Initializinig ONNX Session
    sess_opt = ort.SessionOptions()
    # sess_opt.intra_op_num_threads = 1
    # sess_opt.inter_op_num_threads = 1
    sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess = ort.InferenceSession(model_path, sess_opt)
    input_names = [input.name for input in sess.get_inputs()]
    output_names = [output.name for output in sess.get_outputs()]

    out = sess.run(output_names, {input_names[0]: im})
    out = {'face_rpn_bbox_pred': out[1], 'face_rpn_cls_prob': out[0], 'face_rpn_landmark_pred': out[2]}

    return postprocessing(out, cfg_re50, (im.shape[2],im.shape[3]), 0.7, det_scale=scale)

def det_onnx(im_file, reshape=False, model_path='retinaface_dynamic.onnx', **kwargs):
    im = cv2.imread(im_file)
    im, scale = preprocessing(im, reshape)

    # Initializinig ONNX Session
    sess_opt = ort.SessionOptions()
    sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess = ort.InferenceSession(model_path, sess_opt)
    input_names = [input.name for input in sess.get_inputs()]
    output_names = [output.name for output in sess.get_outputs()]

    out = sess.run(output_names, {input_names[0]: im})
    out = {'face_rpn_bbox_pred': out[1], 'face_rpn_cls_prob': out[0], 'face_rpn_landmark_pred': out[2]}

    result = postprocessing(out, cfg_re50, (im.shape[2],im.shape[3]), 0.7, det_scale=scale)

    return result[:,:5], result[:,5:]

if __name__ == "__main__":
    begin = time.time()
    im = cv2.imread(sys.argv[1])
    dets = detect_openvino(im, True) if len(sys.argv) < 3 else detect_onnx(im, True)

    end = time.time() - begin
    print("Elapsed Time: {}".format(end))
    # print(im.shape)
    for i, det in enumerate(dets):
        im = draw_pred(im, det)
        # print("Box[{}]: ({:.1f}, {:.1f})".format(i, det[2]-det[0], det[3]-det[1]))
    cv2.imshow(sys.argv[1], im)
    cv2.waitKey(0)