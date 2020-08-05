#!/usr/bin/env python
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse

import grpc
from tritongrpcclient import grpc_service_v2_pb2
from tritongrpcclient import grpc_service_v2_pb2_grpc
from tritonclientutils import utils

from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression

from PIL import Image
import cv2
import numpy as np

FLAGS = None

# anchor configuration
#feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
feature_map_sizes = [[45, 45], [23, 23], [12, 12], [6, 6], [4, 4]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

# generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

# for inference , the batch size is 1, the model output shape is [1, N, 4],
# so we expand dim for anchors to [1, anchor_num, 4]
anchors_exp = np.expand_dims(anchors, axis=0)

def get_array(output, dtype):
    shape = []
    for value in output.shape:
        shape.append(value)
    return np.resize(np.frombuffer(output.contents.raw_contents, dtype=dtype), shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='192.168.7.122:31919',
                        help='Inference server URL. Default is localhost:8001.')

    FLAGS = parser.parse_args()

    # We use a simple model that takes 2 input tensors of 16 integers
    # each and returns 2 output tensors of 16 integers each. One
    # output tensor is the element-wise sum of the inputs and one
    # output is the element-wise difference.
    model_name = "kitmodel"
    model_version = ""
    batch_size = 1

    # Create gRPC stub for communicating with the server
    channel = grpc.insecure_channel(FLAGS.url)
    grpc_stub = grpc_service_v2_pb2_grpc.GRPCInferenceServiceStub(channel)

    # Health
    try:
        request = grpc_service_v2_pb2.ServerLiveRequest()
        response = grpc_stub.ServerLive(request)
        print("server {}".format(response))
    except Exception as ex:
        print(ex)

    request = grpc_service_v2_pb2.ServerReadyRequest()
    response = grpc_stub.ServerReady(request)
    print("server {}".format(response))

    request = grpc_service_v2_pb2.ModelReadyRequest(
        name=model_name, version=model_version)
    response = grpc_stub.ModelReady(request)
    print("model {}".format(response))

    # Metadata
    request = grpc_service_v2_pb2.ServerMetadataRequest()
    response = grpc_stub.ServerMetadata(request)
    print("server metadata:\n{}".format(response))

    request = grpc_service_v2_pb2.ModelMetadataRequest(
        name=model_name, version=model_version)
    response = grpc_stub.ModelMetadata(request)
    print("model metadata:\n{}".format(response))

    # Configuration
    request = grpc_service_v2_pb2.ModelConfigRequest(
        name=model_name, version=model_version)
    response = grpc_stub.ModelConfig(request)
    print("model config:\n{}".format(response))

    # Infer
    request = grpc_service_v2_pb2.ModelInferRequest()
    request.model_name = model_name
    request.model_version = model_version
    request.id = model_name+"-id-0"

    ## Input data
    img = cv2.imread('triton/sample_data/maskimage.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    target_shape = (360, 360)
    height, width, _ = img.shape
    resized_img = cv2.resize(img, target_shape)
    image_np = resized_img / 255.0
    image_exp = np.expand_dims(image_np, axis=0)
    image_transposed = image_exp.transpose((0, 3, 1, 2))

    raw_input = np.float32(image_transposed).flatten().tobytes()

    conv2d = grpc_service_v2_pb2.ModelInferRequest().InferInputTensor()
    conv2d.name = "conv2d__0"
    conv2d.shape.extend([1, 3, 360, 360])

    conv2d_contents = grpc_service_v2_pb2.InferTensorContents()
    conv2d_contents.raw_contents = raw_input
    conv2d.contents.CopyFrom(conv2d_contents)

    request.inputs.extend([conv2d])

    loc_conv = grpc_service_v2_pb2.ModelInferRequest().InferRequestedOutputTensor()
    loc_conv.name = "loc_branch_concat__0"
    cls_conv = grpc_service_v2_pb2.ModelInferRequest().InferRequestedOutputTensor()
    cls_conv.name = "cls_branch_concat__1"
    request.outputs.extend([loc_conv, cls_conv])

    response = grpc_stub.ModelInfer(request)
    print("model infer:\n{}".format(response))

    # Post process
    id2class = {0: 'Mask', 1: 'NoMask'}
    y_cls_output = get_array(response.outputs[0], np.float32)
    y_bboxes_output = get_array(response.outputs[1], np.float32)
    # remove the batch dimension, for batch is always 1 for inference.
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    # To speed up, do single class NMS, not multiple classes NMS.
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)

    # keep_idx is the alive bounding box after nms.
    keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                 bbox_max_scores,
                                                 conf_thresh=0.5,
                                                 iou_thresh=0.5,
                                                 )
    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        # clip the coordinate, avoid the value exceed the image boundary.
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)

        if class_id == 0:
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(img, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)
        
    Image.fromarray(img).show()

    print("PASS")
