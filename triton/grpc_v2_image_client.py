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

import cv2
import numpy as np

FLAGS = None

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
    model_name = "model_sk"
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
    height = img.shape[0]
    width = img.shape[1]

    target_shape = (260, 260)
    resized_img = cv2.resize(img, target_shape)
    image_np = resized_img / 255.0
    image_exp = np.expand_dims(image_np, axis=0)
    processed_data = np.float32(image_exp)

    raw_input = processed_data.flatten().tobytes()
    raw_height = np.array([np.int64(height)]).tobytes()
    raw_width = np.array([np.int64(width)]).tobytes()

    input_data = grpc_service_v2_pb2.ModelInferRequest().InferInputTensor()
    input_data.name = "data"
    input_data.shape.extend([1, 260, 260, 3])

    input_data_contents = grpc_service_v2_pb2.InferTensorContents()
    input_data_contents.raw_contents = raw_input
    input_data.contents.CopyFrom(input_data_contents)

    input_height = grpc_service_v2_pb2.ModelInferRequest().InferInputTensor()
    input_height.name = "height"
    input_height.shape.extend([1, 1])

    input_height_contents = grpc_service_v2_pb2.InferTensorContents()
    input_height_contents.raw_contents = raw_height
    input_height.contents.CopyFrom(input_height_contents)

    input_width = grpc_service_v2_pb2.ModelInferRequest().InferInputTensor()
    input_width.name = "width"
    input_width.shape.extend([1, 1])

    input_width_contents = grpc_service_v2_pb2.InferTensorContents()
    input_width_contents.raw_contents = raw_width
    input_width.contents.CopyFrom(input_width_contents)

    request.inputs.extend([input_data, input_height, input_width])

    output_bbox = grpc_service_v2_pb2.ModelInferRequest().InferRequestedOutputTensor()
    output_bbox.name = "tf_op_layer_bboxes"
    output_class_id = grpc_service_v2_pb2.ModelInferRequest().InferRequestedOutputTensor()
    output_class_id.name = "tf_op_layer_class_id"
    output_prob = grpc_service_v2_pb2.ModelInferRequest().InferRequestedOutputTensor()
    output_prob.name = "tf_op_layer_prob"
    request.outputs.extend([output_bbox, output_class_id, output_prob])

    response = grpc_stub.ModelInfer(request)
    print("model infer:\n{}".format(response))

    # Post process
    id2class = {0: 'Mask', 1: 'NoMask'}
    bboxes = get_array(response.outputs[0], np.int64)
    class_ids = get_array(response.outputs[1], np.int64)
    confs = get_array(response.outputs[2], np.float)
    for conf, bbox, class_id in zip(confs[0], bboxes[0], class_ids[0]):
        print(conf, class_id, bbox) 
        if class_id == 0:
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)

            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(img, f'{id2class[class_id]}: {conf:.2f}', (bbox[0]+2, bbox[1]-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)

    print("PASS")
