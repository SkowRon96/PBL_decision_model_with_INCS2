#!/usr/bin/env python3

from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import numpy as np
import logging as log
import pandas as pd
from openvino.inference_engine import IECore
#from ImageProcessor import ImageProcessor

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.", required=True,
                      type=str)
    args.add_argument("-i", "--input", help="Required. Path to a folder with images or path to an image files",
                      required=True,
                      type=str, nargs="+")
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. "
                           "MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the"
                           " kernels implementations.", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL, MYRIAD or HETERO: is "
                           "acceptable. The sample will look for a suitable plugin for device specified. Default "
                           "value is CPU",
                      default="CPU", type=str)
    args.add_argument("--labels", help="Optional. Path to a labels mapping file", default=None, type=str)
    args.add_argument("-nt", "--number_top", help="Optional. Number of top results", default=10, type=int)

    return parser

def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    # Plugin initialization for specified device and load extensions library if specified
    log.info("Creating Inference Engine")
    ie = IECore()
    if args.cpu_extension and 'CPU' in args.device:
        ie.add_extension(args.cpu_extension, "CPU")
    # Read IR
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = ie.read_network(model=model_xml, weights=model_bin)

    assert len(net.input_info.keys()) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"

    log.info("Preparing input blobs")
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))
    net.batch_size = len(args.input)

  # Read and pre-process input images
    n, c = net.input_info[input_blob].input_data.shape
    images = np.ndarray(shape=(n, c))
    dataset = pd.read_csv("C:/PBL/MODEL-PIERWSZEPODEJSCIE/needs1.csv", delimiter=",")
    arr = dataset.to_numpy()
    arr_size = int(arr.size/5)

    # Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=args.device)

    for j in range(arr_size):
        for i in range(n):
        #image = args.input[i]
        #images[i] = np.array([[0.586337,0.734454,0.668948,0.783575,0.578313]])
            images[i]=np.array([arr[j,:]])
        #processor = ImageProcessor()
        #image, cropped = processor.preprocess_image(image)
        #if image.shape[:-1] != (h, w):
        #    log.warning("Image {} is resized from {} to {}".format(args.input[i], image.shape[:-1], (h, w)))
        #    image = cv2.resize(image, (w, h))
        #image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        #images[i] = image
        log.info("Batch size is {}".format(n))



    # Start sync inference
        log.info("Starting inference in synchronous mode")
        res = exec_net.infer(inputs={input_blob: images})

    # Processing output blob
        log.info("Processing output blob")
        res = res[out_blob]
        print("INPUT \r\n",np.array([arr[j, :]]))
        print("NCS \r\n", res, '\r\nPredicted:', res.argmax())


    log.info("End\n")

if __name__ == '__main__':
    sys.exit(main() or 0)
#
#
# # [NCSDK2 API](https://movidius.github.io/ncsdk/ncapi/ncapi2/py_api/readme.html)
# #from openvino.inference_engine import IECore
# #from openvino.inference_engine import IENetwork
# #from mvnc import mvncapi as mvnc
# import numpy
# import cv2
# from ImageProcessor import ImageProcessor
#
#
# def load_to_IE(model):
#     # Loading the Inference Engine API
#     ie = IECore()
#     # Loading IR files
#     net = IENetwork(model=model + ".xml", weights=model + ".bin")
#     # Loading the network to the inference engine
#     exec_net = ie.load_network(network=net, device_name="CPU")
#     return exec_net
#
# def do_inference(exec_net, image):
#     input_blob = next(iter(exec_net.inputs))
#     return exec_net.infer({input_blob: image})
#
#
# processor = ImageProcessor()
# test_image = './data/photo_6.jpg'
# input_image = cv2.imread(test_image)
# cropped_input, cropped = processor.preprocess_image(input_image)
#
#
# model = load_to_IE('C:/PBL/keras_mnist-master/IR/tf_model')
#
#
#
#
# # Using NCS Predict
# # set the logging level for the NC API
# # mvnc.global_set_option(mvnc.GlobalOption.RW_LOG_LEVEL, 0)
#
# # get a list of names for all the devices plugged into the system
# ie = IECore()
# #devices = mvnc.enumerate_devices()
# #if len(devices) == 0:
# #    print('No devices found')
# #    quit()
#
# # get the first NCS device by its name.  For this program we will always open the first NCS device.
# #dev = mvnc.Device(devices[0])
#
# # try to open the device.  this will throw an exception if someone else has it open already
# #try:
# #    dev.open()
# #except:
# #    print("Error - Could not open NCS device.")
# #    quit()
#
# # Read a compiled network graph from file (set the graph_filepath correctly for your graph file)
# #with open("graph", mode='rb') as f:
# #    graphFileBuff = f.read()
#
# #graph = mvnc.Graph('graph1')
#
# # Allocate the graph on the device and create input and output Fifos
# #in_fifo, out_fifo = graph.allocate_with_fifos(dev, graphFileBuff)
#
# # Write the input to the input_fifo buffer and queue an inference in one call
# #graph.queue_inference_with_fifo_elem(in_fifo, out_fifo, cropped_input.astype('float32'), 'user object')
#
# # Read the result to the output Fifo
# output, userobj = out_fifo.read_elem()
#
# # Deallocate and destroy the fifo and graph handles, close the device, and destroy the device handle
# try:
#     in_fifo.destroy()
#     out_fifo.destroy()
#     graph.destroy()
#     dev.close()
#     dev.destroy()
# except:
#     print("Error - could not close/destroy Graph/NCS device.")
#     quit()
#
# print("NCS \r\n", output, '\r\nPredicted:',output.argmax())