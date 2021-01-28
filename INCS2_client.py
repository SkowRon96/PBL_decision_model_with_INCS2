from argparse import ArgumentParser, SUPPRESS
from openvino.inference_engine import IECore
from io import StringIO
import sys
import os
import socket
import numpy as np
import logging as log
import pandas as pd
import datetime
import random as r

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.", required=True,
                      type=str)
    args.add_argument("-i", "--input", help="Optional. Path to a folder with images or path to an image files",
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
    args.add_argument("-ip", "--ip_server", help="Optional. IP adress of server", default='127.0.0.1', type=str)
    args.add_argument("-r", "--rrandom", help="Optional. Random decision making", default='NO', type=str)
    args.add_argument("-port", "--port_server", help="Optional. Port of server", default=65432, type=int)
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
    #net.batch_size = len(args.input)

    # Read and pre-process input inputs
    n, c = net.input_info[input_blob].input_data.shape

    # Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=args.device)

    HOST = args.ip_server  # The server's hostname or IP address
    PORT = args.port_server  # The port used by the server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        while True:
            try:
                s.connect((HOST, PORT))
                log.info("Connected to TCP\IP server")
                break
            except Exception as e:
                log.info("Something's wrong with %s:%d. Exception is %s" % (HOST, PORT, e))

        while True:

            start_time = datetime.datetime.now()
            commend_1='GN;;'
            s.send(commend_1.encode())
            log.info("Waiting for needs....")
            data = s.recv(8192)
            #print(data)

            StringData = StringIO('Food,Water,Dream,Sex,Toilet,High\n'+data.decode())
            inputs = np.ndarray(shape=(n, c))
            dataset = pd.read_csv(StringData, delimiter=",")
            #print(dataset)
            arr = dataset.to_numpy()
            arr_size = int(arr.size/c)
            outputs = []
            for j in range(arr_size):
                for i in range(n):
                    inputs[i]=np.array([arr[j,:]])
                log.info("Batch size is {}".format(n))

            # Start sync inference
                log.info("Starting inference in synchronous mode")
                res = exec_net.infer(inputs={input_blob: inputs})

            # Processing output blob
                log.info("Processing output blob")
                res = res[out_blob]
                print("INPUT \r\n",np.array([arr[j, :]]))
                print("NCS \r\n", res, '\r\nPredicted:', res.argmax())

                if 'YES' in args.rrandom:
                    outputs.append(r.randint(0, 5))
                else:
                    outputs.append(res.argmax())

            str1 = ','.join(str(e) for e in outputs)
            commend_2 = 'SPOL;'+str1+';'
            s.send(commend_2.encode())
            log.info("Sending decisions to server")
            end_time = datetime.datetime.now()
            time_diff = (end_time - start_time)

            execution_time = time_diff.total_seconds() * 1000

            print("Execution time [ms]: ", execution_time)
    log.info("End\n")

if __name__ == '__main__':
    sys.exit(main() or 0)

