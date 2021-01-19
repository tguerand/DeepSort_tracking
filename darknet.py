from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
from util import *


def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_

def parse_cfg(cfgfile):
    """
    Takes a configuration file
    Returns a list of dictionaries of attributes 
    """
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')                        
    lines = [x for x in lines if len(x) > 0]   #remove empty lines           
    lines = [x for x in lines if x[0] != '#']     # remove comments         
    lines = [x.rstrip().lstrip() for x in lines]  
    dicc = {}
    diccs = []

    for line in lines:
        if line[0] == "[":               
            if len(dicc) != 0:          
                diccs.append(dicc)     
                dicc = {}               
            dicc["type"] = line[1:-1].rstrip()     
        else:
            key,value = line.split("=") 
            dicc[key.rstrip()] = value.lstrip()
    diccs.append(dicc)

    return diccs
##################################### Define layers #################################
def create_modules(blocks):
    '''
    create the layers in NN architecture
    take list of dictionaries and return a nn.ModuleList of YOLO modules
    '''
    # the first dic is a parameter dict
    net_info = blocks[0]     
    
    module_list = nn.ModuleList()
    prev_filters = 3 # RGB image has 3 filters 
    output_filters = []
    
    for i in range(1, len(blocks)):
        
        #print('len output_filters : ', len(output_filters))  
        block = blocks[i]
        i = i-1
        module = nn.Sequential()
        # we have these types of blocks {'convolutional', 'route', 'shortcut', 'upsample', 'yolo'}
        ############################# Convolutional #################################
        if block['type'] == 'convolutional':
            # 75 layers
            # keys :'type', 'batch_normalize', 'filters', 'size', 'stride', 'pad', 'activation'
            # Each block can be constructed of one : conv, or two layers : conv + batch_norm 
            # The activation function of the layer
            activation = block["activation"]
            # Normalization
            try:
                batch_normalize = int(block["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            
            filters= int(block["filters"])
            padding = int(block["pad"])
            kernel_size = int(block["size"])
            stride = int(block["stride"])
            
            # Padding
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            #Add the convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(i), conv)

            #Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(i), bn)

            #Activation is either Linear or a Leaky ReLU 
            # if linear we do nothing
            # if leaky relu : 
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(i), activn)

        ############################# Upsample #################################
        elif (block["type"] == "upsample"):
            # 2 blocks of type 'upsample'
            # keys : 'type', 'stride'
            stride = int(block["stride"])
            upsample = nn.Upsample(scale_factor = 2, mode = "bilinear")
            module.add_module("upsample_{}".format(i), upsample)
            
        ############################# Route #####################################
        elif (block["type"] == "route"):
            # 4 blocks of type 'route' in index [84, 87, 96, 99]
            # keys :'type', 'layers'
         
            layers = block['layers']
            
            # layers can have one or two values separated by ','
            l = [int(x) for x in layers.split(',')]
            
            s = l[0]
            # s = -4 or -1 all s are negatives we will take i + s output
            # i in [84, 87, 96, 99]
            if len(l) > 1 :
                e = l[1]
            

            route = EmptyLayer()
            module.add_module("route_{0}".format(i), route)
            
            if len(l) > 1:
                
                filters = output_filters[s+i] + output_filters[e]
            else:
                filters= output_filters[s+i]
   
        ############################# Shortcut ##################################
        elif (block["type"] == "shortcut"):
            # skip connection
            # 23 blocks of type 'shortcut'
            # keys : 'type', 'from', 'activation'
            
            # Activation is all time linear            
            # From is all time = '-3' : add features from the previous and 3rd layer backwards
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(i), shortcut)
   
            
        
        ############################# YOLO ##################################
        elif (block["type"] == "yolo"):
            # 3 blocks of type 'yolo'
            # keys : 'type', 'mask', 'anchors', 'classes', 'num', 'jitter', 'ignore_thresh', 'truth_thresh', 'random'
            
            mask = block["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = block["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[j], anchors[j+1]) for j in range(0, len(anchors),2)]
            anchors = [anchors[j] for j in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(i), detection)
            
            
        ############################ Actualisation ##################################    
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
        
    return (net_info, module_list)
            
        
class EmptyLayer(nn.Module):
    '''
    define an empty layer
    '''
    def __init__(self):
        super(EmptyLayer, self).__init__()    
        
class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
        
##################################### Define Network#################################
class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        # compute the dict containing the architecture
        self.blocks = parse_cfg(cfgfile)
        # parameters and layers of NN   
        self.net_info, self.module_list = create_modules(self.blocks)
        
    def forward(self, x, CUDA = True):
        layers = self.blocks[1:] # all blocks except the parameters block
        outputs = {} 
        
        write = 0     
        for i, layer in enumerate(layers):        
            typ = (layer["type"])
            ############################ Conv | upsample############################
            if typ == "convolutional" or typ == "upsample":
                x = self.module_list[i](x)
                
            ############################ Route ####################################   
            elif typ =="route" :
                # if len(layers) = 1 output is output of layer i+s
                # if len(layers) = 2 output is concatenation of the outputs of layers i+s and e
            
                l = [int(x) for x in layer['layers'].split(',')]
                s = l[0]
                if len(l) ==1:
                    # s in our case is always <0
                    x = outputs[i+s]
                    
                else :
                    e = l[1]
                    # e is always >0, either 36 or 61
                    x1 = outputs[i+s]
                    x2 = outputs[e]
                    x = torch.cat((x1,x2),1)
                    
                    
            ########################### Shortcut #################################### 
            elif typ =='shortcut':
                #add features from the previous and 3rd layer backwards (from =-3 always)
                f = int(layer['from'])
                x = outputs[i-1]+ outputs[i+f]
                
           ########################### YOLO #################################### 
            elif typ =='yolo':
                anchors = self.module_list[i][0].anchors
                
                inp_dim = int (self.net_info["height"])

                num_classes = int (layer["classes"])

                #Transform 
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:              
                    detections = x
                    write = 1

                else:       
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x 
        return detections
                    
    ############# Download pretrained weights #################################
    def load_weights(self, weightfile):
        fp =  open(weightfile, "rb")
        #The first 5 values are header information 
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        
        weights = np.fromfile(fp, dtype = np.float32)
        i_weights = 0
        
             
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
            keys = self.blocks[i + 1].keys()
            
            #the weights belong to only convolutional layer   
            if module_type =='convolutional':
                model = self.module_list[i]
                conv = model[0]
                #When the batch norm layer appears in a convolutional block, there are no biases. However, when there's no batch norm layer, bias "weights" have to read from the file.
                if 'batch_normalize' in keys:
                    bn = model[1]

                    num_bn_biases = bn.bias.numel()

                    bn_biases = torch.from_numpy(weights[i_weights:i_weights + num_bn_biases])
                    i_weights += num_bn_biases

                    bn_weights = torch.from_numpy(weights[i_weights: i_weights + num_bn_biases])
                    i_weights  += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[i_weights: i_weights + num_bn_biases])
                    i_weights  += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[i_weights: i_weights + num_bn_biases])
                    i_weights  += num_bn_biases

                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                else:
                    num_biases = conv.bias.numel()

                    conv_biases = torch.from_numpy(weights[i_weights: i_weights + num_biases])
                    i_weights = i_weights + num_biases

                    conv_biases = conv_biases.view_as(conv.bias.data)

                    conv.bias.data.copy_(conv_biases)
                    
                num_weights = conv.weight.numel()

                conv_weights = torch.from_numpy(weights[i_weights:i_weights +num_weights])
                i_weights +=  num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)

        
        
        
        