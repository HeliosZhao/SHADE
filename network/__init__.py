"""
Network Initializations
"""

import logging
import importlib
import torch
import datasets



def get_net(args):
    """
    Get Network Architecture based on arguments provided
    """
    net = get_model(args=args, num_classes=datasets.num_classes)
    num_params = sum([param.nelement() for param in net.parameters()])
    logging.info('Model params = {:2.3f}M'.format(num_params / 1000000))

    net = net.cuda()
    return net


def warp_network_in_dataparallel(net, gpuid):
    """
    Wrap the network in Dataparallel
    """
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[gpuid], find_unused_parameters=True)
    return net


def get_model(args, num_classes):
    """
    Fetch Network Function Pointer
    """
    network = args.arch
    module = network[:network.rfind('.')]
    model = network[network.rfind('.') + 1:]
    mod = importlib.import_module(module)
    net_func = getattr(mod, model)
    net = net_func(args=args, num_classes=num_classes)
    return net
