import mxnet as mx
import random
from mxnet.io import DataBatch, DataIter
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd as ag
from mxnet.gluon.model_zoo.vision.vgg import vgg16
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Gluon VGG16 Benchmark",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num_gpu', type=int, default='16',
                        help='Number of gpus used. 0 means using cpu')
parser.add_argument('--image_shape', type=str, default='32,3,299,299',
                        help='Input image shape')
parser.add_argument('--num_classes', type=int, default='1000',
                        help='Number of image classes')
parser.add_argument('--max_iter', type=int, default='1000',
                        help='Maximum iteration for each epoch')
parser.add_argument('--num_epoch', type=int, default='10',
                        help='Number of epochs')

class SyntheticDataIter(DataIter):
    def __init__(self, num_classes, data_shape, max_iter, dtype):
        self.batch_size = data_shape[0]
        self.cur_iter = 0
        self.max_iter = max_iter
        self.dtype = dtype
        label = np.random.randint(0, num_classes, [self.batch_size,])
        data = np.random.uniform(-1, 1, data_shape)
        self.data = mx.nd.array(data, dtype=self.dtype, ctx=mx.Context('cpu_pinned', 0))
        self.label = mx.nd.array(label, dtype=self.dtype, ctx=mx.Context('cpu_pinned', 0))

    def __iter__(self):
        return self

    @property
    def provide_data(self):
        return [mx.io.DataDesc('data', self.data.shape, self.dtype)]

    @property
    def provide_label(self):
        return [mx.io.DataDesc('softmax_label', (self.batch_size,), self.dtype)]

    def next(self):
        self.cur_iter += 1
        if self.cur_iter <= self.max_iter:
            return DataBatch(data=(self.data,),
                             label=(self.label,),
                             pad=0,
                             index=None,
                             provide_data=self.provide_data,
                             provide_label=self.provide_label)
        else:
            raise StopIteration

    def __next__(self):
        return self.next()

    def reset(self):
        self.cur_iter = 0

if __name__ == '__main__':
    args = parser.parse_args()
    num_gpu = args.num_gpu
    ctx = [mx.cpu(0)] if num_gpu == 0 else [mx.gpu(i) for i in range(num_gpu)]

    num_classes = args.num_classes
    data_shape = args.image_shape
    max_iter = args.max_iter
    train_data = SyntheticDataIter(num_classes, data_shape, max_iter, dtype=np.float32)

    model = vgg16()
    model.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

    epoch = args.num_epoch
    metric = mx.metric.Accuracy()
    for i in range(epoch):
        train_data.reset()
        for batch in train_data:
            data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
            outputs = []



