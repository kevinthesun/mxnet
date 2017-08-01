import mxnet as mx
from mxnet.gluon.model_zoo.vision.resnet import ResNetV2, BasicBlockV2, BottleneckV2, resnet18_v2, resnet18_v1
from mxnet.gluon.block import Recorder

def get_symbol_graph(sym):
    graph_arr = []
    layers = sym.get_internals()
    for item in layers:
        if not item.get_children():
            continue
        has_param = False
        for child in item.get_children():
            if child.name.startswith(item.name):
                has_param = True
                break
        if has_param and not item.name.startswith('_'):
            graph_arr.append(item)
    return graph_arr

model_name = 'resnet-18'
batch_size = 32
channel = 3
width = 224
height = 224
ctx=[mx.cpu()]
sym, arg_params, aux_params = mx.model.load_checkpoint(model_name, 0)

model = mx.mod.Module(symbol=sym, context=ctx, data_names=('data',))
model.bind(data_shapes=[('data', (batch_size, channel, width, height))],
           label_shapes= [('softmax_label', (batch_size,))])
model.set_params(arg_params, aux_params)
graph_arr = get_symbol_graph(sym)
print("Symbolic graph:")
for layer in graph_arr:
    print(layer)
print("%d layers found." % (len(graph_arr)))

Recorder.records={}
nn_model = resnet18_v2()#ResNetV2(BottleneckV2, [2, 2, 2, 2], [64, 64, 128, 256, 512])
nn_model.collect_params().initialize(mx.init.Zero(), ctx=ctx)
size = (32, 3, 224, 224)
data = mx.nd.ones(size, ctx=mx.cpu(0))
nn_model(data)
node_list = []
#print(Recorder.records)
for item in Recorder.records['gluon']:
    if len(item._params) > 0:
        print(item._prefix)
        node_list.append(item)
print('%d layers' % (len(node_list)))
