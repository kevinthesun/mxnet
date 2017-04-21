
# NDArray Tutorial


One of the main object in MXNet is the multidimensional array provided by the package `mxnet.ndarray`, or `mxnet.nd` for short. If you familiar with the scientific computing python package [NumPy](http://www.numpy.org/), `mxnet.ndarray` is similar to `numpy.ndarray` in many aspects. 

## The basic

A multidimensional array is a table of numbers with the same type. For example, the coordinates of a point in 3D space `[1, 2, 3]` is a 1-dimensional array with that dimension has a length of 3. The following picture shows a 2-dimensional array. The length of the first dimension is 2, and the second dimension has a length of 3
```
[[0, 1, 2]
 [3, 4, 5]]
```
The array class is called `NDArray`. Some important attributes of a `NDArray` object are:

- **ndarray.shape** the dimensions of the array. It is a tuple of integers indicating the length of the array in each dimension. For a matrix with `n` rows and `m` columns, the `shape` will be `(n, m)`.  
- **ndarray.dtype** an `numpy` object describing the type of the elements.
- **ndarray.size** the total number of numbers in the array, which equals to the product of the elements of `shape`
- **ndarray.context** the device this array is stored. A device can be the CPU or the i-th GPU.

### Array Creation 
An array can be created in multiple ways. For example, we can create an array from a regular Python list or tuple by using the `array` function


```python
import mxnet as mx
# create a 1-dimensional array with a python list
a = mx.nd.array([1,2,3])
# create a 2-dimensional array with a nested python list 
b = mx.nd.array([[1,2,3], [2,3,4]])
{'a.shape':a.shape, 'b.shape':b.shape}
```




    {'a.shape': (3L,), 'b.shape': (2L, 3L)}



or from an `numpy.ndarray` object


```python
import numpy as np
import math
c = np.arange(15).reshape(3,5)
# create a 2-dimensional array from a numpy.ndarray object
a = mx.nd.array(c)
{'a.shape':a.shape}
```




    {'a.shape': (3L, 5L)}



We can specify the element type with the option `dtype`, which accepts a numpy type. In default, `float32` is used. 


```python
# float32 is used in deafult
a = mx.nd.array([1,2,3])
# create an int32 array
b = mx.nd.array([1,2,3], dtype=np.int32)
# create a 16-bit float array
c = mx.nd.array([1.2, 2.3], dtype=np.float16)
(a.dtype, b.dtype, c.dtype)
```




    (numpy.float32, numpy.int32, numpy.float16)



If we only know the size but not the element values, there are several functions to create arrays with initial placeholder content. 


```python
# create a 2-dimensional array full of zeros with shape (2,3) 
a = mx.nd.zeros((2,3))
# create a same shape array full of ones
b = mx.nd.ones((2,3))
# create a same shape array with all elements set to 7
c = mx.nd.full((2,3), 7)
# create a same shape whose initial content is random and 
# depends on the state of the memory
d = mx.nd.empty((2,3))
```

### Printing Arrays
We often first convert `NDArray` to `numpy.ndarray` by the function `asnumpy` for printing. Numpy uses the following layout:
- the last axis is printed from left to right,
- the second-to-last is printed from top to bottom,
- the rest are also printed from top to bottom, with each slice separated from the next by an empty line.


```python
b = mx.nd.ones((2,3))
b.asnumpy()
```




    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]], dtype=float32)



### Basic Operations
Arithmetic operators on arrays apply *elementwise*. A new array is created and filled with the result.


```python
a = mx.nd.ones((2,3))
b = mx.nd.ones((2,3))
# elementwise plus
c = a + b
# elementwise minus
d = - c 
# elementwise pow and sin, and then transpose
e = mx.nd.sin(c**2).T
# elementwise max
f = mx.nd.maximum(a, c)  
f.asnumpy()
```




    array([[ 2.,  2.,  2.],
           [ 2.,  2.,  2.]], dtype=float32)



Simiar to `NumPy`, `*` is used for elementwise multiply, while matrix-matrix multiplication is left for `dot`


```python
a = mx.nd.ones((2,2))
b = a * a
c = mx.nd.dot(a,a)
{'b':b.asnumpy(), 'c':c.asnumpy()}
```




    {'b': array([[ 1.,  1.],
            [ 1.,  1.]], dtype=float32), 'c': array([[ 2.,  2.],
            [ 2.,  2.]], dtype=float32)}



The assignment operators such as `+=` and `*=` act in place to modify an existing array rather than create a new one.


```python
a = mx.nd.ones((2,2))
b = mx.nd.ones(a.shape)
b += a
b.asnumpy()
```




    array([[ 2.,  2.],
           [ 2.,  2.]], dtype=float32)



### Indexing and Slicing
The slice operator `[]` applies on axis 0. 


```python
a = mx.nd.array(np.arange(6).reshape(3,2))
a[1:2] = 1
a[:].asnumpy()
```




    array([[ 0.,  1.],
           [ 1.,  1.],
           [ 4.,  5.]], dtype=float32)



We can also slice a particular axis with the method `slice_axis`


```python
d = mx.nd.slice_axis(a, axis=1, begin=1, end=2)
d.asnumpy()
```




    array([[ 1.],
           [ 1.],
           [ 5.]], dtype=float32)



### Shape Manipulation 
The shape of the array can be changed as long as the size remaining the same 


```python
a = mx.nd.array(np.arange(24))
b = a.reshape((2,3,4))
b.asnumpy()
```




    array([[[  0.,   1.,   2.,   3.],
            [  4.,   5.,   6.,   7.],
            [  8.,   9.,  10.,  11.]],
    
           [[ 12.,  13.,  14.,  15.],
            [ 16.,  17.,  18.,  19.],
            [ 20.,  21.,  22.,  23.]]], dtype=float32)



Method `concatenate` stacks multiple arrays along the first dimension. (Their shapes must be the same).


```python
a = mx.nd.ones((2,3))
b = mx.nd.ones((2,3))*2
c = mx.nd.concatenate([a,b])
c.asnumpy()
```




    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.],
           [ 2.,  2.,  2.],
           [ 2.,  2.,  2.]], dtype=float32)



### Reduce

We can reduce the array to a scalar


```python
a = mx.nd.ones((2,3))
b = mx.nd.sum(a)
b.asnumpy()
```




    array([ 6.], dtype=float32)



or along a particular axis


```python
c = mx.nd.sum_axis(a, axis=1)
c.asnumpy()
```




    array([ 3.,  3.], dtype=float32)



### Broadcast
We can also broadcast an array by duplicating. The following codes broadcast along axis 1


```python
a = mx.nd.array(np.arange(6).reshape(6,1))
b = a.broadcast_to((6,2))  # 
b.asnumpy()
```




    array([[ 0.,  0.],
           [ 1.,  1.],
           [ 2.,  2.],
           [ 3.,  3.],
           [ 4.,  4.],
           [ 5.,  5.]], dtype=float32)



or broadcast along axes 1 and 2


```python
c = a.reshape((2,1,1,3))
d = c.broadcast_to((2,2,2,3))
d.asnumpy()
```




    array([[[[ 0.,  1.,  2.],
             [ 0.,  1.,  2.]],
    
            [[ 0.,  1.,  2.],
             [ 0.,  1.,  2.]]],
    
    
           [[[ 3.,  4.,  5.],
             [ 3.,  4.,  5.]],
    
            [[ 3.,  4.,  5.],
             [ 3.,  4.,  5.]]]], dtype=float32)



Broadcast can be applied to operations such as `*` and `+`. 


```python
a = mx.nd.ones((3,2))
b = mx.nd.ones((1,2))
c = a + b
c.asnumpy()
```




    array([[ 2.,  2.],
           [ 2.,  2.],
           [ 2.,  2.]], dtype=float32)



### Copies
Data is *NOT* copied in normal assignment. 


```python
a = mx.nd.ones((2,2))
b = a  
b is a
```




    True



similar for function arguments passing.


```python
def f(x):  
    return x
a is f(a)
```




    True



The `copy` method makes a deep copy of the array and its data


```python
b = a.copy()
b is a
```




    False



The above code allocate a new NDArray and then assign to *b*. We can use the `copyto` method or the slice operator `[]` to avoid additional memory allocation


```python
b = mx.nd.ones(a.shape)
c = b
c[:] = a
d = b
a.copyto(d)
(c is b, d is b)
```




    (True, True)



## The Advanced 
There are some advanced features in `mxnet.ndarray` which make mxnet different from other libraries. 

### GPU Support

In default operators are executed on CPU. It is easy to switch to another computation resource, such as GPU, if available. The device information is stored in `ndarray.context`. When MXNet is compiled with flag `USE_CUDA=1` and there is at least one Nvidia GPU card, we can make all computations run on GPU 0 by using context `mx.gpu(0)`, or simply `mx.gpu()`. If there are more than two GPUs, the 2nd GPU is represented by `mx.gpu(1)`.


```python
def f():
    a = mx.nd.ones((100,100))
    b = mx.nd.ones((100,100))
    c = a + b
    print(c)
# in default mx.cpu() is used
f()  
# change the default context to the first GPU
with mx.Context(mx.gpu()):  
    f()
```

    <NDArray 100x100 @cpu(0)>
    <NDArray 100x100 @gpu(0)>


We can also explicitly specify the context when creating an array


```python
a = mx.nd.ones((100, 100), mx.gpu(0))
a
```




    <NDArray 100x100 @gpu(0)>



Currently MXNet requires two arrays to sit on the same device for computation. There are several methods for copying data between devices.


```python
a = mx.nd.ones((100,100), mx.cpu())
b = mx.nd.ones((100,100), mx.gpu())
c = mx.nd.ones((100,100), mx.gpu())
a.copyto(c)  # copy from CPU to GPU
d = b + c
e = b.as_in_context(c.context) + c  # same to above
{'d':d, 'e':e}
```




    {'d': <NDArray 100x100 @gpu(0)>, 'e': <NDArray 100x100 @gpu(0)>}



### Serialize From/To (Distributed) Filesystems  
There are two ways to save data to (load from) disks easily. The first way uses `pickle`. `NDArray` is pickle compatible.


```python
import pickle as pkl
a = mx.nd.ones((2, 3))
# pack and then dump into disk
data = pkl.dumps(a)
pkl.dump(data, open('tmp.pickle', 'wb'))
# load from disk and then unpack 
data = pkl.load(open('tmp.pickle', 'rb'))
b = pkl.loads(data)
b.asnumpy()
```




    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]], dtype=float32)



The second way is to directly dump into disk in binary format by method `save` and `load`. Besides single NDArray, we can load/save a list


```python
a = mx.nd.ones((2,3))
b = mx.nd.ones((5,6))               
mx.nd.save("temp.ndarray", [a,b])
c = mx.nd.load("temp.ndarray")
c
```




    [<NDArray 2x3 @cpu(0)>, <NDArray 5x6 @cpu(0)>]



or a dict


```python
d = {'a':a, 'b':b}
mx.nd.save("temp.ndarray", d)
c = mx.nd.load("temp.ndarray")
c
```




    {'a': <NDArray 2x3 @cpu(0)>, 'b': <NDArray 5x6 @cpu(0)>}



The load/save is better than pickle in two aspects
1. The data saved with the Python interface can be used by another lanuage binding. For example, if we save the data in python:
```python
a = mx.nd.ones((2, 3))
mx.save("temp.ndarray", [a,])
```
then we can load it into R:
```R
a <- mx.nd.load("temp.ndarray")
as.array(a[[1]])
##      [,1] [,2] [,3]
## [1,]    1    1    1
## [2,]    1    1    1
```
2. If a distributed filesystem such as Amazon S3 or Hadoop HDFS is set up, we can directly save to and load from it. 
```python
mx.nd.save('s3://mybucket/mydata.ndarray', [a,])  # if compiled with USE_S3=1
mx.nd.save('hdfs///users/myname/mydata.bin', [a,])  # if compiled with USE_HDFS=1
```


### Lazy Evaluation and Auto Parallelization *

MXNet uses lazy evaluation for better performance. When we run `a=b+1` in python, the python thread just pushs the operation into the backend engine and then returns. There are two benefits for such optimization:
1. The main python thread can continue to execute other computations once the previous one is pushed. It is useful for frontend languages with heavy overheads. 
2. It is easier for the backend engine to explore further optimization, such as auto parallelization that will be discussed shortly. 

The backend engine is able to resolve the data dependencies and schedule the computations correctly. It is transparent to frontend users. We can explicitly call the method `wait_to_read` on the result array to wait the computation finished. Operations that copy data from an array to other packages, such as `asnumpy`, will implicitly call `wait_to_read`. 



```python
# @@@ AUTOTEST_OUTPUT_IGNORED_CELL
import time

def do(x, n):
    """push computation into the backend engine"""
    return [mx.nd.dot(x,x) for i in range(n)]
def wait(x):
    """wait until all results are available"""
    for y in x:
        y.wait_to_read()
        
tic = time.time()
a = mx.nd.ones((1000,1000))
b = do(a, 50)
print('time for all computations are pushed into the backend engine:\n %f sec' % (time.time() - tic))
wait(b)
print('time for all computations are finished:\n %f sec' % (time.time() - tic))
```

    time for all computations are pushed into the backend engine:
     0.001089 sec
    time for all computations are finished:
     5.398588 sec


Besides analyzing data read and write dependencies, the backend engine is able to schedule computations with no dependency in parallel. For example, in the following codes
```python
a = mx.nd.ones((2,3))
b = a + 1
c = a + 2
d = b * c
```
the second and third sentences can be executed in parallel. The following example first run on CPU and then on GPU.


```python
# @@@ AUTOTEST_OUTPUT_IGNORED_CELL
n = 10
a = mx.nd.ones((1000,1000))
b = mx.nd.ones((6000,6000), mx.gpu())
tic = time.time()
c = do(a, n)
wait(c)
print('Time to finish the CPU workload: %f sec' % (time.time() - tic))
d = do(b, n)
wait(d)
print('Time to finish both CPU/CPU workloads: %f sec' % (time.time() - tic))
```

    Time to finish the CPU workload: 1.089354 sec
    Time to finish both CPU/CPU workloads: 2.663608 sec


Now we issue all workloads at the same time. The backend engine will try to parallel the CPU and GPU computations.


```python
# @@@ AUTOTEST_OUTPUT_IGNORED_CELL
tic = time.time()
c = do(a, n)
d = do(b, n)
wait(c)
wait(d)
print('Both as finished in: %f sec' % (time.time() - tic))
```

    Both as finished in: 1.543902 sec


## Current Status

We try our best to keep the NDArray API as the same numpy's. But it is not fully numpy compatible yet. Here we summary some major difference, which we hope to be fixed in a short time. We are also welcome to any contribution.

- Slice and Index. 
    - NDArray can only slice one dimension at each time, namely we cannot use `x[:, 1]` to slice both dimensions.
    - Only continues indexes are supported, we cannot do `x[1:2:3]`
    - boolean indices are not supported, such as `x[y==1]`.
- Lack of reduce functions such as `max`, `min`...

## Futher Readings
- [NDArray API](http://mxnet.dmlc.ml/en/latest/packages/python/ndarray.html) Documents for all NDArray methods.
- [MinPy](https://github.com/dmlc/minpy) on-going project, fully numpy compatible with GPU and auto differentiation supports 


## Next Steps
* [MXNet Tutorials Index](http://mxnet.io/tutorials/index.html)
* [MXNet Notebooks on GitHub](https://github.com/dmlc/mxnet-notebooks)
