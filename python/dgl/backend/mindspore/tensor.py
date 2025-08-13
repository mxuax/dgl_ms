# dgl/backend/mindspore/tensor.py
# 核心修改：基于 mxnet/tensor.py 的模板，为 MindSpore 框架重写了所有张量操作。
from __future__ import absolute_import

import builtins
import numbers

import mindspore as ms
import mindspore.ops as ops
import numpy as np

from ... import ndarray as dglnd
from ...function.base import TargetCode

# --- 数据类型定义 ---
# 核心修改：将数据类型字符串映射到 MindSpore 的数据类型对象。
def data_type_dict():
    return {
        "float16": ms.float16,
        "float32": ms.float32,
        "float64": ms.float64,
        "uint8": ms.uint8,
        "int8": ms.int8,
        "int16": ms.int16,
        "int32": ms.int32,
        "int64": ms.int64,
        "bool": ms.bool_,
    }

# --- 设备上下文 ---
# 核心修改：MindSpore 使用全局上下文管理设备，因此函数返回代表设备的字符串。
def cpu():
    return "CPU"

# --- 张量创建与转换 ---
def tensor(data, dtype=None):
    if isinstance(data, ms.Tensor):
        if dtype is None or data.dtype == dtype:
            return data
        else:
            return ops.cast(data, dtype)
    else:
        # MindSpore 的 Tensor 构造函数可以很好地处理各种输入类型
        return ms.Tensor(data, dtype=dtype)

def as_scalar(data):
    if data.size != 1:
        raise ValueError("The current array is not a scalar")
    return data.asnumpy().item()

def get_preferred_sparse_format():
    """获取后端首选的稀疏矩阵格式。"""
    # 核心修改：根据要求，不实现稀疏功能。
    raise NotImplementedError("MindSpore backend does not support sparse matrix yet.")

def sparse_matrix(data, index, shape, force_format=False):
    # 核心修改：根据要求，不实现稀疏功能。
    raise NotImplementedError("MindSpore backend does not support sparse matrix yet.")

def sparse_matrix_indices(spmat):
    # 核心修改：根据要求，不实现稀疏功能。
    raise NotImplementedError("MindSpore backend does not support sparse matrix yet.")

def is_tensor(obj):
    return isinstance(obj, ms.Tensor)

def shape(input):
    return input.shape

def dtype(input):
    return input.dtype

def ndim(input):
    return input.ndim

def context(input):
    # MindSpore 的 Tensor 对象有 device 属性
    return input.device

def device_type(ctx_str):
    # ctx_str 预期为 "CPU", "GPU", "Ascend"
    return ctx_str.lower()

def device_id(ctx):
    # MindSpore 在全局上下文中获取 device_id
    return ms.get_context("device_id")

def to_backend_ctx(dglctx):
    # DGL context: 1 for CPU, 2 for GPU, 3 for Ascend (假设)
    dev_type = dglctx.device_type
    if dev_type == 1:
        return "CPU"
    elif dev_type == 2: # GPU
        return "GPU"
    elif dev_type == 3: # Ascend
        return "Ascend"
    else:
        raise ValueError("Unsupported DGL device context:", dglctx)

def astype(input, ty):
    return ops.cast(input, ty)

def asnumpy(input):
    return input.asnumpy()

def copy_to(input, ctx, **kwargs):
    # 核心修改：MindSpore 张量设备在创建时确定。
    # 切换设备需要重新创建张量。在全局上下文中，此操作通常是无意义的。
    # 假设目标 ctx 与当前全局上下文一致，因此返回原张量。
    return input

def is_pinned(input):
    # MindSpore 没有直接对应的 "pinned memory" 概念的公共 API
    return False

# --- 数学运算 ---
def sum(input, dim, keepdims=False):
    return ops.sum(input, axis=dim, keepdims=keepdims)

def floor_div(in1, in2):
    return ops.floor_div(in1, in2)

def reduce_sum(input):
    return input.sum()

def cumsum(input, dim):
    return ops.cumsum(input, axis=dim)

def mean(input, dim):
    return ops.mean(input, axis=dim)

def reduce_mean(input):
    return input.mean()

def max(input, dim):
    # ops.max 返回 (values, indices)，我们只取 values
    return ops.max(input, axis=dim)[0]

def reduce_max(input):
    return input.max()

def min(input, dim):
    # ops.min 返回 (values, indices)，我们只取 values
    return ops.min(input, axis=dim)[0]

def reduce_min(input):
    return input.min()

def topk(input, k, dim, descending=True):
    # MindSpore 的 topk 默认就是 descending
    if not descending:
        # 通过取反实现 ascending topk
        return -ops.top_k(-input, k, sorted=True)[0]
    return ops.top_k(input, k, sorted=True)[0]

def argtopk(input, k, dim, descending=True):
    # MindSpore 的 topk 默认就是 descending
    if not descending:
        return ops.top_k(-input, k, sorted=True)[1]
    return ops.top_k(input, k, sorted=True)[1]

def argsort(input, dim, descending):
    return ops.argsort(input, axis=dim, descending=descending)

def exp(input):
    return ops.exp(input)

def inverse(input):
    return ops.inv(input)

def sqrt(input):
    return ops.sqrt(input)

def softmax(input, dim=-1):
    return ops.softmax(input, axis=dim)

def cat(seq, dim):
    return ops.cat(seq, axis=dim)

def stack(seq, dim):
    return ops.stack(seq, axis=dim)

def split(x, sizes_or_sections, dim):
    return ops.split(x, split_size_or_sections=sizes_or_sections, axis=dim)

def repeat(input, repeats, dim):
    # 核心修改：MindSpore 的 ops.repeat_elements 是最接近的等价物
    return ops.repeat_elements(input, repeats, axis=dim)

def gather_row(data, row_index):
    # 使用 ops.gather 实现，axis=0 表示按行收集
    return ops.gather(data, tensor(row_index, ms.int32), 0)

def slice_axis(data, axis, begin, end):
    # MindSpore 不支持动态切片，但可以用 ops.slice 实现
    start_indices = [0] * data.ndim
    slice_sizes = list(data.shape)
    start_indices[axis] = begin
    slice_sizes[axis] = end - begin
    return ops.slice(data, tuple(start_indices), tuple(slice_sizes))

def take(data, indices, dim):
    return ops.gather(data, indices, dim)

def narrow_row(data, start, stop):
    return data[start:stop]

def index_add_inplace(data, row_idx, value):
    # MindSpore 张量是不可变的，不支持 inplace 操作
    raise NotImplementedError("MindSpore does not support inplace index_add.")

def scatter_row(data, row_index, value):
    # 使用 TensorScatterUpdate 实现 scatter
    return ops.tensor_scatter_update(data, tensor(row_index, ms.int32).reshape(-1, 1), value)

def scatter_row_inplace(data, row_index, value):
    raise NotImplementedError("MindSpore does not support inplace scatter_row.")

def squeeze(input, dim):
    return ops.squeeze(input, axis=dim)

def unsqueeze(input, dim):
    return ops.expand_dims(input, dim)

def reshape(input, shape):
    return ops.reshape(input, shape)

def swapaxes(input, axis1, axis2):
    axes = list(range(input.ndim))
    axes[axis1], axes[axis2] = axes[axis2], axes[axis1]
    return ops.transpose(input, tuple(axes))

def empty(shape, dtype, ctx):
    # ctx 在 MindSpore 中是全局的，此处忽略
    return ops.empty(shape, dtype)

def zeros(shape, dtype, ctx):
    return ops.zeros(shape, dtype)

def zeros_like(input):
    return ops.zeros_like(input)

def ones(shape, dtype, ctx):
    return ops.ones(shape, dtype)

def uniform(shape, dtype, ctx, low, high):
    return ops.uniform(shape, tensor(low), tensor(high), dtype=dtype)

def randint(shape, dtype, ctx, low, high):
    return ops.randint(low, high, shape)

def pad_packed_tensor(input, lengths, value, l_min=None):
    # 核心修改：使用 MindSpore op 重写此复杂函数
    if isinstance(lengths, ms.Tensor):
        lengths = list(lengths.asnumpy())
    max_len = builtins.max(lengths)

    if l_min is not None:
        max_len = builtins.max(max_len, l_min)
    
    batch_size = len(lengths)
    out_tensor = ops.full((batch_size, max_len) + input.shape[1:], value, dtype=input.dtype)

    for i, l in enumerate(lengths):
        if l > 0:
            # 使用切片和赋值来构建
            out_tensor[i, :l] = input[:l]
        input = input[l:]
    return out_tensor

def pack_padded_tensor(input, lengths):
    if isinstance(lengths, ms.Tensor):
        lengths = lengths.asnumpy()
    
    batch_size, max_len = input.shape[:2]
    
    indices = []
    for i, l in enumerate(lengths):
        indices.extend(range(i * max_len, i * max_len + l))
    
    flat_input = reshape(input, (batch_size * max_len, -1))
    return gather_row(flat_input, tensor(indices, ms.int32))

def boolean_mask(input, mask):
    return ops.boolean_mask(input, mask)

def equal(x, y):
    return ops.equal(x, y)

def allclose(x, y, rtol=1e-4, atol=1e-4):
    return ops.allclose(x, y, rtol=rtol, atol=atol)

def logical_not(input):
    return ops.logical_not(input)

def logical_and(input1, input2):
    return ops.logical_and(input1, input2)

def clone(input):
    return input.copy()

def clamp(data, min_val, max_val):
    return ops.clamp(data, min_val, max_val)

def replace_inf_with_zero(x):
    return ops.where(ops.isinf(x), ops.zeros_like(x), x)

def count_nonzero(input):
    return ops.count_nonzero(input)

def unique(input, return_inverse=False, return_counts=False):
    # MindSpore 的 unique 可以返回所有需要的值
    return ops.unique(input, return_inverse=return_inverse, return_counts=return_counts)

def full_1d(length, fill_value, dtype, ctx):
    return ops.full((length,), fill_value, dtype=dtype)

def nonzero_1d(input):
    return ops.nonzero(input).squeeze()

def sort_1d(input):
    return ops.sort(input)

def arange(start, stop, dtype=ms.int64, ctx=None):
    return ops.arange(start, stop, 1, dtype=dtype)

def rand_shuffle(arr):
    # MindSpore 没有直接的 shuffle op, 使用 numpy 实现
    idx = np.arange(arr.shape[0])
    np.random.shuffle(idx)
    return arr[tensor(idx)]

# --- DLPack 转换 ---
def zerocopy_to_dlpack(arr):
    return arr.to_dlpack()

def zerocopy_from_dlpack(dlpack_arr):
    return ms.Tensor.from_dlpack(dlpack_arr)

# --- Numpy 转换 ---
def zerocopy_to_numpy(arr):
    # MindSpore 的 asnumpy() 是拷贝操作
    return arr.asnumpy()

def zerocopy_from_numpy(np_data):
    # MindSpore 的 from_numpy() 是拷贝操作
    return ms.Tensor.from_numpy(np_data)

# --- DGL NDArray 转换 ---
def zerocopy_to_dgl_ndarray(arr):
    return dglnd.from_dlpack(arr.to_dlpack())

def zerocopy_to_dgl_ndarray_for_write(arr):
    # MindSpore to_dlpack 是只读的，没有专门的写模式
    return dglnd.from_dlpack(arr.to_dlpack())

def zerocopy_from_dgl_ndarray(arr):
    return ms.Tensor.from_dlpack(arr.to_dlpack())

def sync():
    # MindSpore 是图模式执行，默认是同步的。此函数为空操作。
    pass

# --- 自动微分 ---
# 核心修改：MindSpore 的梯度机制与 PyTorch/MXNet 不同。
# 它不依赖于张量上的 .grad 属性或 .backward() 方法。
# 而是使用 ops.grad() 函数转换。因此，直接映射是不可能的。
def attach_grad(tensor):
    # 在 MindSpore 中，梯度是针对 ms.Parameter 计算的。
    # 这个操作在 MindSpore 中没有直接对应。
    # 我们返回原张量，并假设调用者会使用 ops.grad()。
    return tensor

def backward(x, head_gradient=None):
    raise NotImplementedError(
        "MindSpore does not support .backward() on tensors. "
        "Use ops.grad() instead."
    )

def grad(x):
    raise NotImplementedError(
        "MindSpore does not have a .grad attribute. "
        "Gradients are returned by ops.grad()."
    )

def is_no_grad(x):
    # 无法直接检查梯度状态，默认返回 False
    return False

def is_recording():
    # MindSpore 默认在图模式下“记录”
    return True

# 这是一个与 PyTorch/MXNet 兼容的上下文管理器
record_grad = ms.autograd.record

# MindSpore 提供了 no_grad 上下文管理器
no_grad = ms.no_grad
