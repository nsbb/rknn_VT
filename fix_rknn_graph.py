"""
ReduceMean → depthwise Conv 교체 + H=1 중간 텐서 Pad/Slice 우회
- ReduceMean(axis=[2]):    (1,C,H,W) → (1,C,1,W)  → depthwise Conv(kernel=(H,1), w=1/H)
- ReduceMean(axis=[2,3]):  (1,C,1,W) → (1,C,1,1)  → depthwise Conv(kernel=(1,W), w=1/W)
- f1 branch H=1 fix:      rm_out → Pad(1→4) → [f1 ops] → Slice(0:1) → Add
  (Conv(1×3) on H=1 fails on NPU as intermediate op; H=4 works)
"""
import numpy as np
import onnx
from onnx import numpy_helper, helper, TensorProto, shape_inference

model = onnx.load('BCResNet-t2-Focal-ep110.onnx')
model = shape_inference.infer_shapes(model)
graph = model.graph

# value_info shape 수집
shapes = {}
for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
    try:
        s = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        shapes[vi.name] = s
    except:
        pass

new_initializers = []
new_nodes = []
# (rm_conv_node, rm_out) pairs for Pad/Slice insertion
rm_conv_pairs = []

# ── Pass 1: ReduceMean → Conv 교체 ─────────────────────────────────────────
for node in graph.node:
    if node.op_type != 'ReduceMean':
        new_nodes.append(node)
        continue

    axes = list(node.attribute[0].ints)
    in_name = node.input[0]
    out_name = node.output[0]
    in_shape = shapes.get(in_name)

    if in_shape is None:
        print(f'WARNING: no shape for {node.name}'); new_nodes.append(node); continue

    if axes == [2]:
        C, H = in_shape[1], in_shape[2]
        w_name = f'{node.name}_w'
        w = (1.0 / H) * np.ones((C, 1, H, 1), dtype=np.float32)
        new_initializers.append(numpy_helper.from_array(w, name=w_name))
        conv_node = helper.make_node(
            'Conv',
            inputs=[in_name, w_name],
            outputs=[out_name],
            name=f'{node.name}_conv',
            dilations=[1, 1],
            group=C,
            kernel_shape=[H, 1],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
        )
        new_nodes.append(conv_node)
        rm_conv_pairs.append((conv_node, out_name))
        print(f'  ReduceMean(axis=2) → Conv depthwise: {node.name}, C={C}, H={H}')

    elif axes == [2, 3]:
        C = in_shape[1]
        W = in_shape[3]
        w_name = f'{node.name}_gap_w'
        w = (1.0 / W) * np.ones((C, 1, 1, W), dtype=np.float32)
        new_initializers.append(numpy_helper.from_array(w, name=w_name))
        conv_node = helper.make_node(
            'Conv',
            inputs=[in_name, w_name],
            outputs=[out_name],
            name=f'{node.name}_gap_conv',
            dilations=[1, 1],
            group=C,
            kernel_shape=[1, W],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
        )
        new_nodes.append(conv_node)
        print(f'  ReduceMean(axis=[2,3]) → depthwise Conv(1,{W}): {node.name}, C={C}')

    else:
        print(f'WARNING: Unexpected axes={axes} in {node.name}')
        new_nodes.append(node)

# ── Pass 2: H=1 f1-branch Pad/Slice ───────────────────────────────────────
# consumer map: tensor → node (input[0] 기준)
def build_consumer_map(nodes):
    m = {}
    for n in nodes:
        for inp in n.input:
            if inp and inp not in m:
                m[inp] = n
    return m

def find_any_consumer(tensor, nodes):
    """tensor를 아무 입력으로라도 소비하는 첫 번째 노드"""
    for n in nodes:
        if tensor in list(n.input):
            return n
    return None

def trace_f1_chain(f1_0_conv_node, nodes):
    """
    f1.0.Conv → Sigmoid → Mul → f1.1.Conv 추적
    반환: (f1_1_conv_node, f1_1_out) or (None, None)
    """
    cur_out = f1_0_conv_node.output[0]
    for expected_op in ['Sigmoid', 'Mul', 'Conv']:
        consumer = find_any_consumer(cur_out, nodes)
        if consumer is None or consumer.op_type != expected_op:
            print(f'  WARNING: expected {expected_op}, got {consumer.op_type if consumer else None} for {cur_out}')
            return None, None
        cur_out = consumer.output[0]
        last_node = consumer
    return last_node, cur_out  # f1.1.Conv node, f1.1.Conv output

# insert_after: node_name → list of (node to insert after it)
insert_after = {}
# Add input remap: old_tensor → new_tensor
add_input_remap = {}

consumer_map = build_consumer_map(new_nodes)

for rm_conv_node, rm_out in rm_conv_pairs:
    # ReduceMean 입력 shape에서 C, H(f2), W 추출
    # rm_conv_node의 입력 텐서 = ReduceMean_conv 입력 = BCBlock 입력
    rm_in = rm_conv_node.input[0]  # e.g. /backbone/cnn_head/.../Relu_output_0
    rm_in_shape = shapes.get(rm_in)  # (1, C, H_f2, W)

    # f1.0.Conv 찾기
    f1_0_conv = consumer_map.get(rm_out)
    if f1_0_conv is None or f1_0_conv.op_type != 'Conv':
        print(f'  SKIP {rm_out}: f1_0_conv not found or wrong type ({f1_0_conv})')
        continue

    # f1 chain 추적
    f1_1_conv_node, f1_1_out = trace_f1_chain(f1_0_conv, new_nodes)
    if f1_1_conv_node is None:
        continue

    # ── Pad 노드 생성: rm_out (1,C,1,W) → pad_out (1,C,4,W) ──
    pad_out = rm_out + '_padded'
    pads_name = rm_out.replace('/', '_') + '_pad_vals'
    pads_val = np.array([0, 0, 0, 0,  0, 0, 3, 0], dtype=np.int64)  # H 아래 3행 zero-pad
    new_initializers.append(numpy_helper.from_array(pads_val, name=pads_name))
    pad_node = helper.make_node(
        'Pad',
        inputs=[rm_out, pads_name],
        outputs=[pad_out],
        name=rm_out.replace('/', '_') + '_pad',
        mode='constant',
    )
    # rm_conv_node 바로 뒤에 삽입
    insert_after.setdefault(rm_conv_node.name, []).append(pad_node)

    # f1.0.Conv 입력 교체: rm_out → pad_out
    for i, inp in enumerate(f1_0_conv.input):
        if inp == rm_out:
            f1_0_conv.input[i] = pad_out
            break

    # ── Slice 노드 생성: f1_1_out (1,C,4,W) → slice_out (1,C,1,W) ──
    slice_out = f1_1_out + '_sliced'
    base = f1_1_out.replace('/', '_')
    starts_name = base + '_sl_starts'
    ends_name   = base + '_sl_ends'
    axes_name   = base + '_sl_axes'
    new_initializers.append(numpy_helper.from_array(np.array([0], dtype=np.int64), name=starts_name))
    new_initializers.append(numpy_helper.from_array(np.array([1], dtype=np.int64), name=ends_name))
    new_initializers.append(numpy_helper.from_array(np.array([2], dtype=np.int64), name=axes_name))
    slice_node = helper.make_node(
        'Slice',
        inputs=[f1_1_out, starts_name, ends_name, axes_name],
        outputs=[slice_out],
        name=base + '_slice',
    )
    # f1.1.Conv 바로 뒤에 삽입
    insert_after.setdefault(f1_1_conv_node.name, []).append(slice_node)

    # ── Expand 노드 생성: slice_out (1,C,1,W) → expand_out (1,C,H_f2,W) ──
    # AddRelu fusion에서 broadcast(H=1→H_f2)가 RKNN에서 깨지는 문제 우회
    expand_out = f1_1_out + '_expanded'
    if rm_in_shape is not None:
        _, C, H_f2, W = rm_in_shape
        expand_shape_val = np.array([1, C, H_f2, W], dtype=np.int64)
    else:
        # fallback: Expand 없이 broadcast에 의존
        expand_out = slice_out
        H_f2 = '?'
    if rm_in_shape is not None:
        exp_shape_name = base + '_exp_shape'
        new_initializers.append(numpy_helper.from_array(expand_shape_val, name=exp_shape_name))
        expand_node = helper.make_node(
            'Expand',
            inputs=[slice_out, exp_shape_name],
            outputs=[expand_out],
            name=base + '_expand',
        )
        insert_after.setdefault(f1_1_conv_node.name, []).append(expand_node)

    # Add 입력 교체: f1_1_out → expand_out (broadcast 없이 same-shape Add)
    add_input_remap[f1_1_out] = expand_out

    print(f'  Pad/Slice/Expand: {rm_out} → f1 → slice(H=1) → expand(H={H_f2}) → Add')

# Add 노드 입력 교체
for node in new_nodes:
    if node.op_type == 'Add':
        for i, inp in enumerate(node.input):
            if inp in add_input_remap:
                node.input[i] = add_input_remap[inp]

# ── 최종 노드 리스트 (위상 정렬 순서 유지) ────────────────────────────────
final_nodes = []
for node in new_nodes:
    final_nodes.append(node)
    for extra_node in insert_after.get(node.name, []):
        final_nodes.append(extra_node)

# 그래프 재구성
del graph.node[:]
graph.node.extend(final_nodes)
graph.initializer.extend(new_initializers)

# shape inference
model = shape_inference.infer_shapes(model)

# ── 검증 ─────────────────────────────────────────────────────────────────────
import onnxruntime as ort
import sys, wave
sys.path.insert(0, '.')
from inference_rknn import LogMel

with wave.open('wallpad_HiWonder_251113/lkk/lkk_1_2.wav', 'rb') as wf:
    audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
feat = LogMel()(audio)[np.newaxis, np.newaxis, ...]

def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()

sess_orig = ort.InferenceSession('BCResNet-t2-Focal-ep110.onnx')
orig_out = sess_orig.run(None, {sess_orig.get_inputs()[0].name: feat})[0]
orig_probs = softmax(orig_out.squeeze())
print(f'\nOriginal ONNX probs: {orig_probs}')

onnx.save(model, 'BCResNet-t2-npu-fixed.onnx')
print('Saved: BCResNet-t2-npu-fixed.onnx')

sess_new = ort.InferenceSession('BCResNet-t2-npu-fixed.onnx')
new_out = sess_new.run(None, {sess_new.get_inputs()[0].name: feat})[0]
new_probs = softmax(new_out.squeeze())
print(f'Fixed  ONNX probs: {new_probs}')
print(f'Match (atol=1e-4): {np.allclose(orig_probs, new_probs, atol=1e-4)}')
print(f'Max diff: {np.abs(orig_probs - new_probs).max():.6f}')
