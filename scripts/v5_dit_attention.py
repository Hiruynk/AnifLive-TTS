"""Pinned native attention graph rewrite. Copyright 2026 Hiruynk; see LICENSE."""

def native_attention_sites(source):
    """Identify and validate the exact qualified ONNX attention chains."""
    import onnx

    model = onnx.load(str(source), load_external_data=False)
    nodes = {node.name: node for node in model.graph.node}
    if sum(node.op_type == 'Softmax' for node in model.graph.node) != 22:
        raise RuntimeError('Expected exactly 22 attention Softmax nodes')
    sites = []
    for index in range(22):
        prefix = '/attn' + (f'_{index}' if index else '')
        def node(suffix, kind):
            value = nodes.get(prefix + '/' + suffix)
            if value is None or value.op_type != kind:
                raise RuntimeError(f'Unexpected ONNX attention node: {prefix}/{suffix}')
            return value
        q, k = node('Mul_13', 'Mul'), node('Mul_14', 'Mul')
        kt = node('Transpose_3', 'Transpose')
        qk, mask = node('MatMul', 'MatMul'), node('Expand', 'Expand')
        where = node('Where_1', 'Where')
        scores, added = node('Cast_11', 'Cast'), node('Add_2', 'Add')
        softmax, isnan = node('Softmax', 'Softmax'), node('IsNaN', 'IsNaN')
        guarded = node('Where_2', 'Where')
        value_cast, pv = node('Cast_12', 'Cast'), node('MatMul_1', 'MatMul')
        transpose = node('Transpose_4', 'Transpose')
        perm = next((list(a.ints) for a in kt.attribute if a.name == 'perm'), None)
        if (list(qk.input) != [q.output[0], k.output[0]] or k.input[0] != kt.output[0]
                or perm != [0, 2, 3, 1] or where.input[0] != mask.output[0]
                or scores.input[0] != qk.output[0] or list(added.input) != [scores.output[0], where.output[0]]
                or softmax.input[0] != added.output[0] or isnan.input[0] != softmax.output[0]
                or guarded.input[0] != isnan.output[0] or guarded.input[2] != softmax.output[0]
                or list(pv.input) != [guarded.output[0], value_cast.output[0]]
                or transpose.input[0] != pv.output[0]):
            raise RuntimeError(f'Unexpected ONNX attention wiring: {prefix}')
        uses = [(n.name, i) for n in model.graph.node for i, v in enumerate(n.input) if v == pv.output[0]]
        if uses != [(transpose.name, 0)]:
            raise RuntimeError(f'Unexpected attention output consumers: {prefix}: {uses}')
        sites.append({'prefix': prefix, 'query': q.output[0], 'key_transposed': k.output[0],
                      'value': value_cast.input[0], 'mask': mask.output[0], 'old_output': pv.output[0],
                      'consumer': transpose.name})
    return sites


def install_native_attention(network, trt, sites, *, use_lengths=False, unmasked=False):
    """Replace parsed PV outputs with the selected native attention representation."""
    if use_lengths and unmasked:
        raise ValueError('Prefix lengths and unmasked attention are separate candidates')
    layers = [network.get_layer(i) for i in range(network.num_layers)]
    tensors = {network.get_input(i).name: network.get_input(i) for i in range(network.num_inputs)}
    for layer in layers:
        for i in range(layer.num_outputs):
            tensor = layer.get_output(i)
            if tensor is not None:
                tensors[tensor.name] = tensor
    lengths = None
    if use_lengths:
        mask_input = tensors.get('mask')
        if mask_input is None or mask_input.dtype != trt.bool or tuple(mask_input.shape) != (1, -1):
            raise RuntimeError('Expected dynamic boolean [1,T] network mask')
        cast_mask = network.add_cast(mask_input, trt.int32)
        if cast_mask is None:
            raise RuntimeError('Could not cast mask for prefix lengths')
        cast_mask.name = 'native_attention_mask_to_int32'
        reduction = network.add_reduce(cast_mask.get_output(0), trt.ReduceOperation.SUM, 1 << 1, False)
        if reduction is None:
            raise RuntimeError('Could not reduce mask to prefix lengths')
        reduction.name = 'native_attention_prefix_lengths'
        lengths = reduction.get_output(0)
        if lengths.dtype != trt.int32 or tuple(lengths.shape) != (1,):
            raise RuntimeError('Unexpected native attention prefix length tensor')
    wires = []
    for site in sites:
        try:
            q, kt, v, mask, old = (tensors[site[name]] for name in
                                    ('query', 'key_transposed', 'value', 'mask', 'old_output'))
        except KeyError as error:
            raise RuntimeError(f'Parsed attention tensor name missing: {site["prefix"]}: {error}') from error
        if (q.dtype != trt.float16 or kt.dtype != trt.float16 or v.dtype != trt.float16
                or mask.dtype != trt.bool or old.dtype != trt.float32
                or tuple(q.shape) != (1, 16, -1, 64) or tuple(kt.shape) != (1, 16, 64, -1)
                or tuple(v.shape) != (1, 16, -1, 64) or tuple(mask.shape) != (1, 16, -1, -1)
                or tuple(old.shape) != (1, 16, -1, 64)):
            raise RuntimeError(f'Unexpected parsed attention shapes/dtypes: {site["prefix"]}: '
                               f'{[(t.name, str(t.dtype), tuple(t.shape)) for t in (q, kt, v, mask, old)]}')
        consumers = [(layer, i) for layer in layers for i in range(layer.num_inputs)
                     if layer.get_input(i) is not None and layer.get_input(i).name == old.name]
        if len(consumers) != 1:
            raise RuntimeError(f'Expected one parsed attention output consumer: {site["prefix"]}')
        untranspose = network.add_shuffle(kt)
        if untranspose is None:
            raise RuntimeError(f'Could not untranspose key: {site["prefix"]}')
        untranspose.first_transpose = (0, 1, 3, 2)
        untranspose.name = site['prefix'] + '/native_key_untranspose'
        attention = network.add_attention_v2(q, untranspose.get_output(0), v,
                                             trt.AttentionNormalizationOp.SOFTMAX, trt.CausalMaskKind.NONE)
        if attention is None:
            raise RuntimeError(f'Could not create native attention: {site["prefix"]}')
        attention.name = site['prefix'] + '/native_attention'
        if use_lengths:
            if not hasattr(attention, 'key_value_lengths'):
                raise RuntimeError('TensorRT does not expose native attention key_value_lengths')
            if attention.key_value_form != trt.AttentionIOForm.PADDED_BHND:
                raise RuntimeError('Native attention requires padded BHND KV form')
            attention.key_value_lengths = lengths
        elif not unmasked:
            attention.mask = mask
        attention.decomposable = False
        output = attention.get_output(0)
        if output.dtype != trt.float16 or tuple(output.shape) != tuple(old.shape):
            raise RuntimeError(f'Unexpected native attention output: {site["prefix"]}')
        cast = network.add_cast(output, trt.float32)
        if cast is None:
            raise RuntimeError(f'Could not cast native attention output: {site["prefix"]}')
        cast.name = site['prefix'] + '/native_output_float'
        consumers[0][0].set_input(consumers[0][1], cast.get_output(0))
        wires.append({**site, 'parsed_consumer': consumers[0][0].name,
                      'native_mask_shape': list(mask.shape),
                      'native_mask_representation': ('prefix_lengths' if use_lengths else
                                                     'unmasked' if unmasked else 'boolean_mask'),
                      'native_output_dtype': 'float16',
                      'replacement_output_dtype': 'float32'})
    if len(wires) != 22:
        raise RuntimeError('Expected exactly 22 native attention replacements')
    return wires
