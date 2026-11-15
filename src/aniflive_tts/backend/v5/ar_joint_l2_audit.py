"""Read-only audit of FFN2 access-policy windows across a joint-graph child."""
import ctypes
import json
from pathlib import Path


FIELDS = ('base', 'bytes', 'hitRatio', 'hitProp', 'missProp')
EXPECTED_ROWS = 24


def _graph_handle(graph):
    value = graph.raw_cuda_graph() if hasattr(graph, 'raw_cuda_graph') else graph
    if isinstance(value, ctypes.c_void_p):
        value = value.value
    if value is None:
        raise ValueError('CUDA graph handle is null')
    return int(value)


def _index_rows(rows, label):
    if not isinstance(rows, list) or len(rows) != EXPECTED_ROWS:
        raise ValueError(f'{label} must contain exactly {EXPECTED_ROWS} FFN2 rows')
    indexed = {}
    for row in rows:
        weightptr = row.get('weightptr')
        if not isinstance(weightptr, str) or not weightptr.startswith('0x'):
            raise ValueError(f'{label} contains an invalid verified weight pointer')
        if weightptr in indexed:
            raise ValueError(f'{label} contains duplicate verified weight pointer {weightptr}')
        if any(field not in row for field in FIELDS):
            raise ValueError(f'{label} row {weightptr} is missing an access-policy field')
        indexed[weightptr] = {field: row[field] for field in FIELDS}
    return indexed


def audit(originalgraph, outergraph, librarypath):
    """Compare all 24 FFN2 windows, keyed by validated runtime weight pointer."""
    library = ctypes.CDLL(str(Path(librarypath).resolve(strict=True)))
    function = library.ar_joint_l2_audit
    function.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p,
                         ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
    function.restype = ctypes.c_int
    output = ctypes.create_string_buffer(32768)
    error = ctypes.create_string_buffer(1024)
    status = function(_graph_handle(originalgraph), _graph_handle(outergraph),
                      output, len(output), error, len(error))
    if status:
        raise RuntimeError(error.value.decode('utf-8', errors='replace'))

    raw = json.loads(output.value.decode('utf-8'))
    original = _index_rows(raw.get('original'), 'original graph')
    embedded = _index_rows(raw.get('embedded'), 'embedded child graph')
    if original.keys() != embedded.keys():
        missing = sorted(original.keys() - embedded.keys())
        extra = sorted(embedded.keys() - original.keys())
        raise RuntimeError(f'FFN2 weight-pointer sets differ; missing={missing}, extra={extra}')

    rows = []
    for weightptr in sorted(original):
        before, after = original[weightptr], embedded[weightptr]
        rows.append({'weightptr': weightptr, 'original': before, 'embedded': after,
                     'attributes_identical': before == after})
    report = {'status': 'passed' if all(row['attributes_identical'] for row in rows) else 'failed',
              'comparison': 'all FFN2 nodes, including unselected windows, keyed by verified weight pointer',
              'rows': rows, 'count': len(rows),
              'identical_count': sum(row['attributes_identical'] for row in rows)}
    if report['status'] != 'passed':
        raise RuntimeError('Joint graph changed FFN2 access-policy attributes: ' + json.dumps(report))
    return report
