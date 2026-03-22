from .dataset import PrecomputedCSVDataset, BLOSUMCSVDataset, PrecomputedCSVForCRFDataset, PrecomputedCSVForOverlapCRFDataset
try:
    from .metrics import compute_metrics, add_dict_to_writer, compute_crf_metrics
except ImportError:
    def add_dict_to_writer(writer, metrics_dict, global_step, prefix=''):
        '''Adds a dictionary of scalar metrics to a TensorBoard writer.'''
        if prefix:
            prefix = f'{prefix}/'
        for k, v in metrics_dict.items():
            writer.add_scalar(f'{prefix}{k}', v, global_step)

    def compute_metrics(*args, **kwargs):
        pass

    def compute_crf_metrics(*args, **kwargs):
        pass