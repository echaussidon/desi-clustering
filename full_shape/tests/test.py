"""
salloc -N 1 -C "gpu&hbm80g" -t 02:00:00 --gpus 4 --qos interactive --account desi_g
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
srun -n 4 python test.py
"""
import os
import sys
import functools
from pathlib import Path

import jax

sys.path.insert(0, '../')
import tools
from tools import setup_logging
from compute_fiducial_stats import compute_fiducial_stats_from_options


def test(stats=['mesh2_spectrum']):
    meas_dir = Path(Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks')
    for tracer in ['LRG']:
        for zrange in tools.propose_fiducial('zranges', tracer):
             for region in ['NGC', 'SGC']:
                catalog_args = dict(version='holi-v1-altmtl', tracer=tracer, zrange=zrange, region=region, imock=451)
                compute_fiducial_stats_from_options(stats, catalog=catalog_args, get_measurement_fn=functools.partial(tools.get_measurement_fn, meas_dir=meas_dir), mesh2_spectrum={'cut': True, 'auw': True}, particle2_correlation={'cut': True})


if __name__ == '__main__':

    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
    jax.distributed.initialize()
    setup_logging()
    test(stats=['particle2_correlation'])
    jax.distributed.shutdown()