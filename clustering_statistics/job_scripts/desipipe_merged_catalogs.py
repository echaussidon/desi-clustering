"""
Script to create and spawn desipipe tasks to compute merged catalogs.
To create and spawn the tasks on NERSC, use the following commands:
```bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
python desipipe_merged_catalogs.py          # create the list of tasks
desipipe tasks  -q merged_catalogs          # check the list of tasks
desipipe spawn  -q merged_catalogs --spawn  # spawn the jobs
desipipe queues -q merged_catalogs          # check the queue
```
"""
import os
import sys
from pathlib import Path
import functools
from time import time

import numpy as np
from desipipe import Queue, Environment, TaskManager, spawn, setup_logging

from clustering_statistics import tools


setup_logging()

queue = Queue('merged_catalogs')
queue.clear(kill=False)

output, error = 'slurm_outputs/merged_catalogs/slurm-%j.out', 'slurm_outputs/merged_catalogs/slurm-%j.err'
kwargs = {}
# tmp_dir = Path(os.getenv('SCRATCH'), 'tmp')
# tmp_dir.mkdir(exist_ok=True)
environ = Environment('nersc-cosmodesi')
tm = TaskManager(queue=queue, environ=environ)
tm = tm.clone(scheduler=dict(max_workers=30), 
              provider=dict(provider='nersc', time='00:30:00', mpiprocs_per_worker=1, nodes_per_worker=0.2, 
                            output=output, error=error, stop_after=1, constraint='cpu'))

@tm.python_app
def merge_catalogs(output, inputs, merge_catalogs=tools.merge_catalogs, read_catalog=tools._read_catalog, **kwargs):
    from mockfactory import setup_logging
    setup_logging()
    merge_catalogs(output, inputs, **kwargs)


@tm.python_app
def merge_randoms_catalogs(output, inputs, parent_randoms_fn=None, merge_catalogs=tools.merge_randoms_catalogs, 
                           read_catalog=tools._read_catalog, expand_randoms=tools.expand_randoms, **kwargs):
    from mockfactory import setup_logging
    setup_logging()
    merge_catalogs(output, inputs, parent_randoms_fn=parent_randoms_fn, **kwargs)


if __name__ == '__main__':

    mode = 'slurm'
    # mode = 'interactive'

    version = 'glam-uchuu-v1-altmtl'
    out_dir = Path('/global/cfs/cdirs/desi/mocks/cai/LSS/DA2/mocks/desipipe/') / version / 'merged'
    # out_dir = Path(os.getenv('SCRATCH')) / 'cai-dr2-benchmarks' / version / 'merged' # / '{noric or ric}'
    
    kinds = ['data','single_randoms']
    tracers = ['LRG', 'ELG_LOPnotqso', 'QSO']
    # tracers = ['QSO']
    regions = ['NGC', 'SGC']
    imocks = np.arange(100,150+1) # in this it is the number of mocks to merge
    nran_list = np.arange(18) # randoms to process
    factor = len(imocks)
    
    for kind in kinds:    
        for tracer in tracers:
            for region in regions:
                catalog_kws = dict(version=version, tracer=tracer, region=region)
    
                if 'data' in kind:
                    # Merge data mock catalogs
                    input_data_fns,_ = tools.checks_if_exists_and_readable(get_fn=functools.partial(tools.get_catalog_fn, kind=kind, **catalog_kws), 
                                                                           test_if_readable=False, imock=imocks)[0]
                    output_data_fn = tools.get_catalog_fn(kind=kind, cat_dir=out_dir, **catalog_kws)
                    
                    merge_catalogs(output_data_fn, input_data_fns, factor=factor)
    
                if 'randoms' in kind:
                    # Merge randoms catalogs
                    exists, missing, unreadable = tools.checks_if_exists_and_readable(get_fn=functools.partial(tools.get_catalog_fn, kind=kind, cat_dir=out_dir, **catalog_kws),
                                                                                      nran=nran_list)
                    rerun = [inran for inran in nran_list if (inran in unreadable[1]['nran']) or (inran not in exists[1]['nran'])]
                    for inran in rerun:
                        if 'glam' in version:
                            # 'glam-uchuu-v1-altmtl' randoms do not have RA and DEC columns so we use `expand_randoms`
                            parent_randoms_fn = tools.get_catalog_fn(kind='parent_randoms', version='data-dr2-v2', tracer=tracer, nran=len(nran_list))[inran]
                        else:
                            expand = None
                        input_randoms_fns,_ = tools.checks_if_exists_and_readable(get_fn=functools.partial(tools.get_catalog_fn, kind=kind, nran=inran, **catalog_kws),
                                                                                  test_if_readable=False, imock=imocks)[0]
                        output_randoms_fn = tools.get_catalog_fn(kind=kind, cat_dir=out_dir, nran=inran, **catalog_kws)
                        
                        merge_randoms_catalogs(output_randoms_fn, input_randoms_fns, parent_randoms_fn=parent_randoms_fn, factor=factor)
                    
