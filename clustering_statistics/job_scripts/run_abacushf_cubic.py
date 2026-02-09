"""
salloc -N 1 -C gpu -t 02:00:00 --gpus 4 --qos interactive --account desi_g
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
srun python run_abacushf_cubic.py
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import time
import logging
import itertools
import numpy as np
from dataclasses import dataclass
from typing import Iterable, Optional
from clustering_statistics.tools_abacushf_cubic import (
    abacus_hf_mock_path,
    get_hf_stats_fn,
    get_clustering_positions_weights,
    zsnap_to_zrange,
)
from clustering_statistics.spectrum2_tools import compute_box_mesh2_spectrum, compute_box_mesh2_cross_spectrum
from clustering_statistics.spectrum3_tools import compute_box_mesh3_spectrum, compute_box_mesh3_cross_spectrum
from clustering_statistics.tools import get_box_stats_fn
from mockfactory import setup_logging

logger = logging.getLogger('run_abacushf_cubic')

@dataclass(frozen=True)
class Task:
    version: str
    tracer: tuple[str, ...]
    zsnap: float
    imock: int
    los: str
    flavor: tuple[str | None, ...]
    cbox: str = "c000"

    def compute_cross_spectrum(self):
        if len(set(self.tracer)) == 1:
            return False
        return True

    def check(self):
        if len(self.flavor) != len(self.tracer):
            raise ValueError("the length of flavor does not match the length of tracer")
        cache = {}
        for tracer, flavor in zip(self.tracer, self.flavor):
            if tracer not in cache:
                cache[tracer] = flavor
                continue
            if cache[tracer] != flavor:
                raise ValueError(f"having more than two different flavors for {tracer} ({cache[tracer]}, {flavor}) does not make sense")

    def get_data_funcs(self):
        cache = {}
        retval = []
        for tracer, flavor in zip(self.tracer, self.flavor):
            if tracer in cache:
                retval.append(cache[tracer])
                continue
            data_fn = abacus_hf_mock_path(version=self.version, tracer=tracer, zsnap=self.zsnap, imock=self.imock, flavor=flavor, cbox=self.cbox)
            get_data = lambda data_fn=data_fn, los=self.los, zsnap=self.zsnap: get_clustering_positions_weights(data_fn, los=los, zsnap=zsnap)
            cache[tracer] = get_data
            retval.append(get_data)
        retval = tuple(retval)
        if not self.compute_cross_spectrum():
            retval = retval[:1]
        return retval


def flavors_for(version: str, tracer: str) -> tuple[Optional[str], ...]:
    if version == "v1":
        return (None,)
    if version == "v2":
        return FLAVORS_V2[tracer]
    return FLAVORS_VAR[tracer]


def iter_tasks(
    tracers: Iterable[tuple[str | tuple[str, ...], float]],
    versions: Iterable[str],
    los_list: Iterable[str],
    cbox_list: Iterable[str],
    imocks: Optional[Iterable[int]] = None,
) -> list[Task]:
    out: list[Task] = []
    for version in versions:
        nreal = NREAL[version]
        ims = list(range(nreal)) if imocks is None else list(imocks)
        for los in los_list:
            for cbox in (cbox_list if version == "variations" else ["c000"]):
                for imock in ims:
                    for tracer, zsnap in tracers:
                        if isinstance(tracer, tuple):
                            # do not correlate two different flavors of the same tracer
                            unique_tracers, unique_inverse = np.unique(tracer, return_inverse=True)
                            flavors = itertools.product(*[flavors_for(version, x) for x in unique_tracers])
                            flavors = [tuple(np.array(flavor)[unique_inverse].tolist()) for flavor in flavors]
                            for flavor in flavors:
                                out.append(Task(version, tracer, float(zsnap), int(imock), los, flavor, cbox))
                        else:
                            for flavor in flavors_for(version, tracer):                
                                out.append(Task(version, (tracer,), float(zsnap), int(imock), los, (flavor,), cbox))
    for task in out:
        task.check()
    return out

def _maybe_skip(fn: Path, overwrite: bool) -> bool:
    return (not overwrite) and fn.exists()


def run_task(task: Task, todo: set[str], spectrum_args: dict, overwrite: bool = False):
    import jax
    from jaxpower.mesh import create_sharding_mesh

    cache = {}

    stats_common = dict(
        stats_dir=STATS_DIR,
        version=task.version,
        tracer=task.tracer,
        zsnap=task.zsnap,
        cosmo=task.cbox,
        hod=tuple(f or 'base' for f in task.flavor),
        zrange=zsnap_to_zrange(task.zsnap),
        los=task.los,
        imock=task.imock,
        ext="h5",
    )


    if "mesh2_spectrum" in todo and len(task.tracer) in [1, 2]:
        out_fn = get_box_stats_fn(**stats_common, kind="mesh2_spectrum")
        if not _maybe_skip(out_fn, overwrite):
            with create_sharding_mesh():
                if task.compute_cross_spectrum():
                    spectrum = compute_box_mesh2_cross_spectrum(
                        *task.get_data_funcs(), get_shifted=GET_SHIFTED, get_shifted2=GET_SHIFTED2, cache=cache, los=task.los, **spectrum_args
                    )
                else:
                    spectrum = compute_box_mesh2_spectrum(
                        *task.get_data_funcs(), get_shifted=GET_SHIFTED, cache=cache, los=task.los, **spectrum_args
                    )
                if out_fn is not None and jax.process_index() == 0:
                    logger.info(f'Writing to {out_fn}')
                    spectrum.write(out_fn)
                jax.clear_caches()

    if "mesh3_spectrum_scoccimarro" in todo and len(task.tracer) in [1, 3]:
        bargs = spectrum_args | dict(basis="scoccimarro", ells=[0, 2], cellsize=8)
        out_fn = get_box_stats_fn(**stats_common, kind="mesh3_spectrum", basis=bargs["basis"])
        if not _maybe_skip(out_fn, overwrite):
            with create_sharding_mesh():
                if task.compute_cross_spectrum():
                    spectrum = compute_box_mesh3_cross_spectrum(
                        *task.get_data_funcs(), get_shifted=GET_SHIFTED, cache=cache , los=task.los, **bargs
                    )
                else:
                    spectrum = compute_box_mesh3_spectrum(
                        *task.get_data_funcs(), get_shifted=GET_SHIFTED, cache=cache , los=task.los, **bargs
                    )
                if out_fn is not None and jax.process_index() == 0:
                    logger.info(f'Writing to {out_fn}')
                    spectrum.write(out_fn)
                jax.clear_caches()

    if "mesh3_spectrum_sugiyama" in todo and len(task.tracer) in [1, 3]:
        bargs = spectrum_args | dict(
            basis="sugiyama-diagonal",
            ells=[(0, 0, 0), (2, 0, 2)],
            cellsize=6.25,
        )
        out_fn = get_box_stats_fn(**stats_common, kind="mesh3_spectrum", basis=bargs["basis"])
        if not _maybe_skip(out_fn, overwrite):
            with create_sharding_mesh():
                if task.compute_cross_spectrum():
                    spectrum = compute_box_mesh3_cross_spectrum(
                        *task.get_data_funcs(), get_shifted=GET_SHIFTED, cache=cache , los=task.los, **bargs
                    )
                else:
                    spectrum = compute_box_mesh3_spectrum(
                        *task.get_data_funcs(), get_shifted=GET_SHIFTED, cache=cache , los=task.los, **bargs
                    )
                if out_fn is not None and jax.process_index() == 0:
                    logger.info(f'Writing to {out_fn}')
                    spectrum.write(out_fn)
                jax.clear_caches()


# ---------------- config ----------------

TRACER_ZSNAPS = [
    # ("LRG", 0.5),
    # ("LRG", 0.725),
    # ("LRG", 0.95),
    # ("ELG", 0.95),
    # ("ELG", 1.175),
    # ("ELG", 1.475),
    # ("QSO", 1.4),
    # ("BGS-21.35", 0.3),
    ("QSO", 0.95),
]

TRACER_ZSNAPS_CROSS = [
    (("LRG", "ELG"), 0.95),
    (("LRG", "QSO"), 0.95),
    (("ELG", "QSO"), 0.95),
]

VERSIONS = ["v2"]          # ["v1", "v2", "variations"]
LOS_LIST = ["z"]           # ["x","y","z"]
CBOX_LIST = ["c000"]       # only used for variations

TODO = {
    "mesh2_spectrum",
    "mesh3_spectrum_scoccimarro",
    "mesh3_spectrum_sugiyama",
}

# ---------------- end config ----------------

SPECTRUM_ARGS = dict(boxsize=2000.,ells=(0, 2, 4), cellsize=5.)   # for mesh2
CROSS_SPECTRUM_ARGS = SPECTRUM_ARGS | dict(ells=(0, 1, 2, 3, 4, 5))   # for cross mesh2
GET_SHIFTED = None
GET_SHIFTED2 = None
OVERWRITE = True

NREAL = {"v1": 25, "v2": 25, "variations": 6}

FLAVORS_V2 = {
    "BGS-21.35": ("base", "base_B", "base_dv", "base_B_dv"),
    "LRG": ("base", "base_B", "base_dv", "base_B_dv"),
    "ELG": ("base_conf_nfwexp",),
    "QSO": ("base",),
}
FLAVORS_VAR = {
    "BGS-21.35": ("base", "base_A", "base_B", "base_dv", "base_A_dv", "base_B_dv"),
    "LRG": ("base", "base_dv", "base_B_dv", "base_A_dv"),
    "ELG": ("base_conf", "base_conf_nfwexp"),
    "QSO": ("base", "base_dv"),
}

STATS_DIR = Path(os.getenv("SCRATCH", ".")) / "measurements_abacushf"

def main():
    setup_logging()
    import jax
    from jax import config
    config.update('jax_enable_x64', True)
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'
    jax.distributed.initialize()

    # tasks = iter_tasks(TRACER_ZSNAPS, VERSIONS, LOS_LIST, CBOX_LIST)
    tasks = iter_tasks(TRACER_ZSNAPS_CROSS, VERSIONS, LOS_LIST, CBOX_LIST)

    for t in tasks:
        # run_task(t, TODO, SPECTRUM_ARGS, overwrite=OVERWRITE)
        run_task(t, TODO, CROSS_SPECTRUM_ARGS, overwrite=OVERWRITE)


if __name__ == '__main__':
    main()
