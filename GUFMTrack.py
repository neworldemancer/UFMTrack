#!/usr/bin/env python
# coding: utf-8
"""
   Copyright 2015-2023, University of Bern, Laboratory for High Energy Physics and Theodor Kocher Institute, M. Vladymyrov

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
   
   This module performs tracking of cells under flow based on cells segmentation.
"""

from __future__ import annotations


# load libs

import os
import copy
import time

from time import time as timer
from typing import List
from collections.abc import Iterable, Sequence
from tqdm import tqdm
import pickle

# import threading
# import asyncio
# import concurrent.futures
# import multiprocessing
# from threading import Thread
# from functools import partial

from math import *
import numpy as np

from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from ortools.graph import pywrapgraph as pg
from ortools.sat.python import cp_model

from matplotlib import pyplot as plt
import matplotlib.cm as cm

from config_manager import ConfigManager as Cfg

# import seaborn as sns


# Methods

# General utility methods

cfgm = Cfg()


def load_pckl(file_name, path=None):
    if path is not None:
        file_name = os.path.join(path, file_name)

    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pckl(d, file_name, pr=None, path=None):
    if path is not None:
        file_name = os.path.join(path, file_name)

    with open(file_name, 'wb') as f:
        pickle.dump(d, f, protocol=pr if pr is not None else pickle.DEFAULT_PROTOCOL)


# noinspection PyPep8Naming
def median_MAD(arr: Iterable[float], return_ofs: bool = False) -> tuple:
    """
    calculates median and median absolute deviation
    """
    arr_np = np.array(arr)
    m = np.median(arr_np)
    ofs = np.abs(arr_np - m)
    mad = np.median(ofs)

    if return_ofs:
        return m, mad, ofs
    else:
        return m, mad


def exclude_outliers(arr: Sequence[float], n_sigma: float = 3) -> Sequence[float]:
    med, mad, ofs = median_MAD(arr, return_ofs=True)
    sigma_est = mad * 1.4826

    n = len(arr)
    n_s = n_sigma * (0.5 if n < 3 else (1 if n > 15 else (0.5 + 0.5 * (n - 3) / 12)))
    lim = n_s * sigma_est
    mask = ofs < lim
    arr_new = [el for el, ok in zip(arr, mask) if ok]
    return arr_new


def angle_to_0pi(phi: float) -> float:
    while phi < 0:
        phi += np.pi
    while phi > np.pi:
        phi -= np.pi
    return phi


def angle_to_mpi2ppi2(phi: float) -> float:
    while phi < -np.pi / 2:
        phi += np.pi
    while phi > np.pi / 2:
        phi -= np.pi
    return phi


def ellipse_fit_linfit_old(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    assert (2 < len(x) == len(y))
    fit_failed = True
    a, b, phi = 0, 0, 0
    try:
        a, b = np.polyfit(x, y, 1)  # y == x * a + b
        phi = atan2(a, 1)
        fit_failed = False
        print(f'try y == x * a + b: y == x * {a:.1f} + {b:.1f}, phi = {phi}')
    except Exception as exc:
        try:
            a, b = np.polyfit(y, x, 1)  # x == y * a + b
            phi = atan2(1, a)
            fit_failed = False
            print(f'try x == y * a + b: x == y * {a:.1f} + {b:.1f}, phi = {phi}')
        except:
            pass

    print(a, b, phi, fit_failed)

    if fit_failed:
        std_x = np.std(x)
        std_y = np.std(y)
        ecc = (std_x / std_y) if (std_x > std_y) else (std_y / std_x)
        phi = 0
        return ecc, phi

    x = np.array(x)
    y = np.array(y)

    rot_mtr = np.array([[cos(phi), sin(phi)], [-sin(phi), cos(phi)]])
    in_xy = np.stack((x, y), axis=0)
    rot_xy = np.matmul(rot_mtr, in_xy)
    rot_x, rot_y = rot_xy
    std_x = np.std(rot_x)
    std_y = np.std(rot_y)

    if std_y > std_x:  # redo
        print('redoing + pi/2')
        phi += np.pi / 2
        rot_mtr = np.array([[cos(phi), sin(phi)], [-sin(phi), cos(phi)]])
        in_xy = np.stack((x, y), axis=0)
        rot_xy = np.matmul(rot_mtr, in_xy)
        rot_x, rot_y = rot_xy
        std_x = np.std(rot_x)
        std_y = np.std(rot_y)

    plt.scatter(x - x.mean(), y - y.mean())
    plt.scatter(rot_x - rot_x.mean(), rot_y - rot_y.mean())
    plt.axes().set_aspect('equal', 'datalim')

    assert (std_x >= std_y)
    ecc = 1 if std_x == std_y else (std_x / std_y if std_y != 0 else 1000)

    phi = angle_to_0pi(phi)

    if ecc < 1:
        plt.title('ecc, phi' + str((ecc, phi)))
        plt.show()
        plt.close()

    return ecc, phi


def angle_3d(v1: Iterable[float], v2: Iterable[float]) -> float:
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    v1_2 = (v1 ** 2).sum()
    v2_2 = (v2 ** 2).sum()
    if v1_2 == 0 or v2_2 == 0:
        return 0
    v1_v2 = (v1 * v2).sum()

    cos_phi = v1_v2 / (sqrt(v1_2 * v2_2))
    cos_phi = min(1, max(-1, cos_phi))
    return acos(cos_phi)


def mean_std(arr: Sequence[float]) -> tuple[float, float]:
    return (np.mean(arr), np.std(arr)) if len(arr) else (0., 0.)


def min_max(arr: Sequence[float]) -> tuple[float, float]:
    return (np.min(arr), np.max(arr)) if len(arr) else (0., 0.)


def iterable(obj):
    try:
        iter(obj)
    except Exception:
        return False
    return True


def ellipse_fit(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    assert (2 < len(x) == len(y))
    fit_failed = True

    ecc, phi = 1., 0.
    std_x, std_y = 1., 1.
    ei_vals = None
    try:
        dx, dy = np.mean(x), np.mean(y)
        xy = np.stack((x - dx, y - dy), axis=0)
        xy += np.random.normal(scale=1e-3, size=xy.shape)
        cov = np.cov(xy)
        ei_vals, ei_vects = np.linalg.eig(cov)
        ei_vects = ei_vects.transpose()

        std_x, std_y = np.sqrt(ei_vals)

        var_x, var_y = ei_vals
        var_y, var_x = (var_x, var_y) if var_y > var_x else (var_y, var_x)
        assert (var_x >= var_y)

        ecc = 1. if var_x == var_y else (var_x / var_y if var_y != 0 else 1000000.)
        ecc = sqrt(ecc)

        first_ei_vect = ei_vects[np.argmax(ei_vals)]
        ev_x, ev_y = first_ei_vect
        phi = atan2(ev_y, ev_x)
        phi = angle_to_0pi(phi)
        fit_failed = False
    except:
        pass

    if fit_failed:
        std_x = np.std(x)
        std_y = np.std(y)

        if std_y > std_x:
            std_y, std_x = (std_x, std_y)
            phi = np.pi / 2
        else:
            phi = 0

        assert (std_x >= std_y)

        ecc = 1. if std_x == std_y else (std_x / std_y if std_y != 0 else 1000)

    assert ecc >= 1., f'ecc={ecc:.2f}={std_x}/{std_y}...{x};{y}; {ei_vals}'
    return ecc, phi


def gen_ellipsoid_points(x0: float, y0: float,
                         w: float,
                         ecc: float, phi: float,
                         w_fact: float = 10) -> tuple[np.ndarray, np.ndarray]:
    """generates w*f_fact points distributed according tp ecc & phi"""
    b = sqrt(w / (pi * ecc))
    cov = [
        [b * ecc, 0],
        [0, b]
    ]

    n = max(1, int(w_fact * w))
    pts = np.random.multivariate_normal([0, 0], cov, size=n)

    rot_mtr = np.array([[cos(phi), -sin(phi)], [sin(phi), cos(phi)]])

    in_xy = np.stack(pts.transpose(), axis=0)
    rot_xy = np.matmul(rot_mtr, in_xy)

    x, y = rot_xy

    x += x0
    y += y0

    return x, y


def ellipse_fit_ellipsoids(xs: Iterable[float], ys: Iterable[float],
                           ws: Iterable[float],
                           eccs: Iterable[float], phis: Iterable[float],
                           w_fact: float = 10) -> tuple[float, float]:
    all_x = []
    all_y = []
    for x, y, w, ecc, phi in zip(xs, ys, ws, eccs, phis):
        px, py = gen_ellipsoid_points(x, y, w, ecc, phi, w_fact)
        all_x.extend(px)
        all_y.extend(py)

    ecc, phi = ellipse_fit(all_x, all_y)
    return ecc, phi


def make_image_dirs():
    if cfgm.SAVE_IMS:
        for d in cfgm.IMG_DIRS:
            os.makedirs(d, exist_ok=True)


def wait_to_start():
    if cfgm.USE_SYNCH_FILES:
        if os.path.exists(cfgm.SYNC_FILE_DONE):
            os.remove(cfgm.SYNC_FILE_DONE)

        while not os.path.exists(cfgm.SYNC_FILE_START):
            time.sleep(10)

        os.remove(cfgm.SYNC_FILE_START)


def set_proc_end():
    if cfgm.USE_SYNCH_FILES:
        with open(cfgm.SYNC_FILE_DONE, 'wt') as f:
            f.write('ok')


# containers
class Cell:
    """
    Class representing one other_cell or a group of under-segmented cells
    """

    def __init__(self, in_line: str):
        self._dt = cfgm.DT
        self._dt_offset = cfgm.DT_OFFSET

        arr = in_line.split()

        # t idx M nM [list m-idxs], x, y, z, w, phi, ecc, [aux-channels]
        self.t_idx = int(arr[0]) - self._dt_offset
        if self.t_idx < 0:
            raise ValueError(f'negative t_idx={self.t_idx} in {in_line}. Skipping cell expected downstream.')

        self.t = self.t_idx * self._dt
        self.idx = int(arr[1])
        # is_m = int(arr[2])  # is multiple
        n_m = int(arr[3])
        arr = arr[4:]
        self.m_idx = [int(ai) for ai in arr[:n_m]]
        arr = arr[n_m:]
        arr = [float(ai) for ai in arr]

        self.x, self.y, self.z, self.w, self.phi, self.ecc, *self.aux_ch = arr
        self._sub_cells = []

    def is_single(self):
        return len(self.m_idx) == 0

    def is_merged(self):
        return len(self.m_idx) > 1

    def is_subcell(self):
        return len(self.m_idx) == 1

    def get_subcells(self):
        return self._sub_cells

    def get_subcells_idx(self):
        return self.m_idx

    def clear_subcells_idx(self):
        self.m_idx.clear()

    def clear_subcells(self):
        self._sub_cells.clear()

    def add_subcells_idx(self, idx):
        self.m_idx.append(idx)

    def add_subcell(self, sub_cell: Cell):
        return self._sub_cells.append(sub_cell)

    def min_r_to(self, other_cell: Cell):
        """
        smallest distance between subcells ot whole cell
        """
        r_s = [[self.x, self.y, self.z]] + [[c.x, c.y, c.z] for c in self._sub_cells]
        r_c = [[other_cell.x, other_cell.y, other_cell.z]] + [[c.x, c.y, c.z] for c in other_cell._sub_cells]
        r_s = np.array(r_s)[np.newaxis]
        r_c = np.array(r_c)[:, np.newaxis]

        dr_self_to_other = r_s - r_c  # vector
        # print(dr_self_to_other)
        dr2 = dr_self_to_other * dr_self_to_other
        dr2 = dr2.sum(axis=2)  # along x,y,z
        dr2_min = dr2.min()
        dr_min = sqrt(dr2_min)
        return dr_min

    def get_w_list(self):
        return [self.w] + [c.w for c in self._sub_cells]

    def get_m(self):
        return max(1, len(self._sub_cells))

    def get_r(self):
        # warning: assumed 2D
        return self.x, self.y

    @staticmethod
    def cell_from_subcells(subcells: list, subcell_idxs: list, c_idx: int):
        assert len(subcells), "can't make other_cell from no subcells"
        assert len(subcells) == len(subcell_idxs), "idx and subcells inconsistent"

        cell: Cell = copy.deepcopy(subcells[0])
        cell._sub_cells = [copy.deepcopy(sc) for sc in subcells]
        cell.m_idx = subcell_idxs.copy()

        cell.idx = c_idx
        cell.fill_avg_from_subcells()
        return cell

    def fill_avg_from_subcells(self):
        """fills all params as weighted by w from subcells"""
        self.x = 0
        self.y = 0
        self.z = 0
        self.w = 0
        self.phi = 0
        self.ecc = 0
        sub_cells = self._sub_cells

        assert len(sub_cells)

        n_aux = len(sub_cells[0].aux_ch)
        self.aux_ch = [0. for _ in range(n_aux)]

        for sc in sub_cells:
            w = sc.w
            self.w += w
            self.x += w * sc.x
            self.y += w * sc.y
            self.z += w * sc.z

            for i in range(n_aux):
                self.aux_ch[i] += w * sc.aux_ch[i]

        self.x /= self.w
        self.y /= self.w
        self.z /= self.w

        for i in range(n_aux):
            self.aux_ch[i] /= self.w

        xs = [sc.x for sc in sub_cells]
        ys = [sc.y for sc in sub_cells]
        ws = [sc.w for sc in sub_cells]
        eccs = [sc.ecc for sc in sub_cells]
        phis = [sc.phi for sc in sub_cells]

        self.ecc, self.phi = ellipse_fit_ellipsoids(xs, ys, ws, eccs, phis)
        self.phi = angle_to_mpi2ppi2(self.phi)

    # noinspection PyPep8Naming
    def get_DoC(self):
        # raw_mean raw_var cell_prob_mean cell_prob_var diap_prob_mean diap_prob_var diap_to_cell_mean diap_to_cell_var
        assert len(self.aux_ch) >= cfgm.EXPECTED_NUM_AUX_CHANNELS

        # return self.aux_ch[-2]/255.
        return self.aux_ch[cfgm.DOC_AUX_CHANNEL] / 255.


def cell_r_ofs_t(cell):
    return cell.x, cell.y, cell.t_idx


def load_cells(file_name: str) -> tuple[dict[int, list[Cell]], dict[int, list[Cell]]]:
    """
    Load cells from text file
    Args:
        file_name (str): filename

    Returns:
        (dict, dict): dictionaries of merged cells and raw cells

    """
    cells = {}
    with open(file_name, 'rt') as f:
        while True:
            s = f.readline()
            if not s:
                break

            try:
                cell = Cell(s)
            except ValueError:
                continue  # skip lines with time < cfgm.DT_OFFSET

            if cell.t_idx not in cells:
                cells[cell.t_idx] = []

            assert (len(cells[cell.t_idx]) == cell.idx)

            cells[cell.t_idx].append(cell)

    cells_merged = {}
    cells_full = {}  # both subcells of merged and filled merged present in the array

    for t, t_cells in cells.items():
        t_cells_single = []
        t_cells_merged = []
        t_cells_subcell = []

        local_subcell_idx_to_local_midx = {}

        for cell in t_cells:
            if cell.is_single():
                cell = copy.deepcopy(cell)
                t_cells_single.append(cell)

            elif cell.is_merged():
                mcell = copy.deepcopy(cell)
                subcells_idx = cell.get_subcells_idx()

                local_mcell_idx = len(t_cells_merged)
                for sidx in subcells_idx:
                    scell = t_cells[sidx]
                    scell = copy.deepcopy(scell)
                    mcell.add_subcell(scell)

                    local_subcell_idx_to_local_midx[len(t_cells_subcell)] = local_mcell_idx
                    t_cells_subcell.append(scell)

                t_cells_merged.append(mcell)

        n_single = len(t_cells_single)
        n_merged = len(t_cells_merged)

        ofs_merged = n_single
        ofs_subcell = ofs_merged + n_merged

        for cell_idx, cell in enumerate(t_cells_single):
            cell.idx = cell_idx

        for cell_idx, cell in enumerate(t_cells_merged):
            cell.idx = cell_idx + ofs_merged
            cell.clear_subcells_idx()

        for cell_idx, cell in enumerate(t_cells_subcell):
            sc_idx = cell_idx + ofs_subcell
            cell.idx = sc_idx

            loc_m_idx = local_subcell_idx_to_local_midx[cell_idx]
            t_cells_merged[loc_m_idx].add_subcells_idx(sc_idx)

            m_idx = loc_m_idx + ofs_merged
            cell.clear_subcells_idx()
            cell.add_subcells_idx(m_idx)

        t_cells_sm = t_cells_single + t_cells_merged
        t_cells_smsc = t_cells_sm + t_cells_subcell

        # verify all indexes
        for cell_idx, cell in enumerate(t_cells_sm):
            assert (cell.idx == cell_idx)

        for cell_idx, cell in enumerate(t_cells_smsc):
            assert (cell.idx == cell_idx)
            if cell.is_merged():
                for sc_idx in cell.get_subcells_idx():
                    scell = t_cells_smsc[sc_idx]
                    assert (len(scell.get_subcells_idx()) == 1)
                    assert (scell.get_subcells_idx()[0] == cell_idx)

        cells_merged[t] = t_cells_sm
        cells_full[t] = t_cells_smsc

    return cells_merged, cells_full


def _get_xyztiduidx_aux(cell: Cell, uidx):
    uidx += 1
    xyztiduidx = [cell.x, cell.y, cell.z, cell.t_idx, cell.t, cell.idx, uidx]
    multiplicity = max(1, len(cell.get_subcells()))
    aux = [multiplicity, cell.w, cell.phi, cell.ecc,
           *cell.aux_ch[::2]]  # only mean values from aux (contains mean & sgm^2)
    return xyztiduidx, aux, uidx


def _save_cell_for_display(cells_ds: dict[int, list[Cell]], file_name: str, save_subcells=False) -> None:
    """
    Saves cell dataset for visualisation with napari diplay
    Args:
        cells_ds (dict):
        file_name (str):
        save_subcells (bool): save subcells

    Returns:

    """
    # 1. fill dictionary
    uidx = -1  # only container related

    cells_xyz_t_id_uidx = []
    cells_aux_pars = []

    for t, cells in cells_ds.items():
        cell: Cell or None = None
        for cell in cells:
            if save_subcells:
                sub_cells = [cell] if cell.is_single() else cell.get_subcells()
                for scell in sub_cells:
                    xyztiduidx, aux, uidx = _get_xyztiduidx_aux(scell, uidx)
                    cells_xyz_t_id_uidx.append(xyztiduidx)
                    cells_aux_pars.append(aux)
            else:
                xyztiduidx, aux, uidx = _get_xyztiduidx_aux(cell, uidx)
                cells_xyz_t_id_uidx.append(xyztiduidx)
                cells_aux_pars.append(aux)

    n_ch = len(cell.aux_ch) // 2
    channel_names = ['raw_px', 'cell_p', 'diap_p', 'diap_p/cell_p'] + [f'fluor_{i}' for i in range(n_ch - 4)]
    cells_xyz_map = {0: 'x', 1: 'y', 2: 'z',
                     3: 't_idx', 4: 't', 5: 'c_idx',
                     6: 'uidx'}
    cells_aux_map = {0: 'mult', 1: 'w', 2: 'phi', 3: 'ecc'}
    for idx, name in enumerate(channel_names):
        cells_aux_map[idx + 4] = name

    dataset = {
        'cells_xyz': cells_xyz_t_id_uidx,
        'cells_aux': cells_aux_pars,
        'xyz_map': cells_xyz_map,
        'aux_map': cells_aux_map
    }
    with open(file_name, 'wb') as f:
        pickle.dump(dataset, f)


def save_cell_for_display(cells_ds: dict[int, list[Cell]], path: str = '') -> None:
    path_m = os.path.join(path, 'disp_cells_m.pckl')
    path_s = os.path.join(path, 'disp_cells_s.pckl')
    _save_cell_for_display(cells_ds=cells_ds, save_subcells=False, file_name=path_m)
    _save_cell_for_display(cells_ds=cells_ds, save_subcells=True, file_name=path_s)


class ObjContainer2D:
    """
    2D container of xy-blocks for Cell retrieval by proximity
    A grid of blocks is created with elemnts in corresponding block
    """

    def __init__(self, objects_in_r2: Sequence, block_xy_sz: float = 20):
        """

        Args:
            objects_in_r2: list of all objects, must have `x` and `y` attributes
            block_xy_sz: size of each block
        """
        c_xy = [[cell.x, cell.y] for cell in objects_in_r2]
        if len(c_xy) == 0:
            c_xy = [[0, 0]]
        c_xy = np.array(c_xy)
        self.x_min = np.min(c_xy[:, 0])
        self.x_max = np.max(c_xy[:, 0])
        self.y_min = np.min(c_xy[:, 1])
        self.y_max = np.max(c_xy[:, 1])
        self.n_x = int(ceil((self.x_max - self.x_min) / block_xy_sz)) + 1
        self.n_y = int(ceil((self.y_max - self.y_min) / block_xy_sz)) + 1
        self.block_xy_sz = block_xy_sz

        self.map = [[[] for _ in range(self.n_x)] for __ in range(self.n_y)]

        for idx, cell in enumerate(objects_in_r2):
            xy = cell.x, cell.y
            ix, iy = self._get_xy_idx(xy)
            self.map[iy][ix].append(idx)

        self.cells = objects_in_r2

    def _get_xy_idx(self, r: Sequence[float]) -> tuple[int, int]:
        dx = r[0] - self.x_min
        dy = r[1] - self.y_min
        i_x = int(dx // self.block_xy_sz)
        i_y = int(dy // self.block_xy_sz)
        i_x = min(self.n_x - 1, max(0, i_x))
        i_y = min(self.n_y - 1, max(0, i_y))
        return i_x, i_y

    def _list_obj_around(self, r: tuple, dr: float):
        r_min, r_max = (r[0] - dr, r[1] - dr), (r[0] + dr, r[1] + dr)
        idx_min, idx_max = self._get_xy_idx(r_min), self._get_xy_idx(r_max)

        lst = []
        for iy in range(idx_min[1], idx_max[1] + 1):
            for ix in range(idx_min[0], idx_max[0] + 1):
                lst.extend(self.map[iy][ix])

        return lst

    def obj_around_obj(self, object_in_r2, dr: float):
        return self.obj_around((object_in_r2.x, object_in_r2.y), dr)

    def obj_around(self, r, dr):
        lst = self._list_obj_around(r, dr)
        ol = []
        for il in lst:
            ol.append(self.cells[il])
        return ol

    def display(self, s=1):
        w = (self.x_max - self.x_min) / 1000 * s
        h = (self.y_max - self.y_min) / 1000 * s
        plt.figure(figsize=(w, h))
        x = [cell.x for cell in self.cells]
        y = [cell.y for cell in self.cells]
        t = [cell.t_idx for cell in self.cells]
        plt.scatter(x, y, c=t)

    def crop(self, r_min, r_max):
        cell_list = []

        r_min, r_max = list(r_min), list(r_max)
        idx_min, idx_max = self._get_xy_idx(r_min), self._get_xy_idx(r_max)

        y_min, y_max = min(idx_min[1], idx_max[1]), max(idx_min[1], idx_max[1])
        for iy in range(y_min, y_max + 1):

            x_min, x_max = min(idx_min[0], idx_max[0]), max(idx_min[0], idx_max[0])
            for ix in range(x_min, x_max + 1):
                # print((idx_min[1], idx_max[1]+1), (idx_min[0], idx_max[0]+1))
                idxs = self.map[iy][ix]
                # print(idxs)
                for i in idxs:
                    cell = self.cells[i]
                    if r_min[0] <= cell.x <= r_max[0] and r_min[1] <= cell.y <= r_max[1]:
                        cell_list.append(cell)

        for idx, cell in enumerate(cell_list):
            cell.idx = idx

        oc = ObjContainer2D(cell_list, self.block_xy_sz)
        return oc


class Stack:
    """
    sequence of timeframes based on ObjContainer2D
    """

    def __init__(self, t_cells_dict: dict[int, list[Cell]]):
        self.st = {t: ObjContainer2D(cells) for t, cells in t_cells_dict.items()}
        self.x_min, self.x_max, self.y_min, self.y_max = 0, 0, 0, 0
        self.update_bpundaries()

    def update_bpundaries(self):
        xy_mm = [[sti.x_min, sti.x_max, sti.y_min, sti.y_max] for sti in self.st.values() if
                 not (sti.x_min == sti.x_max == sti.y_min == sti.y_max == 0)]
        xy_mm = np.array(xy_mm)
        self.x_min, self.x_max, self.y_min, self.y_max = (xy_mm[:, 0].min(),
                                                          xy_mm[:, 1].max(),
                                                          xy_mm[:, 2].min(),
                                                          xy_mm[:, 3].max()
                                                          )

    def f_fid_vol(self):
        return self.x_min, self.x_max, self.y_min, self.y_max

    @staticmethod
    def in_range(cell, rng):
        return True if rng is None else (rng[0] <= cell.x <= rng[1] and rng[2] <= cell.y <= rng[3])

    def display(self, s=1, rng=None, links=None):
        if rng:
            w, h = (rng[1] - rng[0]) / 1000 * s, (rng[3] - rng[2]) / 1000 * s
        else:
            w = (self.x_max - self.x_min) / 1000 * s
            h = (self.y_max - self.y_min) / 1000 * s
        plt.figure(figsize=(w, h))

        x = []
        y = []
        t = []
        m = []
        for sti in self.st.values():
            for cell in sti.cells:
                if self.in_range(cell, rng):
                    x.append(cell.x)
                    y.append(cell.y)
                    t.append(cell.t)
                    m.append(15 * cell.get_m())

        plt.scatter(x, y, c=t, s=m)

        if links:
            cols = ['k', 'b', 'g', 'r']
            lines = []
            for (t1, idx1), (t2, idx2), m in links:
                cell1 = self.st[t1].cells[idx1]
                cell2 = self.st[t2].cells[idx2]
                if self.in_range(cell1, rng) or self.in_range(cell2, rng):
                    lines.extend([(cell1.x, cell2.x), (cell1.y, cell2.y), cols[m]])
            plt.plot(*lines)

    def display_list(self, t_idx_arr, s=1):
        x = []
        y = []
        t = []
        for tidx, cidx in t_idx_arr:
            cell = self.st[tidx].cells[cidx]
            x.append(cell.x)
            y.append(cell.y)
            t.append(cell.t_idx)

        def rng(arr):
            return np.max(arr) - np.min(arr)
        w = max(1, rng(x) / 1000 * s)
        h = max(1, rng(y) / 1000 * s)
        # print(x, y, t, w, h)
        plt.figure(figsize=(w, h))
        if len(x) > 1:
            plt.scatter(x, y, c=t)
        else:
            plt.plot(x, y)
        plt.show()

    def crop(self, r_min, r_max, t_min, t_max):
        st = copy.deepcopy(self)
        st.st = {t: o.crop(r_min, r_max) for t, o in st.st.items() if t_min <= t <= t_max}
        st.update_bpundaries()
        return st


# merge subceells withing radius
def get_cell_merge_groups(stack, dr_max):
    """
    Returns:
        dict(t_idx->set(c_idx)): cells (single or merged) to be merged
    """

    def get_loc_idx(cell_idx, c_idx_to_loc_idx, loc_idx):
        cell_loc_idx = c_idx_to_loc_idx.get(cell_idx, None)
        if cell_loc_idx is None:
            cell_loc_idx = loc_idx[0]
            loc_idx[0] += 1
            c_idx_to_loc_idx[cell_idx] = cell_loc_idx
        return cell_loc_idx

    merge_groups = {}
    dr2_max = dr_max ** 2

    for t_idx, st in stack.st.items():
        cells = st.cells

        links_u = []
        c_idx_to_loc_idx = {}
        loc_idx = [0]
        for cell_0_idx, cell_0 in enumerate(cells):
            if cell_0.is_subcell():
                continue

            x0, y0 = cell_0.get_r()

            for cell_1_idx in st._list_obj_around((x0, y0), dr_max):
                if cell_1_idx == cell_0_idx:
                    continue

                cell_1 = cells[cell_1_idx]
                if cell_1.is_subcell():
                    continue

                x1, y1 = cell_1.get_r()
                dr2 = (x1 - x0) ** 2 + (y1 - y0) ** 2
                if dr2 > dr2_max:
                    continue

                cell_0_loc_idx = get_loc_idx(cell_0_idx, c_idx_to_loc_idx, loc_idx)
                cell_1_loc_idx = get_loc_idx(cell_1_idx, c_idx_to_loc_idx, loc_idx)
                links_u.append((cell_0_loc_idx, cell_1_loc_idx))

        groups = []
        if len(links_u):
            row = []
            col = []
            val = []
            for f, t in links_u:
                row.append(f)
                col.append(t)
                val.append(1)

            n_cells = loc_idx[0]
            graph = csr_matrix((val, (row, col)), shape=(n_cells, n_cells))
            n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
            # print(t_idx, 'n_components=', n_components, labels, c_idx_to_loc_idx)

            ids = np.arange(n_cells)
            loc_idx_to_c_idx = {loc_idx: c_idx for c_idx, loc_idx in c_idx_to_loc_idx.items()}
            for gid in range(n_components):
                msk = labels == gid
                groups.append([loc_idx_to_c_idx[i] for i in ids[msk]])

        merge_groups[t_idx] = groups
    return merge_groups


def get_subcell_merge_groups(stack, dr_max):
    """
    Returns:
        tuple(
            dict(t_idx->set(c_idx_to_be_merged)),
            dict(t_idx->set(c_idx_of_m_cells_to_be_removed))
            ): other_cell indexes (single or subcells of a merged) to be merged
            and other_cell indexes of merged cells to be removed (M cells of the used subcells)
    """

    merge_groups = get_cell_merge_groups(stack, dr_max)
    merge_groups_sc = {}
    remove_mc = {}

    for t_idx, groups in merge_groups.items():
        st = stack.st[t_idx]
        cells = st.cells

        groups_sc = []
        used_m_cells = []

        for group in groups:
            group_sc = []

            for c_idx in group:
                cell = cells[c_idx]
                assert (cell.is_subcell() is False)  # only merged and single cells expected in the input

                if cell.is_single():
                    group_sc.append(c_idx)
                elif cell.is_merged():
                    for s_c_idx in cell.get_subcells_idx():
                        group_sc.append(s_c_idx)
                    used_m_cells.append(c_idx)
                else:
                    raise ValueError("Unexpected other_cell type, can't proceed")

            groups_sc.append(group_sc)

        merge_groups_sc[t_idx] = groups_sc
        remove_mc[t_idx] = used_m_cells

    return merge_groups_sc, remove_mc


def merge_cell_groups(cells, merge_groups_sc, used_mc):
    """
    merges single & subcells according to merge groups, and returns cells as timeframe dicts for s+m and s+m+sc
    Args:
        cells (dict(t_idx->list(Cell))):             s+m+sc cells maps
        merge_groups_sc (dict(t_idx->list(list(c_idx)))): merge grpups
        used_mc (dict(t_idx->list(c_idx))):             merged cells whose subscells will be used for new M cells
    """

    cells_merged = {}
    cells_full = {}  # both subcells of merged and filled merged present in the array

    for t_idx, t_cells in cells.items():
        merge_groups_t_sc = merge_groups_sc[t_idx]
        merge_sc_t_set = {c_idx for merge_group_t_sc in merge_groups_t_sc for c_idx in merge_group_t_sc}
        used_mc_t_set = set(used_mc[t_idx])

        t_cells_single = []
        t_cells_merged = []
        t_cells_subcell = []

        # orig_mcell_idx_to_local_mcell_idx = {}
        # orig_subcell_idx_to_local_subcell_idx = {}

        local_subcell_idx_to_local_midx = {}

        n_single_orig = np.sum([cell.is_single() for cell in t_cells])
        n_scells_orig = np.sum([cell.is_subcell() for cell in t_cells])

        for c_idx, cell in enumerate(t_cells):
            assert (c_idx == cell.idx)

            if c_idx in used_mc_t_set or c_idx in merge_sc_t_set:  # the one to be remerged will be filled on next step
                continue

            if cell.is_single():
                cell = copy.deepcopy(cell)
                t_cells_single.append(cell)

            elif cell.is_merged():
                mcell = copy.deepcopy(cell)
                mcell.clear_subcells()

                subcells_idx = cell.get_subcells_idx()

                local_mcell_idx = len(t_cells_merged)
                for sidx in subcells_idx:
                    assert (sidx not in merge_sc_t_set)  # not one of reserved subcells
                    scell = t_cells[sidx]
                    assert (sidx == scell.idx)

                    scell = copy.deepcopy(scell)
                    mcell.add_subcell(scell)

                    local_scell_idx = len(t_cells_subcell)
                    # orig_subcell_idx_to_local_subcell_idx[sidx] = local_scell_idx
                    local_subcell_idx_to_local_midx[local_scell_idx] = local_mcell_idx
                    t_cells_subcell.append(scell)

                t_cells_merged.append(mcell)
                # orig_mcell_idx_to_local_mcell_idx[c_idx] = local_mcell_idx

        # create new m-cells according to merge groups
        for merge_group_t_sc in merge_groups_t_sc:
            subcells_idx = merge_group_t_sc
            subcells = [t_cells[c_idx] for c_idx in subcells_idx]
            assert (len(subcells))
            local_mcell_idx = len(t_cells_merged)
            mcell = Cell.cell_from_subcells(subcells, subcells_idx, local_mcell_idx)

            for sidx in subcells_idx:
                scell = t_cells[sidx]
                assert (sidx == scell.idx)

                scell = copy.deepcopy(scell)

                local_scell_idx = len(t_cells_subcell)
                local_subcell_idx_to_local_midx[local_scell_idx] = local_mcell_idx
                t_cells_subcell.append(scell)

            t_cells_merged.append(mcell)

        n_single = len(t_cells_single)
        n_merged = len(t_cells_merged)
        n_scells = len(t_cells_subcell)

        assert ((n_single + n_scells) == (n_single_orig + n_scells_orig))  # should be constant

        ofs_merged = n_single
        ofs_subcell = ofs_merged + n_merged

        for cell_idx, cell in enumerate(t_cells_single):
            cell.idx = cell_idx

        for cell_idx, cell in enumerate(t_cells_merged):
            cell.idx = cell_idx + ofs_merged
            cell.clear_subcells_idx()

        for cell_idx, cell in enumerate(t_cells_subcell):
            sc_idx = cell_idx + ofs_subcell
            cell.idx = sc_idx

            loc_m_idx = local_subcell_idx_to_local_midx[cell_idx]
            t_cells_merged[loc_m_idx].add_subcells_idx(sc_idx)

            m_idx = loc_m_idx + ofs_merged
            cell.clear_subcells_idx()
            cell.add_subcells_idx(m_idx)

        t_cells_sm = t_cells_single + t_cells_merged
        t_cells_smsc = t_cells_sm + t_cells_subcell

        # verify all indexes
        for cell_idx, cell in enumerate(t_cells_sm):
            assert (cell.idx == cell_idx)

        for cell_idx, cell in enumerate(t_cells_smsc):
            assert (cell.idx == cell_idx)
            if cell.is_merged():
                for sc_idx in cell.get_subcells_idx():
                    scell = t_cells_smsc[sc_idx]
                    assert (len(scell.get_subcells_idx()) == 1)
                    assert (scell.get_subcells_idx()[0] == cell_idx)

        cells_merged[t_idx] = t_cells_sm
        cells_full[t_idx] = t_cells_smsc

    return cells_merged, cells_full


def plot_neighbour_cell_count_distrinution(stack, dr_max=50):
    drs = []
    for t_idx, st in stack.st.items():
        for cell_0 in st.cells:
            if cell_0.is_subcell():
                continue

            x0, y0 = cell_0.get_r()

            for cell_1 in st.obj_around_obj(cell_0, dr_max):
                if cell_1 == cell_0:
                    continue
                if cell_1.is_subcell():
                    continue

                x1, y1 = cell_1.get_r()
                dr = sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
                if dr > dr_max:
                    continue

                drs.append(dr)
    plt.hist(drs, 200)
    plt.savefig('neighbour_cell_count_distrinution.png')
    plt.close()


def merge_nearby_cells(stack, cells, dr_max):
    merge_groups_sc, remove_mc = get_subcell_merge_groups(stack, dr_max)
    cells_merged_n, cells_full_n = merge_cell_groups(cells, merge_groups_sc, remove_mc)
    return cells_merged_n, cells_full_n


# do linking
def make_links(max_dr, eps, stack, max_dt=2, taketop=3):  # dr in um
    links = []
    times = sorted(stack.st.keys())

    for t in times:
        frame_f = stack.st.get(t, None)  # from
        if frame_f is None:
            continue

        for cell_f in frame_f.cells:
            for dt in range(1, max_dt + 1):
                links_dt = []
                t_to = t + dt
                frame_t = stack.st.get(t_to, None)  # to
                if frame_t is None:
                    continue

                max_r = dt * max_dr
                for cell_t in frame_t.obj_around_obj(cell_f, max_r + eps):
                    dr = cell_f.min_r_to(cell_t)
                    if dr < max_r:
                        links_dt.append([(t, cell_f.idx), (t_to, cell_t.idx)])
                # if dt==3 and len(links_dt) > 3:
                #    print(get_links_w(stack, links_dt))

                links.extend(links_dt)
    return links


def fnll_w(w1, w2):
    return 2 * (w2 - w1) / (w1 + w2)


# solve linking
def get_links_w(stack, links, w_nn):
    links_nll = []
    for (tf1, idx1), (tf2, idx2) in links:
        c1 = stack.st[tf1].cells[idx1]
        c2 = stack.st[tf2].cells[idx2]

        dr = c1.min_r_to(c2)
        dt = abs(c2.t - c1.t)
        dt_idx = abs(c2.t_idx - c1.t_idx)

        v = dr / dt

        ws1, ws2 = c1.get_w_list(), c2.get_w_list()

        nll_v = (v - cfgm.MEAN_V) ** 2 / cfgm.STD_V2  # get constants

        nlls_w = [fnll_w(w1, w2) ** 2 / cfgm.STD_REL_DW2 for w1 in ws1 for w2 in ws2]
        # print(nlls_w)
        nll_w = min(nlls_w)

        nll = 0.5 * (nll_w + nll_v)
        nll += (dt_idx - 1) * w_nn
        links_nll.append(nll)
    return links_nll


def discard_bad_links(links, links_nll):
    nll_max = cfgm.W_NC_0
    links = [link for link, w in zip(links, links_nll) if w < nll_max]
    links_nll = [w for w in links_nll if w < nll_max]
    return links, links_nll


class NodeStruct_M:
    """
    Single element for the graph structure for the constrained optimization
    for segment candidate search

    """

    def __init__(self, idx, m, w_d_mult):
        # c sync flags
        self.b_nc_l = None
        self.b_nc_r = None

        # c - connection variable
        self.c_nc_l = None
        self.c_nc_r = None

        self.c_l = []  # each in range [0-m]
        self.c_r = []

        self.cw_l = []  # each in range [0-1], cw_l[i] == (c_l[i]>0)
        self.cw_r = []

        self.dm_lr = None  # multiplicity_r - multiplicity_l
        self.adm_lr = None  # abs(dm_lr) if both sides connected else 0

        self.links_l = []  # elements - ([node_on_the_left_idx, r_link_(of_left_node)_idx])
        self.links_r = []  # elements - ([node_on_the_rght_idx, l_link_(of_rght_node)_idx])

        self.w_r = []  # weihgts of links on the right side. left are on the right side of connected nodes.
        self.w_nc_l = 0
        self.w_nc_r = 0

        self.idx = idx
        self.m = m
        self.w_d_ml = w_d_mult
        self.solved = None

    def add_r_link_to_node(self, node, w):
        self_r_link_idx = len(self.links_r)
        node_l_link_idx = len(node.links_l)

        self.links_r.append((node.idx, node_l_link_idx))
        node.links_l.append((self.idx, self_r_link_idx))

        self.w_r.append(w)

    def set_nc_weight(self, w_nc_l, w_nc_r):
        self.w_nc_l = w_nc_l
        self.w_nc_r = w_nc_r

    def _get_c_on_the_right(self, r_idx, nodes):
        link = self.links_r[r_idx]
        r_node_idx, l_idx = link
        r_node = nodes[r_node_idx]
        c_r_node_l_link = r_node.c_l[l_idx]
        return c_r_node_l_link

    def setup_vars(self, model):
        self.b_nc_l = model.NewBoolVar(f'n{self.idx}_b_nc_l')
        self.b_nc_r = model.NewBoolVar(f'n{self.idx}_b_nc_r')

        self.c_nc_l = model.NewIntVar(0, 1, f'n{self.idx}_c_nc_l')
        self.c_nc_r = model.NewIntVar(0, 1, f'n{self.idx}_c_nc_r')

        self.c_l = [model.NewIntVar(0, self.m, f'n{self.idx}_c_l{i}') for i in range(len(self.links_l))]
        self.c_r = [model.NewIntVar(0, self.m, f'n{self.idx}_c_r{i}') for i in range(len(self.links_r))]
        self.cw_l = [model.NewIntVar(0, 1, f'n{self.idx}_cw_l{i}') for i in range(len(self.links_l))]
        self.cw_r = [model.NewIntVar(0, 1, f'n{self.idx}_cw_r{i}') for i in range(len(self.links_r))]

        self.dm_lr = model.NewIntVar(-self.m, self.m, f'n{self.idx}_dm_lr')  # multiplicity_r - multiplicity_l
        self.adm_lr = model.NewIntVar(0, self.m, f'n{self.idx}_adm_lr')  # abs(dm_lr) if both sides connected else 0

    def setup_node_constraints(self, model):
        # left side
        if len(self.c_l):
            sum_links_l = cp_model.LinearExpr.Sum(self.c_l)
            model.Add(self.c_nc_l == 1).OnlyEnforceIf(self.b_nc_l)
            model.Add(sum_links_l == 0).OnlyEnforceIf(self.b_nc_l)
            model.Add(self.c_nc_l == 0).OnlyEnforceIf(self.b_nc_l.Not())
            model.Add(sum_links_l >= 1).OnlyEnforceIf(self.b_nc_l.Not())

            model.Add((sum_links_l + self.c_nc_l) >= 1)
            model.Add((sum_links_l + self.c_nc_l) <= self.m)

            for ci, cwi in zip(self.c_l, self.cw_l):
                model.Add(ci > 0).OnlyEnforceIf(cwi)
                model.Add(ci == 0).OnlyEnforceIf(cwi.Not())

        else:
            model.Add(self.c_nc_l == 1)
            model.Add(self.b_nc_l == 1)

        # right side
        if len(self.c_r):
            sum_links_r = cp_model.LinearExpr.Sum(self.c_r)
            model.Add(self.c_nc_r == 1).OnlyEnforceIf(self.b_nc_r)
            model.Add(sum_links_r == 0).OnlyEnforceIf(self.b_nc_r)
            model.Add(self.c_nc_r == 0).OnlyEnforceIf(self.b_nc_r.Not())
            model.Add(sum_links_r >= 1).OnlyEnforceIf(self.b_nc_r.Not())

            model.Add((sum_links_r + self.c_nc_r) >= 1)
            model.Add((sum_links_r + self.c_nc_r) <= self.m)

            for ci, cwi in zip(self.c_r, self.cw_r):
                model.Add(ci > 0).OnlyEnforceIf(cwi.Not().Not())  # might be impl trick or a joke
                model.Add(ci == 0).OnlyEnforceIf(cwi.Not())
        else:
            model.Add(self.c_nc_r == 1)
            model.Add(self.b_nc_r == 1)

        if len(self.c_l) and len(self.c_r):
            sum_links_l = cp_model.LinearExpr.Sum(self.c_l)
            sum_links_r = cp_model.LinearExpr.Sum(self.c_r)

            # print('\n',model.Validate())
            model.Add(self.dm_lr == sum_links_r - sum_links_l).OnlyEnforceIf(self.b_nc_l.Not()).OnlyEnforceIf(
                self.b_nc_r.Not())
            model.Add(self.dm_lr == 0).OnlyEnforceIf(self.b_nc_l)
            model.Add(self.dm_lr == 0).OnlyEnforceIf(self.b_nc_r)

            model.AddAbsEquality(self.adm_lr, self.dm_lr)
            # print(model.Validate())

    def setup_internode_constraints(self, model, nodes):
        for r_idx, c_r_i in enumerate(self.c_r):
            c_l_j = self._get_c_on_the_right(r_idx, nodes)
            model.Add(c_r_i == c_l_j)

    def get_node_loss_vars_weights(self, to_int_factor):
        vs = [self.c_nc_l, self.c_nc_r, self.adm_lr] + self.cw_r
        ws = [self.w_nc_l, self.w_nc_r, self.w_d_ml] + self.w_r

        ws = [int(wi * to_int_factor) for wi in ws]

        return vs, ws

    def fill_solved(self, solver):
        self.solved = copy.copy(self)

        self.solved.b_nc_l = solver.Value(self.b_nc_l)
        self.solved.b_nc_r = solver.Value(self.b_nc_r)
        self.solved.c_nc_l = solver.Value(self.c_nc_l)
        self.solved.c_nc_r = solver.Value(self.c_nc_r)
        self.solved.c_l = [solver.Value(ci) for ci in self.c_l]
        self.solved.c_r = [solver.Value(ci) for ci in self.c_r]

        self.solved.cw_l = [solver.Value(ci) for ci in self.cw_l]
        self.solved.cw_r = [solver.Value(ci) for ci in self.cw_r]

        self.solved.dm_lr = solver.Value(self.dm_lr)
        self.solved.adm_lr = solver.Value(self.adm_lr)

    def get_solved_links(self):
        links = []
        if self.solved.b_nc_l:
            links.append((-2, self.idx, 1))
        if self.solved.b_nc_r:
            links.append((self.idx, -1, 1))

        for link_idx, ci in enumerate(self.solved.c_l):
            if ci > 0:
                l_node_idx, _ = self.links_l[link_idx]
                links.append((l_node_idx, self.idx, ci))

        return links


class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""
    _last_hist = []

    def __init__(self):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__solution_count = 0
        # self.__solution_limit = 1000

        SolutionPrinter._last_hist = []

    def on_solution_callback(self):
        """Called at each new solution."""
        # print('Solution %i, time = %f s, objective = %i' %
        #      (self.__solution_count, self.WallTime(), self.ObjectiveValue()))
        SolutionPrinter._last_hist.append((self.__solution_count, self.WallTime(), self.ObjectiveValue()))
        self.__solution_count += 1

        # if self.__solution_count >= self.__solution_limit:
        #     print('Stop search after %i solutions' % self.__solution_limit)
        #     self.StopSearch()
        pass


def d_nearest_boundary(r, fid_vol):
    x, y = r
    x_min, x_max, y_min, y_max = fid_vol
    return min(abs(x - x_min), abs(x - x_max), abs(y - y_min), abs(y - y_max))


# border_exp_att_factor: (f(_BORDER_ATT_DIST)==1, f(_CELL_RADIUS)==1/9 )
def border_exp_att_factor(x):
    return (cfgm.EXPFT_A * exp(x - cfgm.BORDER_ATT_DIST) + cfgm.EXPFT_Y) if x < cfgm.BORDER_ATT_DIST else 1.


def w_nc_fid_vol_corr(w_nc, fid_vol, cell_r):
    d = d_nearest_boundary(cell_r, fid_vol)
    # _BORDER_ATT_DIST um - 9, _CELL_RADIUS um - 1
    # exp fit:

    f = border_exp_att_factor(d)
    w = w_nc * f
    w = min(cfgm.W_NC_0, max(1, w))
    return w


def solve(n, m_arr, cells_r, fid_vol, links, links_w, w_nc, w_dm):
    if len(links) == 0 or n < 2:
        return [], []

    nodes = []

    for idx, m in enumerate(m_arr):
        cell_r = cells_r[idx]
        w_nc_cell = w_nc_fid_vol_corr(w_nc, fid_vol, cell_r)
        node = NodeStruct_M(idx, m, w_dm)
        node.set_nc_weight(w_nc_cell, w_nc_cell)
        nodes.append(node)

    uid = n
    for link, w in zip(links, links_w):
        idx_from, idx_to = link

        node_from = nodes[idx_from]
        node_to = nodes[idx_to]

        node_from.add_r_link_to_node(node_to, w)

    model = cp_model.CpModel()

    for node in nodes:
        node.setup_vars(model)
    for node in nodes:
        node.setup_node_constraints(model)
    for node in nodes:
        node.setup_internode_constraints(model, nodes)

    vs = []
    ws = []
    for node in nodes:
        vsi, wsi = node.get_node_loss_vars_weights(1000)
        vs.extend(vsi)
        ws.extend(wsi)

    # for vi,wi in zip(vs, ws):
    #    print(vi, '\t',wi)

    # print(model.Validate())
    model.Minimize(cp_model.LinearExpr.ScalProd(vs, ws))
    # model.AddDecisionStrategy(vs, cp_model.CHOOSE_LOWEST_MIN, cp_model.SELECT_MIN_VALUE)
    #

    solver = cp_model.CpSolver()

    # >>>>>>>>>>>>>>>
    solver.parameters.num_search_workers = cfgm.NUM_WORKERS
    solver.parameters.max_time_in_seconds = cfgm.LINKING_SOLVER_TIMEOUT  # 60 * 15

    # solver.parameters.catch_sigint_signal = True

    # print('w', solver.parameters.num_search_workers)
    # print('a', solver.parameters.minimization_algorithm)
    # print('t', solver.parameters.max_time_in_seconds)
    # print('m', solver.parameters.max_memory_in_mb)

    # print(model.ModelStats())
    t1 = time.time()
    status = solver.Solve(model)

    # solution_printer = SolutionPrinter()
    # status = solver.SolveWithSolutionCallback(model, solution_printer)

    status_str = solver.StatusName(status)

    t2 = time.time()
    dt = t2 - t1
    if dt > cfgm.LONG_RUN_PRINT_TIMEOUT:
        print('\ndt=%.1fs:\n' % dt, model.ModelStats(), status_str)

    # print(status, cp_model.OPTIMAL)

    links = []

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        # if status == cp_model.OPTIMAL:
        for node in nodes:
            node.fill_solved(solver)

        for node in nodes:
            lnks = node.get_solved_links()
            links.extend(lnks)
        # for v in vs:
        #    print(v, ' = %i' % solver.Value(v))

        # for l in links:
        #    print(l)

    return links, nodes


# disjoint subsets solve:
def solve_disjoint(stack, links, links_w, w_nc, w_dm):
    idx_to_uid = {}
    uid_to_idx = []

    fid_vol = stack.f_fid_vol()

    uid = 0
    for t, st in stack.st.items():
        for cell in st.cells:
            idx = cell.idx
            idx_to_uid[(t, idx)] = uid
            uid_to_idx.append((t, idx))
            uid += 1

    links_u = [(idx_to_uid[idx_from], idx_to_uid[idx_to]) for idx_from, idx_to in links]

    row = []
    col = []
    val = []
    for f, t in links_u:
        row.append(f)
        col.append(t)
        val.append(1)

    graph = csr_matrix((val, (row, col)), shape=(uid, uid))
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    print('n_components=', n_components)
    # for gid in range(n_components):
    #    print (gid, ':', (labels==gid).sum())

    groups = []
    ids = np.arange(uid)
    for gid in range(n_components):
        msk = labels == gid
        groups.append(ids[msk])

    # print(groups)

    resolved_links_t_cidx_all = []
    for gr_idx, group in enumerate(tqdm(groups, desc='segment candidates', ascii=True)):
        # print(gr_idx, '/', n_components, end='\r')
        # stack.display_list([uid_to_idx[idx] for idx in group], s=20)
        gs = set(group)
        lidx = 0
        lidx_to_uid = []
        uid_to_lidx = {}
        for idx in group:
            lidx_to_uid.append(idx)
            uid_to_lidx[idx] = lidx
            lidx += 1

        m_vals = []
        cells_r = []
        for idx in group:
            t, cidx = uid_to_idx[idx]
            cell = stack.st[t].cells[cidx]
            m = cell.get_m()
            m_vals.append(m)
            r = cell.get_r()
            cells_r.append(r)

        llinks = []
        llinksw = []
        for link, w in zip(links_u, links_w):
            link_l, link_r = link
            if (link_l not in gs) and (link_r not in gs):
                continue
            llinks.append((uid_to_lidx[link_l], uid_to_lidx[link_r]))
            llinksw.append(w)

        # print(lidx, group, m_vals, llinks)
        resolved_links, *_ = solve(n=lidx, m_arr=m_vals, cells_r=cells_r, fid_vol=fid_vol,
                                   links=llinks, links_w=llinksw,
                                   w_nc=w_nc, w_dm=w_dm)

        resolved_links_guid = [(lidx_to_uid[l], lidx_to_uid[r], m) for l, r, m in resolved_links if l >= 0 and r >= 0]
        resolved_links_t_cidx = [(uid_to_idx[l], uid_to_idx[r], m) for l, r, m in resolved_links_guid]
        resolved_links_t_cidx_all.extend(resolved_links_t_cidx)
        # if len(resolved_links)>50:
        #    return resolved_links,resolved_links_guid,resolved_links_t_cidx

    return resolved_links_t_cidx_all


def solve_disjoint_cached(stack, links, links_w, w_nc, w_dm, use_cache):
    resolved_filename = f'resolved_links_fused_{cfgm.CELL_FUSE_RADIUS}.pkl' if cfgm.MERGE_CLOSE_CELLS else 'resolved_links.pkl'
    t1 = time.time()

    if os.path.exists(resolved_filename) and use_cache:  # on prod - always rerun all to avoid false data
        with open(resolved_filename, 'rb') as f:
            resolved_links = pickle.load(f)
    else:
        resolved_links = solve_disjoint(stack=stack,
                                        links=links, links_w=links_w,
                                        w_nc=w_nc, w_dm=w_dm)

        with open(resolved_filename, 'wb') as f:
            pickle.dump(resolved_links, f)

    t2 = time.time()
    print('time spent solving disjoint = %.2fs' % (t2 - t1))
    return resolved_links


class LinkedCell:
    class Link:
        def __init__(self, tgt_t_idx, tgt_c_idx, w, dt):
            self.tgt_t_idx = tgt_t_idx
            self.tgt_c_idx = tgt_c_idx

            self.w = w
            self.dt = dt
            self.m = 0
            self.selected = False

        def set_selected(self, selected=True, m=1):
            self.selected = selected
            self.m = m

    def __init__(self, t_idx, c_idx):
        self.t_idx = t_idx
        self.c_idx = c_idx

        self.l_links = []
        self.r_links = []
        self.links = [self.l_links, self.r_links]

        self.sel_links = [[], []]
        self.link_map = {}  # (tgt_t_idx,tgt_c_idx)->(lr, idx) in self.links

        self.rank = [[], []]
        self.l_to_r = None
        self.r_to_l = None

    def add_l_link(self, tgt_t_idx, tgt_c_idx, w, dt):
        return self.add_link(0, tgt_t_idx, tgt_c_idx, w, dt)

    def add_r_link(self, tgt_t_idx, tgt_c_idx, w, dt):
        return self.add_link(1, tgt_t_idx, tgt_c_idx, w, dt)

    def add_link(self, lr, tgt_t_idx, tgt_c_idx, w, dt):
        link = LinkedCell.Link(tgt_t_idx, tgt_c_idx, w, dt)

        idx = len(self.links[lr])
        self.links[lr].append(link)

        tgt_idx = (tgt_t_idx, tgt_c_idx)
        lr_idx = (lr, idx)
        self.link_map[tgt_idx] = lr_idx

    def add_selected_link(self, tgt_t_idx, tgt_c_idx, m):
        tgt_idx = (tgt_t_idx, tgt_c_idx)
        lr_idx = self.link_map.get(tgt_idx)
        if lr_idx is None:
            return False

        lr, idx = lr_idx
        link = self.links[lr][idx]
        link.set_selected(m)

    def get_all_chi2(self):
        """return list of chi2 of all links"""
        chi2 = [link.w for lrlinks in self.links for link in lrlinks]
        return chi2

    def get_selected_chi2(self):
        """return list of chi2 of selected links"""
        chi2 = [link.w for lrlinks in self.links for link in lrlinks if link.selected]
        return chi2

    def fill_struct(self):
        """so far generates ordered lists for weight, and default M node resolvings"""
        for lr in range(2):
            links = self.links[lr]
            n = len(links)

            if n == 0:
                self.rank[lr] = []
            else:
                sorted_idx = np.argsort([link.w for link in links])
                ranks = np.empty_like(sorted_idx)
                ranks[sorted_idx] = np.arange(n)
                self.rank[lr] = ranks

        # fill i & o lists according to ordered weights
        idx_sel_l = [idx for idx, link in enumerate(self.l_links) if link.selected]
        idx_sel_r = [idx for idx, link in enumerate(self.r_links) if link.selected]

        links_sel_l = [self.l_links[i] for i in idx_sel_l]
        links_sel_r = [self.r_links[i] for i in idx_sel_r]

        self.sel_links = [links_sel_l, links_sel_r]

        w_sel_l = [link.w for link in links_sel_l]
        w_sel_r = [link.w for link in links_sel_r]
        m_sel_l = [link.m for link in links_sel_l]
        m_sel_r = [link.m for link in links_sel_r]

        sorted_idx_l = np.argsort(w_sel_l)
        sorted_idx_r = np.argsort(w_sel_r)

        l_list = []
        r_list = []

        for lr_list, m_sel, w_sel, links_sel, srt_idx in zip([l_list, r_list],
                                                             [m_sel_l, m_sel_r],
                                                             [w_sel_l, w_sel_r],
                                                             [links_sel_l, links_sel_r],
                                                             [sorted_idx_l, sorted_idx_r]
                                                             ):
            for idx in srt_idx:
                m = m_sel[idx]
                lnk: LinkedCell.Link = links_sel[idx]
                for m_idx in range(m):
                    lr_list.append((lnk.tgt_t_idx, lnk.tgt_c_idx, m_idx))

        # make pairs
        l_to_r = {None: []}
        r_to_l = {None: []}

        n_common = min(len(l_list), len(r_list))
        for lnk_l, lnk_r in zip(l_list[:n_common], r_list[:n_common]):
            l_to_r[lnk_l] = lnk_r
            r_to_l[lnk_r] = lnk_l

        for lnk_l in l_list[n_common:]:
            l_to_r[lnk_l] = None
            r_to_l[None].append(lnk_l)

        for lnk_r in r_list[n_common:]:
            l_to_r[None].append(lnk_r)
            r_to_l[lnk_r] = None

        self.l_to_r = l_to_r
        self.r_to_l = r_to_l

    def get_starts(self):
        if self.l_to_r is None:
            raise ValueError('fill_struct has to be called prior to get_starts.')
        return self.l_to_r[None]

    def get_starts_clean(self):
        if self.l_to_r is None:
            raise ValueError('fill_struct has to be called prior to get_starts.')
        return self.l_to_r[None] if len(self.l_to_r) == 1 else []  # nothing on the left

    def get_next_node_link(self, l_lnk):  # l_lnk is left tgt_tc_idx & m_idx. returns l_lnk
        if self.l_to_r is None:
            raise ValueError('fill_struct has to be called prior to get_next_node_link.')

        r_lnk = self.l_to_r[l_lnk]
        return r_lnk

    def get_selected_rank(self):
        if self.l_to_r is None:
            raise ValueError('fill_struct has to be called prior to get_selected_rank.')

        rank = [self.rank[lr][l_idx] for lr in range(2) for l_idx, link in enumerate(self.links[lr]) if link.selected]
        return rank

    def _get_selected_lr_nodes(self, lr):
        tc_idx = []
        for lnk in self.sel_links[lr]:
            tc_idx.append((lnk.tgt_t_idx, lnk.tgt_c_idx))
        return tc_idx

    def get_selected_l_nodes(self):
        return self._get_selected_lr_nodes(0)

    def get_selected_r_nodes(self):
        return self._get_selected_lr_nodes(1)

    def is_m_node(self):
        f_l = np.sum([link.m for link in self.l_links if link.selected])  # n flow
        f_r = np.sum([link.m for link in self.r_links if link.selected])
        nc_l = np.sum([1 for link in self.l_links if link.selected])  # n connections
        nc_r = np.sum([1 for link in self.r_links if link.selected])
        return ((f_l > 0 and f_r > 0) and  # not a terminal node
                ((f_l != f_r) or  # change of flow
                 (nc_l != nc_r)  # change of conformation
                 )
                )

    def get_link_m(self, tgt_tc):
        if tgt_tc not in self.link_map:
            return 0
        lr, idx = self.link_map[tgt_tc]
        lnk = self.links[lr][idx]
        # assert(lnk.selected)
        return lnk.m


def get_linked_cell_map(stack, resolved_links, links, links_nll):
    lc_map = {}
    for t_idx, frm in stack.st.items():
        for c_idx, cell in enumerate(frm.cells):
            tc_idx = (t_idx, c_idx)
            lc = LinkedCell(*tc_idx)
            lc_map[tc_idx] = lc

    for link, w in zip(links, links_nll):
        tc_idx_l, tc_idx_r = link
        dt = tc_idx_r[0] - tc_idx_l[0]
        lc_map[tc_idx_l].add_r_link(*tc_idx_r, w, dt)
        lc_map[tc_idx_r].add_l_link(*tc_idx_l, w, dt)

    for link in resolved_links:
        tc_idx_l, tc_idx_r, m = link

        lc_map[tc_idx_l].add_selected_link(*tc_idx_r, m)
        lc_map[tc_idx_r].add_selected_link(*tc_idx_l, m)

    for lc in lc_map.values():
        lc.fill_struct()

    return lc_map


def get_disjointed_groups_links(idx_list, links, min_grp_size=0):
    idx_to_uid = {}
    uid_to_idx = idx_list

    uid = 0
    for uid, idx in enumerate(idx_list):
        idx_to_uid[idx] = uid
        uid += 1

    links_u = [(idx_to_uid[idx_from], idx_to_uid[idx_to]) for idx_from, idx_to, *_ in links]

    row = []
    col = []
    val = []
    for f, t in links_u:
        row.append(f)
        col.append(t)
        val.append(1)

    graph = csr_matrix((val, (row, col)), shape=(uid, uid))
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

    groups_uid = [[] for _ in range(n_components)]
    for uid, grp in enumerate(labels):
        groups_uid[grp].append(uid)

    groups = [[uid_to_idx[uid] for uid in group] for group in groups_uid if len(group) >= min_grp_size]
    return groups


def plot_lc_quality_info(lc_map):
    # fill chi2 and rank
    all_chi2 = []
    sel_chi2 = []
    sel_rank = []

    lc: LinkedCell
    for lc in lc_map.values():
        all_chi2.extend(lc.get_all_chi2())
        sel_chi2.extend(lc.get_selected_chi2())
        sel_rank.extend(lc.get_selected_rank())

    # plot chi2
    _ = plt.hist(all_chi2, 100, log=True)
    plt.title(r'All $\chi2$')
    plt.show()
    plt.close()

    _ = plt.hist(sel_chi2, 100, log=True)
    plt.title(r'$\chi2$ of resolved connections')
    plt.show()
    plt.close()

    _ = plt.hist(sel_rank, 100, log=True)
    plt.title('$Idx$ of resolved connections')
    plt.show()
    plt.close()


def get_track_starts(lc_map, lc_disjoint_groups):
    """
    makes track starting points. Three representations are made:
        * all track starts, ungrouped
        * all track starts, grouped by connected components
        * track starts, grouped by connected components, but only if there only 1 link attached to the start
    Args:
        lc_map: LinkedCell map
        lc_disjoint_groups:

    Returns:
        tuple[
        list[tuple[node_tc_idx, link],
        list[tuple[node_tc_idx, link],
        list[tuple[node_tc_idx, link]
        ), where link is a tuple (tgt_t_idx, tgt_c_idx, m_idx)

    """
    all_starts = []  # (node_tc_idx, link); lnk = (tgt_t_idx, tgt_c_idx, m_idx)

    lc: LinkedCell
    for tc_idx, lc in lc_map.items():
        cell_starts = lc.get_starts()
        all_starts.extend([(tc_idx, lnk) for lnk in cell_starts])

    all_starts_single_g = []  # (node_tc_idx, link); lnk = (tgt_t_idx, tgt_c_idx, m_idx)
    all_starts_g = []  # (node_tc_idx, link); lnk = (tgt_t_idx, tgt_c_idx, m_idx)
    for group in lc_disjoint_groups:
        all_starts_single_gi = []
        all_starts_gi = []
        for tc_idx in group:
            lc = lc_map[tc_idx]
            cell_starts_p = lc.get_starts_clean()
            cell_starts = lc.get_starts()
            all_starts_single_gi.extend([(tc_idx, lnk) for lnk in cell_starts_p])
            all_starts_gi.extend([(tc_idx, lnk) for lnk in cell_starts])
        all_starts_single_g.append(all_starts_single_gi)
        all_starts_g.append(all_starts_gi)
    return all_starts, all_starts_g, all_starts_single_g


def get_track_start_info(stack, all_starts, lc_map):
    """
    obtain coordinates (x, y, t) and tc_idx lists for track start nodes
    Args:
        stack:
        all_starts:
        lc_map:

    Returns:

    """
    # fill tc_idx for all track cells
    all_tracks = []  # item: list(node_tc_idx)
    all_tracks_start_tcidx = {}  # item: list(node_tc_idx)

    for start in all_starts:
        track = []
        p_lc_tc_idx, pm_link = start

        *n_lc_tc_idx, m_idx_p = pm_link
        n_lc_tc_idx = tuple(n_lc_tc_idx)
        track.append(p_lc_tc_idx)  # start node
        track.append(n_lc_tc_idx)  # next node

        while True:
            n_lc = lc_map[n_lc_tc_idx]
            mp_link = (*p_lc_tc_idx, m_idx_p)  # middle->prev
            mn_link = n_lc.get_next_node_link(mp_link)  # middle->next
            if mn_link is None:
                break

            p_lc_tc_idx = n_lc_tc_idx
            *n_lc_tc_idx, m_idx_p = mn_link
            n_lc_tc_idx = tuple(n_lc_tc_idx)
            track.append(n_lc_tc_idx)

        all_tracks_start_tcidx[start[0]] = len(all_tracks)
        all_tracks.append(track)

    # fill xyt array
    all_tracks_xyt = [[cell_r_ofs_t(stack.st[t_idx].cells[c_idx]) for t_idx, c_idx in tr] for tr in all_tracks]

    return all_tracks_xyt, all_tracks_start_tcidx


def save_tracks_simple(tracks_xyt, filename):
    with open(filename, 'wt') as f:
        f.write('%d\n' % len(tracks_xyt))
        for tr_idx, tr in enumerate(tracks_xyt):
            f.write('%d %d\n' % (len(tr), tr_idx))

            for x, y, t in tr:
                f.write('%.3f %.3f %.3f %d\n' % (x, y, 0., t))


def plot_tracks_simple(stack, tracks_xyt):
    # plot tracks
    stack.display(s=30, rng=None, links=None)
    draw_list = []
    # cols = ['k', 'b', 'g', 'r']
    for tr_idx, tr_xyt in enumerate(tracks_xyt):
        x, y, t = np.array(tr_xyt).transpose()
        # c = cols[tr_idx % 4]
        draw_list.extend([x, y])
    _ = plt.plot(*draw_list)
    plt.show()
    plt.close()


def get_n_m_nodes_per_group(lc_map, lc_disjoint_groups):
    # go through groups, find # m_nodes
    m_per_grp = []
    for grp in lc_disjoint_groups:
        m = 0
        for tc_idx in grp:
            lc = lc_map[tc_idx]
            m += 1 if lc.is_m_node() else 0
        m_per_grp.append(m)

    m_per_grp = np.array(m_per_grp)
    m_map = m_per_grp > 0

    # _=plt.hist(m_per_grp[m_map], np.linspace(0, 25, 26))
    # plt.title('M nodes per disjoint groups')

    groups_with_m_nodes = np.arange(len(m_map))[m_map]
    return m_per_grp, groups_with_m_nodes


def plot_linked_cells_in_m_groups(groups_with_m_nodes, all_starts_g, all_starts_single_g,
                                  all_tracks_start_tcidx, all_tracks_xyt):
    for gr_idx in groups_with_m_nodes:
        grp_starts = all_starts_g[gr_idx]

        fig, ax = plt.subplots(1, 3, figsize=(21, 7))

        for start in grp_starts:
            track_tcids_idx = all_tracks_start_tcidx[start[0]]
            track_xyt = all_tracks_xyt[track_tcids_idx]
            track_xyt = np.array(track_xyt)
            x, y, t = track_xyt.transpose()
            ax[0].plot(t, x)
            ax[0].set_xlabel('t')
            ax[0].set_ylabel('x')
            ax[1].plot(t, y)
            ax[1].set_xlabel('t')
            ax[1].set_ylabel('y')
            ax[2].plot(x, y)
            ax[2].set_xlabel('x')
            ax[2].set_ylabel('y')
        plt.suptitle('xt, yt plots of intersecting tracks within group %d, n_tr=%d/%d' % (gr_idx,
                                                                                          len(all_starts_single_g[
                                                                                                  gr_idx]),
                                                                                          len(all_starts_g[gr_idx])
                                                                                          ))
        plt.savefig('tracks_in_group\\%03d' % gr_idx)
        # plt.show()
        plt.close()


# switch to graph of segments and nodes picture
class Vertex:
    def __init__(self, lc: LinkedCell):
        self.t_idx, self.c_idx = lc.t_idx, lc.c_idx

        self.is_l_end = len(lc.l_to_r) == 1  # only None->r links
        self.is_r_end = len(lc.r_to_l) == 1  # only l links -> None

        self.l_links = [l for l in lc.sel_links[0]]
        self.r_links = [l for l in lc.sel_links[1]]

        self.l_nc_f_slv = 0
        self.r_nc_f_slv = 0

    def __repr__(self):
        return 'end (%d %d)' % (self.is_l_end, self.is_r_end) + ' ' + str(self.l_links) + ' ' + str(self.r_links)


class Segment:
    st_normal = 0
    st_merge_jump = 1
    st_flow_jump = 2
    st_merged = 3  # combination of all types

    def __init__(self, lc_map, s_type=st_normal):
        self.nodes = []  # list of tc_idx
        self.node_map = {}  # t_idx->[c_idx]

        self.l_links = []  # list of links, ((l_tc_idx,r_tc_idx))
        self.r_links = []  # all to same node
        self.flow_map = {}  # ordered: flow_map[t] for t in sorted(flow_map.keys()). t - half integer, between nodes
        self.flow_median = 0
        self.flow_slv = 0

        self.mj_t_ranges = []  # list of (tidx_from,tidx_to) time spans on which merge jumps occured
        self.fj_t_ranges = []  # list of (tidx_from,tidx_to) time spans on which  flow jumps occured

        self.lc_map = lc_map

        self.w = 0  # used only for special types of segments (jump-segs) during segment search as usage weight.

        self.m_dr = 0
        self.s_dr = 0

        self.type = s_type

    def add_node(self, tc_idx):
        t_idx, c_idx = tc_idx
        if len(self.nodes) and self.nodes[0][0] > t_idx:
            self.nodes.insert(0, tc_idx)
        else:
            self.nodes.append(tc_idx)

        if t_idx not in self.node_map:
            self.node_map[t_idx] = []
        self.node_map[t_idx].append(c_idx)

    def remove_nodes_at(self, t_idx):
        """
        Returns:
            list tc_idx-s of the removed nodes
        """
        removed_tc_idx = [(t_idx, c_idx) for c_idx in self.node_map.get(t_idx, [])]
        sremoved_tc_idx = set(removed_tc_idx)
        if len(sremoved_tc_idx):
            del self.node_map[t_idx]

        self.nodes = [n for n in self.nodes if n not in sremoved_tc_idx]

        return removed_tc_idx

    def get_node_t_minmax(self):
        if len(self.node_map) == 0:
            raise ValueError('no nodes in the segment')
        t_idxs = sorted(list(self.node_map.keys()))
        return t_idxs[0], t_idxs[-1]

    def add_merge_jump_timespan(self, t_idx_from, t_idx_to):
        self.mj_t_ranges.append((t_idx_from, t_idx_to))

    def add_flow_jump_timespan(self, t_idx_from, t_idx_to):
        self.fj_t_ranges.append((t_idx_from, t_idx_to))

    def add_l_link(self, lnk):
        self.l_links.append(lnk)

    def add_r_link(self, lnk):
        self.r_links.append(lnk)

    def get_t_minmax(self):
        t_min = self.l_links[0][0][0]
        t_max = self.r_links[0][1][0]

        if t_min == -1:
            t_min = self.l_links[0][1][0]
        if t_max == -1:
            t_max = self.r_links[0][0][0]
        return t_min, t_max

    def fill_struct(self, stack):  # fill all struct after addign nodes
        # print(self.l_links, self.r_links)
        assert (len(self.l_links) and len(self.r_links))

        t_min, t_max = self.get_t_minmax()

        if self.type == self.st_merge_jump:
            self.add_merge_jump_timespan(t_min, t_max)
        if self.type == self.st_flow_jump:
            self.add_flow_jump_timespan(t_min, t_max)

        # t_node in [t_min, t_max],  t_flow in [t_min+0.5, t_max-0.5]
        for lnk in self.l_links:
            if lnk[0][0] not in [t_min, -1]:
                raise ValueError('all starts of segment should be same')
        for lnk in self.r_links:
            if lnk[1][0] not in [t_max, -1]:
                raise ValueError('all ends of segment should be same')

        self.flow_map = {(t + 0.5): 0 for t in range(t_min, t_max)}

        r1_t1_r2_t2 = []

        # fill flow for segment l-links:
        for l_tc_idx, r_tc_idx in self.l_links:
            # print(l_tc_idx, r_tc_idx)
            lc: LinkedCell = self.lc_map.get(l_tc_idx)
            if lc is None:
                continue
            m = lc.get_link_m(r_tc_idx)
            for t in range(l_tc_idx[0], r_tc_idx[0]):
                self.flow_map[t + 0.5] = m

        # fill flow for r-links of nodes in the segment:
        for tc_idx_l in self.nodes:
            t_idx_l, c_idx_l = tc_idx_l
            lc = self.lc_map[tc_idx_l]
            r_l = stack.st[t_idx_l].cells[c_idx_l].get_r()

            lnk: LinkedCell.Link
            r_links = lc.sel_links[1]
            save_dr_l = len(r_links) == 1

            for lnk in r_links:  # right links
                assert lnk.selected
                m = lnk.m
                t_idx_r = lnk.tgt_t_idx
                c_idx_r = lnk.tgt_c_idx
                for t in range(t_idx_l, t_idx_r):
                    self.flow_map[t + 0.5] = m

                lc_r = self.lc_map[(t_idx_r, c_idx_r)]
                l_links = lc_r.sel_links[0]
                save_dr_r = len(l_links) == 1

                if save_dr_l and save_dr_r:
                    r_r = stack.st[t_idx_r].cells[c_idx_r].get_r()
                    r1_t1_r2_t2.append((r_l, t_idx_l, r_r, t_idx_r))
                    # print((r_l, t_idx_l, r_r, t_idx_r))

        dr_dt = [sqrt((r1[0] - r2[0]) ** 2 + (r1[1] - r2[1]) ** 2) / abs(t2 - t1) for r1, t1, r2, t2 in r1_t1_r2_t2 if
                 t1 != t2]
        flow_map = list(self.flow_map.values())
        self.flow_median = int(np.median(flow_map)) if len(flow_map) else 0
        self.m_dr, self.s_dr = mean_std(dr_dt) if len(dr_dt) else (0, 0)


def get_link_m(link, lc_map):
    l, r = link
    lc: LinkedCell = lc_map[l]
    return lc.get_link_m(r)


def non_own_branches_joined(lc, cells_on_branches, root_node, isfirst):
    # print(cells_on_branches)
    left_nodes = lc.get_selected_l_nodes()

    left_nodes = set(left_nodes)
    not_on_branches = left_nodes - cells_on_branches
    if isfirst:
        not_on_branches.difference_update([root_node])

    return len(not_on_branches) > 0


def is_end_node(lc):
    # print([[[lnk.tgt_t_idx,lnk.tgt_c_idx] for lnk in links] for links in lc.sel_links])
    # [0][0].tgt_t_idx, lc.sel_links[1][0].tgt_t_idx
    return len(lc.sel_links[1]) == 0


# search for segments and vertices in the obtained graph of linked cells
def search_segment(sgms, lc_map, stack, tc_idx, tc_idx_from, proc_set, vtx_candidate_tc_idx, max_branching_length=7):
    """
    Starts from a single node, maybe branches, ends on either single node
    or branches merge at once into a single node.
        V-*=======*-V
    or
        V-*=======*=V

    Node on the end link will be added as vertex
    """
    tc_idx_start = tc_idx  # first possible other_cell in the segment (might be a vtx)

    accepted_confirmed_tci = set()  # all nodes that confirmed to be in segment
    accepted_tci = set()  # will become part of segment if all branches merge into 1 node
    last_accepted_tci = set()  # the ones from which links go to potential vertex, if curr active will be vtx
    active_tci = {tc_idx_start}  # next nodes to be evaluated if is vtx or end
    active_confirmed_tci = {}  # copied from active_tci upon confirmation. tci of next vtx
    t_active_confirmed = 0

    # 1. not suitable: First node or has more then 1 input
    # make dummy segment From prev to this
    # make sure dunnies are made for target which was previously used too
    lc: LinkedCell = lc_map[tc_idx_start]

    # 2 mark node as candidate, traverse branches till not end or vtx:
    while True:
        # check active is only one other_cell: update confirmed
        if len(active_tci) == 1:
            active_confirmed_tci = active_tci.copy()
            accepted_confirmed_tci = accepted_tci.copy()
            t_active_confirmed = list(active_confirmed_tci)[0][0]

        # check any of active with minimal t_idx is not suitable
        t_active = [tc[0] for tc in active_tci]
        t_active_min = min(t_active)

        if t_active_min - t_active_confirmed > max_branching_length:  # disallow too long branchings
            break

        act_tci_min_t = [tc for tc in active_tci if tc[0] == t_active_min]

        end_map = [is_end_node(lc_map[tc]) for tc in act_tci_min_t]

        vtx_map = [non_own_branches_joined(lc_map[tc], accepted_tci,
                                           tc_idx_from, isfirst=tc == tc_idx_start
                                           ) for tc in act_tci_min_t]
        cnd_map = [tc in vtx_candidate_tc_idx for tc in act_tci_min_t]

        # print(end_map, vtx_map)
        if any(end_map) or any(vtx_map) or any(cnd_map):
            break

        # advance from min_t active nodes, update accepted
        accepted_tci.update(act_tci_min_t)
        last_accepted_tci = set(act_tci_min_t)  # ones from not updated active - removed.
        active_tci.difference_update(act_tci_min_t)
        updated_nodes = [tci for tc in act_tci_min_t for tci in lc_map[tc].get_selected_r_nodes()]
        active_tci.update(updated_nodes)

    # 3. create segment connecting tc_idx_from to active_confirmed_tci
    assert (len(active_confirmed_tci) == 1)
    sgm = Segment(lc_map)

    for tc in accepted_confirmed_tci:
        sgm.add_node(tc)
        proc_set.add(tc)

    tc_idx_to = list(active_confirmed_tci)[0]
    if len(accepted_confirmed_tci) == 0:
        lnk = (tc_idx_from, tc_idx_to)
        sgm.add_l_link(lnk)
        sgm.add_r_link(lnk)
    else:
        t = list(sgm.node_map.keys())
        t_min, t_max = min(t), max(t)

        l_nodes_wrt_to = lc_map[tc_idx_to].get_selected_l_nodes()
        l_nodes = [(t_min, c_idx) for c_idx in sgm.node_map[t_min]]  # min t
        r_nodes = accepted_confirmed_tci.intersection(l_nodes_wrt_to)  # all accepted which are linked to tc_idx_to
        for tc_idx in l_nodes:
            sgm.add_l_link((tc_idx_from, tc_idx))
        for tc_idx in r_nodes:
            sgm.add_r_link((tc_idx, tc_idx_to))

    sgm.fill_struct(stack)
    sgms.append(sgm)

    # print(tc_idx_from, tc_idx_to)
    # print(accepted_confirmed_tci, active_confirmed_tci)
    return sgm


def all_inputs_ready(lc_map, node_tc_idx, proc_set):
    lc: LinkedCell = lc_map[node_tc_idx]
    for lnk in lc.sel_links[0]:
        tc_idx = lnk.tgt_t_idx, lnk.tgt_c_idx
        if tc_idx not in proc_set:
            return False
    return True


def process_from_node(vtxs, sgms, lc_map, stack, node_tc_idx, proc_set, is_root=True, vtx_candidate_tc_idx=set()):
    # print('start pfn')
    if node_tc_idx in proc_set:
        return  # already went through this branch

    if not (all_inputs_ready(lc_map, node_tc_idx, proc_set) or is_root):
        # all input branches have to be ready before continuing
        # if not ready yet - the node will be processed once the missing branch reaches the node
        return

    proc_set.add(node_tc_idx)

    lc: LinkedCell = lc_map[node_tc_idx]
    v = Vertex(lc)
    vtxs.append(v)

    for lnk in lc.sel_links[1]:  # all right links
        t_idx, c_idx = lnk.tgt_t_idx, lnk.tgt_c_idx
        tc_idx = (t_idx, c_idx)
        found_seg = search_segment(sgms, lc_map, stack, tc_idx, node_tc_idx, proc_set, vtx_candidate_tc_idx)

        next_tc_idx = found_seg.r_links[0][1]

        process_from_node(vtxs, sgms, lc_map, stack, next_tc_idx, proc_set, is_root=False,
                          vtx_candidate_tc_idx=vtx_candidate_tc_idx)


def get_group_segments_vertices_from_links(stack, lc_map, all_starts_single_g, vtx_candidate_tc_idx=set()):
    # simplify graph by finding consistent segments
    vtx_g = []
    sgm_g = []

    for g_idx, pure_starts in enumerate(all_starts_single_g):
        processed_nodes = set()

        grp_vtx = []
        grp_sgm = []

        for start_tc_idx, link in pure_starts:
            # start - is a vertex
            process_from_node(grp_vtx, grp_sgm, lc_map, stack, start_tc_idx, processed_nodes)

        vtx_g.append(grp_vtx)
        sgm_g.append(grp_sgm)

    return vtx_g, sgm_g


def plot_segments_in_groups(stack, groups_with_m_nodes,
                            all_starts_g,
                            all_tracks_start_tcidx, all_tracks_xyt,
                            vtx_g, sgm_g, sgm_c_g=None,
                            folder='ims',
                            only_m_nodes=False,
                            multiplicity_colors=False,
                            ):
    sgroups_with_m_nodes = set(groups_with_m_nodes) if groups_with_m_nodes is not None else None
    only_m_nodes = only_m_nodes and groups_with_m_nodes is not None

    colors = ['k', 'b', 'g', 'y', 'm', 'c']+['r']*100  # 100 colors for multiplicity>5, namely not resolvable, thus red
    for gr_idx in np.arange(len(vtx_g)):
        if only_m_nodes and gr_idx not in sgroups_with_m_nodes:
            continue

        grp_starts = all_starts_g[gr_idx]

        s = 14
        fig, ax = plt.subplots(1, 3, figsize=(3 * s, s))

        for start in grp_starts:
            track_tcids_idx = all_tracks_start_tcidx[start[0]]
            track_xyt = all_tracks_xyt[track_tcids_idx]
            track_xyt = np.array(track_xyt)
            x, y, t = track_xyt.transpose()

            ax[0].plot(t, x, 'grey' if multiplicity_colors else '')
            ax[0].set_xlabel('t')
            ax[0].set_ylabel('x')
            ax[1].plot(t, y, 'grey' if multiplicity_colors else '')
            ax[1].set_xlabel('t')
            ax[1].set_ylabel('y')
            ax[2].plot(x, y, 'grey' if multiplicity_colors else '')
            ax[2].set_xlabel('x')
            ax[2].set_ylabel('y')
            ax[2].set_aspect('equal')

        grp_vtx = vtx_g[gr_idx]
        grp_sgm = sgm_g[gr_idx]
        grp_sgm_c = sgm_c_g[gr_idx] if sgm_c_g is not None else None

        vtx_xyt = [cell_r_ofs_t(stack.st[vtx.t_idx].cells[vtx.c_idx]) for vtx in grp_vtx]
        vtx_xyt = np.array(vtx_xyt)
        vx, vy, vt = vtx_xyt.transpose()
        ax[0].scatter(vt, vx, s=64, c='k')
        ax[1].scatter(vt, vy, s=64, c='k')
        ax[2].scatter(vx, vy, s=64, c='k')

        all_segments_plots = [[], [], []]
        for sgm in grp_sgm:
            tc_i = sgm.l_links[0][0]
            tc_o = sgm.r_links[0][1]
            s_i_xyt = cell_r_ofs_t(stack.st[tc_i[0]].cells[tc_i[1]])
            s_o_xyt = cell_r_ofs_t(stack.st[tc_o[0]].cells[tc_o[1]])

            col = colors[sgm.flow_slv] if multiplicity_colors else 'k'
            all_segments_plots[0].extend([[s_i_xyt[2], s_o_xyt[2]], [s_i_xyt[0], s_o_xyt[0]], col])
            all_segments_plots[1].extend([[s_i_xyt[2], s_o_xyt[2]], [s_i_xyt[1], s_o_xyt[1]], col])
            all_segments_plots[2].extend([[s_i_xyt[0], s_o_xyt[0]], [s_i_xyt[1], s_o_xyt[1]], col])

        for sgm in (grp_sgm_c if grp_sgm_c is not None else []):
            tc_i = sgm.l_links[0][0]
            tc_o = sgm.r_links[0][1]
            s_i_xyt = cell_r_ofs_t(stack.st[tc_i[0]].cells[tc_i[1]])
            s_o_xyt = cell_r_ofs_t(stack.st[tc_o[0]].cells[tc_o[1]])

            col = '--' + (colors[sgm.flow_slv] if multiplicity_colors else 'k')
            all_segments_plots[0].extend([[s_i_xyt[2], s_o_xyt[2]], [s_i_xyt[0], s_o_xyt[0]], col])
            all_segments_plots[1].extend([[s_i_xyt[2], s_o_xyt[2]], [s_i_xyt[1], s_o_xyt[1]], col])
            all_segments_plots[2].extend([[s_i_xyt[0], s_o_xyt[0]], [s_i_xyt[1], s_o_xyt[1]], col])

        ax[0].plot(*(all_segments_plots[0]), alpha=0.5)
        ax[1].plot(*(all_segments_plots[1]), alpha=0.5)
        ax[2].plot(*(all_segments_plots[2]), alpha=0.5)

        plt.suptitle('xt, yt plots of intersecting tracks within group %d, n_tr=%d' % (gr_idx,
                                                                                       len(all_starts_g[gr_idx])
                                                                                       ))
        plt.tight_layout(pad=0.5, h_pad=0, w_pad=0)
        plt.savefig(f'{folder}\\{gr_idx:03d}')
        # if gr_idx in sgroups_with_m_nodes:
        #    plt.show()
        plt.close()


def get_all_merge_jump_dr(vtx_g, sgm_g, stack):
    dr_merge = []
    for grp_idx, (grp_vtx, grp_sgm) in enumerate(zip(vtx_g, sgm_g)):
        for vtx in grp_vtx:
            vtx: Vertex
            v_t_idx, v_c_idx = vtx.t_idx, vtx.c_idx
            *v_r, _ = cell_r_ofs_t(stack.st[v_t_idx].cells[v_c_idx])
            for links in [vtx.l_links, vtx.r_links]:
                if len(links) > 1:
                    for link in links:
                        t_idx, c_idx = link.tgt_t_idx, link.tgt_c_idx
                        *c_r, _ = cell_r_ofs_t(stack.st[t_idx].cells[c_idx])
                        dr2 = (v_r[0] - c_r[0]) ** 2 + (v_r[1] - c_r[1]) ** 2
                        dr = sqrt(dr2)
                        dr_merge.append(dr)
    return np.array(dr_merge)


def get_min_dr_sqrt_merge(dr_merge, dr_merge_min=3, plot=False):
    # square root looks more normal
    filter_mask = dr_merge > dr_merge_min
    dr_merge_filtered = dr_merge[filter_mask]
    dr_sqrt_merge_filtered = np.sqrt(dr_merge_filtered)
    dr_sqrt_merge_mean, dr_sqrt_merge_std = mean_std(dr_sqrt_merge_filtered)

    if len(dr_sqrt_merge_filtered) < 5:
        dr_sqrt_merge_mean, dr_sqrt_merge_std = cfgm.DRJS_MEAN, cfgm.DRJS_STD

    dr_sqrt_merge_min_is_vtx = dr_sqrt_merge_mean - 2 * dr_sqrt_merge_std

    if plot:
        print(dr_sqrt_merge_mean, dr_sqrt_merge_std,
              dr_sqrt_merge_mean - dr_sqrt_merge_std,
              dr_sqrt_merge_mean - 2 * dr_sqrt_merge_std,
              *mean_std(np.sqrt(dr_merge[~filter_mask]))
              )
        bins = np.linspace(0, dr_sqrt_merge_filtered.max(), 50)
        print('in get_min_dr_sqrt_merge')
        _ = plt.hist(np.sqrt(dr_merge), bins)
        _ = plt.hist(dr_sqrt_merge_filtered, bins, alpha=0.75)
    return dr_sqrt_merge_min_is_vtx, dr_sqrt_merge_mean, dr_sqrt_merge_std


def find_vtx_on_segments(sgm_g, stack, lc_map, min_sqrt_dr_candidate):
    # only nodes that have one selected l link, one r selected r link, const flow
    vtx_cand_tc_idx = set()
    for grp_idx, grp_sgm in enumerate(sgm_g):
        for sgm_idx, sgm in enumerate(grp_sgm):
            sgm: Segment
            sgm_nodes = set(sgm.nodes)
            sgm_max_not_jmp = sgm.m_dr + 0.5 * sgm.s_dr
            for n_tc_idx in sgm_nodes:
                n_t_idx, n_c_idx = n_tc_idx
                n_cell = stack.st[n_t_idx].cells[n_c_idx]
                n_lc = lc_map[n_tc_idx]
                n_l_nodes, n_r_nodes = n_lc.get_selected_l_nodes(), n_lc.get_selected_r_nodes()

                if len(n_l_nodes) != 1 or len(n_r_nodes) != 1:  # constant flow, nobifurcation
                    continue
                r_tc_idx = n_r_nodes[0]
                r_t_idx, r_c_idx = r_tc_idx

                if r_tc_idx not in sgm_nodes:  # only within segment
                    continue

                r_cell = stack.st[r_t_idx].cells[r_c_idx]
                r_lc = lc_map[r_tc_idx]
                r_l_nodes, r_r_nodes = r_lc.get_selected_l_nodes(), r_lc.get_selected_r_nodes()

                if len(r_l_nodes) != 1 or len(r_r_nodes) != 1:  # constant flow, nobifurcation
                    continue

                *n_r, _ = cell_r_ofs_t(n_cell)
                *r_r, _ = cell_r_ofs_t(r_cell)
                dr2 = (n_r[0] - r_r[0]) ** 2 + (n_r[1] - r_r[1]) ** 2
                dr = sqrt(dr2)
                dr_sqrt = sqrt(dr)

                # print(sgm_idx, n_tc_idx, r_tc_idx, dr, dr_sqrt)
                if dr_sqrt > min_sqrt_dr_candidate and dr > sgm_max_not_jmp:  # is a VTX!
                    n_area, r_area = n_cell.w, r_cell.w

                    # choose node with bigger area
                    vtx_tc_idx = [n_tc_idx, r_tc_idx][np.argmax([n_area, r_area])]
                    vtx_cand_tc_idx.add(vtx_tc_idx)
    return vtx_cand_tc_idx


# Solving for consistent picture of multiplicity
class NodeStruct_F:
    """
    Nodes are vertexes: cells where merging or splitting occurs.
    They are linked by segments with a flow estimate, treated as minimal flow on segment
    Solving aims at finding a flow in each segment which makes influx and outflux consistent.
    Solving is subjected minimizing the cost:
      - difference between found flow and flow estimate (fully unresolved).
      - not-connected flow in nodes
      - flow>1 in end nodes
    NC weight is reduced close to fiducial volume boundary.
    In case of global solving, w_nc_left is additionally reduced at the accumulation phase .
    """

    def __init__(self, idx, w_f_mult_end):
        # w_f_mult_end - per unit flow above 1

        # f - flow variable. Non-connected and connected
        self.f_nc_l = None
        self.f_nc_r = None

        self.f_end_l = None  # var, last node flow l
        self.f_end_r = None  # var, last node flow r
        self.f_lost_l = None  # var, lost floaw at the node l
        self.f_lost_r = None  # var, lost floaw at the node r

        self.f_l = []  # vars, flow, each in range [f_est, f_est+f_max]
        self.f_r = []

        self.f_est_l = []  # estimators of flow. f_l[i] >= f_est_l[i]
        self.f_est_r = []
        self.f_hnt_r = []

        self.f_p_l = []  # vars, Possible flow: each in range [0, f_max]
        self.f_p_r = []

        self.b_p_l = []  # vars, Possible used, boolean
        self.b_p_r = []

        self.b_c_l = None  # var, has conn l
        self.b_c_r = None  # var, has conn r

        self.links_l = []  # elements - ([node_on_the_left_idx, r_link_(of_left_node)_idx])
        self.links_r = []  # elements - ([node_on_the_rght_idx, l_link_(of_rght_node)_idx])

        self.plinks_l = []  # elements - ([node_on_the_left_idx, r_plink_(of_left_node)_idx]). possible links.
        self.plinks_r = []  # elements - ([node_on_the_rght_idx, l_plink_(of_rght_node)_idx]). possible links.

        self.w_r_above_est = []  # weihgts of  links on the right side. left are on the right side of connected nodes.
        self.w_pl_r = []  # weihgts of plinks on the right side. left are on the right side of connected nodes.

        self.w_nc_l = 0  # nll other_cell starts here. important for global solving, connect bad vs track end traid-off
        self.w_nc_r = 0  # nll other_cell ends here
        self.w_f_mult_end = w_f_mult_end  # nll double in one place. not attenuated with time-line

        self.idx = idx

        self.solved = None

    def add_r_link_to_node(self, node, f_est, w_above_est, f_hnt):
        """w_above_est per flow unit bove f_est"""
        self_r_link_idx = len(self.links_r)
        node_l_link_idx = len(node.links_l)

        self.links_r.append((node.idx, node_l_link_idx))
        node.links_l.append((self.idx, self_r_link_idx))

        self.f_est_r.append(f_est)
        self.f_hnt_r.append(f_hnt)
        node.f_est_l.append(f_est)

        self.w_r_above_est.append(w_above_est)
        return self_r_link_idx, node_l_link_idx

    def add_r_plink_to_node(self, node, w):
        """w - weight if flow > 0"""
        self_r_plink_idx = len(self.plinks_r)
        node_l_plink_idx = len(node.plinks_l)

        self.plinks_r.append((node.idx, node_l_plink_idx))
        node.plinks_l.append((self.idx, self_r_plink_idx))

        self.w_pl_r.append(w)
        return self_r_plink_idx, node_l_plink_idx

    def set_nc_weight(self, w_nc_l, w_nc_r):
        # per flow unit
        self.w_nc_l = w_nc_l
        self.w_nc_r = w_nc_r

    def _get_f_on_the_right(self, r_idx, nodes):
        link = self.links_r[r_idx]
        r_node_idx, l_idx = link
        r_node = nodes[r_node_idx]
        f_r_node_l_link = r_node.f_l[l_idx]
        return f_r_node_l_link

    def _get_fp_on_the_right(self, r_idx, nodes):
        link = self.plinks_r[r_idx]
        r_node_idx, l_idx = link
        r_node = nodes[r_node_idx]
        f_r_node_l_plink = r_node.f_p_l[l_idx]
        return f_r_node_l_plink

    def setup_vars(self, model):
        f_max = 5  # way above any reasonable scale. nothing can be resolved with curent methods.

        self.f_nc_l = model.NewIntVar(0, f_max, f'n{self.idx}_f_nc_l')
        self.f_nc_r = model.NewIntVar(0, f_max, f'n{self.idx}_f_nc_r')
        self.f_end_l = model.NewIntVar(0, f_max, f'n{self.idx}_f_end_l')
        self.f_end_r = model.NewIntVar(0, f_max, f'n{self.idx}_f_end_r')
        self.f_lost_l = model.NewIntVar(0, f_max, f'n{self.idx}_f_lost_l')
        self.f_lost_r = model.NewIntVar(0, f_max, f'n{self.idx}_f_lost_r')

        self.f_l = [model.NewIntVar(f_est, f_est + f_max, f'n{self.idx}_f_l{i}') for i, f_est in
                    enumerate(self.f_est_l)]
        self.f_r = [model.NewIntVar(f_est, f_est + f_max, f'n{self.idx}_f_r{i}') for i, f_est in
                    enumerate(self.f_est_r)]

        for f_r, hnt in zip(self.f_r, self.f_hnt_r):
            if hnt > 0:
                model.AddHint(f_r, hnt)

        self.f_p_l = [model.NewIntVar(0, f_max, f'n{self.idx}_f_p_l{i}') for i, _ in enumerate(self.plinks_l)]
        self.f_p_r = [model.NewIntVar(0, f_max, f'n{self.idx}_f_p_r{i}') for i, _ in enumerate(self.plinks_r)]

        self.b_p_l = [model.NewBoolVar(f'n{self.idx}_b_p_l{i}') for i, _ in enumerate(self.plinks_l)]
        self.b_p_r = [model.NewBoolVar(f'n{self.idx}_b_p_r{i}') for i, _ in enumerate(self.plinks_r)]

        self.b_c_l = model.NewBoolVar(f'n{self.idx}_b_c_l')
        self.b_c_r = model.NewBoolVar(f'n{self.idx}_b_c_r')

    def setup_node_constraints(self, model):
        all_f_l = self.f_l + self.f_p_l
        all_f_r = self.f_r + self.f_p_r

        sum_links_l = cp_model.LinearExpr.Sum(all_f_l) if len(all_f_l) else 0
        sum_links_r = cp_model.LinearExpr.Sum(all_f_r) if len(all_f_r) else 0

        model.Add(sum_links_r + self.f_nc_r == sum_links_l + self.f_nc_l)  # flow conservation
        if len(all_f_l):  # some links exist
            model.Add(sum_links_l > 0).OnlyEnforceIf(self.b_c_l)
            model.Add(sum_links_l == 0).OnlyEnforceIf(self.b_c_l.Not())
        else:
            model.Add(self.b_c_l == 0)  # flag showing connection set to False

        if len(all_f_r):  # some links exist
            model.Add(sum_links_r > 0).OnlyEnforceIf(self.b_c_r)
            model.Add(sum_links_r == 0).OnlyEnforceIf(self.b_c_r.Not())
        else:
            model.Add(self.b_c_r == 0)  # flag showing connection set to False

        # flag of track end
        model.Add(self.f_end_l == 0).OnlyEnforceIf(self.b_c_l)
        model.Add(self.f_end_r == 0).OnlyEnforceIf(self.b_c_r)
        model.Add(self.f_lost_l == 0).OnlyEnforceIf(self.b_c_l.Not())
        model.Add(self.f_lost_r == 0).OnlyEnforceIf(self.b_c_r.Not())
        model.Add(self.f_end_l == self.f_nc_l).OnlyEnforceIf(self.b_c_l.Not())
        model.Add(self.f_end_r == self.f_nc_r).OnlyEnforceIf(self.b_c_r.Not())
        model.Add(self.f_lost_l == self.f_nc_l).OnlyEnforceIf(self.b_c_l)
        model.Add(self.f_lost_r == self.f_nc_r).OnlyEnforceIf(self.b_c_r)

        for flows, flags in zip([self.f_p_l, self.f_p_r], [self.b_p_l, self.b_p_r]):
            for flow, flag in zip(flows, flags):
                model.Add(flow > 0).OnlyEnforceIf(flag)
                model.Add(flow == 0).OnlyEnforceIf(flag.Not())

    def setup_internode_constraints(self, model, nodes):
        for r_idx, f_r_i in enumerate(self.f_r):
            f_l_j = self._get_f_on_the_right(r_idx, nodes)
            model.Add(f_r_i == f_l_j)
        for r_idx, f_p_r_i in enumerate(self.f_p_r):
            f_p_l_j = self._get_fp_on_the_right(r_idx, nodes)
            model.Add(f_p_r_i == f_p_l_j)

    def get_node_loss_vars_weights(self, to_int_factor):
        vs_L_nc_end = [self.f_end_l, self.f_end_r, self.f_lost_l, self.f_lost_r]
        ws_L_nc_end = [self.w_f_mult_end, self.w_f_mult_end, self.w_nc_l, self.w_nc_r]

        vs_L_seg_flow = self.f_r  # as flow can't be below est by constraint, we can count all flow
        ws_L_seg_flow = self.w_r_above_est

        vs_L_seg_pflow = self.b_p_r
        ws_L_seg_pflow = self.w_pl_r

        vs = vs_L_nc_end + vs_L_seg_flow + vs_L_seg_pflow
        ws = ws_L_nc_end + ws_L_seg_flow + ws_L_seg_pflow

        ws = [int(wi * to_int_factor) for wi in ws]

        return vs, ws

    def fill_solved(self, solver):
        # trick to maintain structure - but store values instead of vars
        self.solved = copy.copy(self)

        self.solved.f_nc_l = solver.Value(self.f_nc_l)
        self.solved.f_nc_r = solver.Value(self.f_nc_r)
        self.solved.f_l = [solver.Value(ci) for ci in self.f_l]
        self.solved.f_r = [solver.Value(ci) for ci in self.f_r]

        self.solved.f_end_l = solver.Value(self.f_end_l)
        self.solved.f_end_r = solver.Value(self.f_end_r)
        self.solved.f_lost_l = solver.Value(self.f_lost_l)
        self.solved.f_lost_r = solver.Value(self.f_lost_r)

        self.solved.f_p_l = [solver.Value(ci) for ci in self.f_p_l]
        self.solved.f_p_r = [solver.Value(ci) for ci in self.f_p_r]
        self.solved.b_p_l = [solver.Value(ci) for ci in self.b_p_l]
        self.solved.b_p_r = [solver.Value(ci) for ci in self.b_p_r]

        self.solved.b_c_l = solver.Value(self.b_c_l)
        self.solved.b_c_r = solver.Value(self.b_c_r)

    def get_solved_links(self):
        # format: ((l_n_idx, l_lnk_idx, r_n_idx, r_lnk_idx), f)
        # end_l: ((-2, -2, r_n_idx, -2), f)
        # end_r: ((l_n_idx, -1, -1, -1), f)

        links = []
        if self.solved.f_nc_l:
            links.append(((-2, -2, self.idx, -1), self.solved.f_nc_l))
        if self.solved.f_nc_r:
            links.append(((self.idx, -1, -1, -1), self.solved.f_nc_r))

        r_node_idx = self.idx
        for r_node_l_lnk_idx, (flow, link) in enumerate(zip(self.solved.f_l, self.links_l)):
            l_node_idx, l_node_r_lnk_idx = link

            lnk = (l_node_idx, l_node_r_lnk_idx, r_node_idx, (0, r_node_l_lnk_idx))
            links.append((lnk, flow))

        for r_node_l_lnk_idx, (flow, link) in enumerate(zip(self.solved.f_p_l, self.plinks_l)):
            if flow > 0:
                l_node_idx, l_node_r_lnk_idx = link
                links.append(((l_node_idx, l_node_r_lnk_idx, r_node_idx, (1, r_node_l_lnk_idx)), flow))

        return links


def solve_flow(vtx_w_nc_lr,
               segs_ends, segs_f_est, segs_f_hnt,
               psegs_ends, psegs_w,
               fid_vol,
               w_nc, w_f_mult_end, w_f_above_est, timeout_sec=None):
    if len(segs_ends) == 0:
        return [], []

    nodes = []
    
    timeout_sec = timeout_sec if timeout_sec is not None else cfgm.DEFAULT_SOLVER_TIMEOUT

    for idx, w_nc_lr in enumerate(vtx_w_nc_lr):
        node: NodeStruct_F = NodeStruct_F(idx, w_f_mult_end)
        node.set_nc_weight(*w_nc_lr)
        nodes.append(node)

    node_link_to_sgm_idx_map = {}
    for sgm_idx, (sgm, f_est, f_hnt) in enumerate(zip(segs_ends, segs_f_est, segs_f_hnt)):
        idx_from, idx_to = sgm

        node_from = nodes[idx_from]
        node_to = nodes[idx_to]

        r_link_idx, l_link_idx = node_from.add_r_link_to_node(node_to, f_est, w_f_above_est, f_hnt)
        # then associate with segment idx
        node_link_to_sgm_idx_map[(idx_from, r_link_idx, idx_to, (0, l_link_idx))] = sgm_idx

    for sgm_idx, (sgm, w) in enumerate(zip(psegs_ends, psegs_w)):
        idx_from, idx_to = sgm

        node_from = nodes[idx_from]
        node_to = nodes[idx_to]

        r_link_idx, l_link_idx = node_from.add_r_plink_to_node(node_to, w)
        # then associate with segment idx
        node_link_to_sgm_idx_map[(idx_from, r_link_idx, idx_to, (1, l_link_idx))] = sgm_idx

    model = cp_model.CpModel()

    for node in nodes:
        node.setup_vars(model)
    for node in nodes:
        node.setup_node_constraints(model)
    for node in nodes:
        node.setup_internode_constraints(model, nodes)

    vs = []
    ws = []
    for node in nodes:
        vsi, wsi = node.get_node_loss_vars_weights(1000)
        vs.extend(vsi)
        ws.extend(wsi)

    # for vi,wi in zip(vs, ws):
    #    print(vi, '\t',wi)

    # print(model.Validate())
    model.Minimize(cp_model.LinearExpr.ScalProd(vs, ws))
    # model.AddDecisionStrategy(vs, cp_model.CHOOSE_LOWEST_MIN, cp_model.SELECT_MIN_VALUE)
    #

    solver = cp_model.CpSolver()

    # >>>>>>>>>>>>>>>
    solver.parameters.num_search_workers = cfgm.NUM_WORKERS
    solver.parameters.max_time_in_seconds = timeout_sec
    # print('w', solver.parameters.num_search_workers)
    # print('a', solver.parameters.minimization_algorithm)
    # print('t', solver.parameters.max_time_in_seconds)
    # print('m', solver.parameters.max_memory_in_mb)

    # print(model.ModelStats())
    t1 = time.time()

    #     status = solver.Solve(model)

    solution_printer = SolutionPrinter()
    status = solver.SolveWithSolutionCallback(model, solution_printer)
    status_str = solver.StatusName(status)

    t2 = time.time()
    dt = t2 - t1
    if dt > cfgm.LONG_RUN_PRINT_TIMEOUT:
        print(status_str)
        print('\ndt=%.1fs:\n' % dt, model.ModelStats())

    links = []
    # format ((from_idx, to_idx), lnk_idx, f)
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        for node in nodes:
            node.fill_solved(solver)

        for node in nodes:
            node_links = node.get_solved_links()
            # print(node_links)
            for lnk, f in node_links:
                l_node_idx, l_node_r_lnk_idx, r_node_idx, r_node_l_lnk_idx = lnk
                if l_node_idx == -2:  # l nc flow
                    l = ((-2, r_node_idx), -1, f)
                elif r_node_idx == -1:  # r nc flow
                    l = ((l_node_idx, -1), -1, f)
                else:  # actual segment
                    # print('\n', r_node_l_lnk_idx, lnk)
                    link_tp, _ = r_node_l_lnk_idx
                    sgm_idx = node_link_to_sgm_idx_map[lnk]
                    l = ((l_node_idx, r_node_idx), (link_tp, sgm_idx), f)
                # print(l)
                links.append(l)
        # for v in vs:
        #    print(v, ' = %i' % solver.Value(v))

        # for l in links:
        #    print(l)

    return links, nodes


# disjoint groups solve flow:
def solve_groups_flow(stack, vtx_g, sgm_g, w_nc, w_f_mult_end, w_f_above_est):
    fid_vol = stack.f_fid_vol()
    for grp_idx, (grp_vtx, grp_sgm) in enumerate(tqdm(zip(vtx_g, sgm_g), desc='segment multiplicity', ascii=True)):
        # print(grp_idx, '/', len(vtx_g), end='\r')

        tc_to_lidx = {}

        vtx_w_nc_lr = []
        for vtx_idx, vtx in enumerate(grp_vtx):
            tidx, cidx = vtx.t_idx, vtx.c_idx
            tcidx = tidx, cidx
            cell = stack.st[tidx].cells[cidx]
            r = cell.get_r()

            w_nc_cell = w_nc_fid_vol_corr(w_nc, fid_vol, r)  # to be corrected with time

            vtx_w_nc_lr.append((w_nc_cell, w_nc_cell))
            tc_to_lidx[tcidx] = vtx_idx

        segs_ends = []
        segs_f_est = []
        segs_f_hnt = []
        for sgm in grp_sgm:
            node_l_tc_idx = sgm.l_links[0][0]
            node_r_tc_idx = sgm.r_links[0][1]
            segs_ends.append((tc_to_lidx[node_l_tc_idx], tc_to_lidx[node_r_tc_idx]))
            segs_f_hnt.append(-1)
            segs_f_est.append(sgm.flow_median)

        resolved_links, *_ = solve_flow(vtx_w_nc_lr, segs_ends, segs_f_est, segs_f_hnt, [], [],
                                        fid_vol, w_nc, w_f_mult_end, w_f_above_est
                                        )
        for link in resolved_links:
            (from_idx, to_idx), lnk_info, f = link
            if lnk_info == -1:
                if from_idx == -2:
                    grp_vtx[to_idx].l_nc_f_slv = f
                elif to_idx == -1:
                    grp_vtx[from_idx].r_nc_f_slv = f
            else:
                lint_type, lnk_idx = lnk_info
                assert (lint_type == 0)
                grp_sgm[lnk_idx].flow_slv = f
    print('')


def vtx_connected_by_segments(tc_idx_l, tc_idx_r, sgm_lr_map):
    """ checks whether two vertices are connceted by segments"""
    if tc_idx_l[0] > tc_idx_r[0]:
        tc_idx_l, tc_idx_r = tc_idx_r, tc_idx_l

    tc = [tc_idx_l]

    while len(tc):
        tc_next = [tci_next for tci in tc for tci_next in sgm_lr_map.get(tci, [])]
        if tc_idx_r in tc_next:
            return True
        tc = tc_next
    return False


def get_sgm_lr_map(sgm_g):
    """ creates dictionry of connections between vertices:
    sgm_lr_map[tc_from]=list(tc_to)
    """
    sgm_lr_map = {}
    for grp_sgm in sgm_g:
        for sgm in grp_sgm:
            tc_from = sgm.l_links[0][0]
            tc_to = sgm.r_links[0][1]
            if tc_from not in sgm_lr_map:
                sgm_lr_map[tc_from] = [tc_to]
            else:
                sgm_lr_map[tc_from].append(tc_to)
    return sgm_lr_map


def search_vtx_conn(vtx_tc_idx, stack, sgm_g,
                    merge_jmp_dr_sqrt_mean_std, flow_jmp_dr_sqrt_mean_std, w_nc_0=cfgm.W_NC_0,
                    dt_mrg_rng=cfgm.SEARCH_MERGE_VTX_TIDX_RANGE, dt_flw_rng=cfgm.SEARCH_FLOW_VTX_TIDX_RANGE):
    # dt in [dt_mrg_rng)
    fid_vol = stack.f_fid_vol()
    vtx_tc_idx = set(vtx_tc_idx)

    # segmentation-jump
    drjs_mean, drjs_std = merge_jmp_dr_sqrt_mean_std
    drsm_max_srch = drjs_mean + 4 * drjs_std  # square root of search radius

    # flow-jump
    drjf_mean, drjf_std = flow_jmp_dr_sqrt_mean_std
    dt_mrg_max = np.abs(dt_mrg_rng).max() - 1
    dt_flw_max = np.abs(dt_flw_rng).max() - 1
    dt_max = max(dt_mrg_max, dt_flw_max)

    drsf_max_srch = max(drjf_mean * dt_flw_max, dt_flw_max * 3 * drjf_std)  # square root of search radius

    dr_max_srch = max(drsm_max_srch ** 2, drsf_max_srch)  # ** 2

    conns = []  # (from_tc_idx, to_tc_idx, w)
    sgm_lr_map = get_sgm_lr_map(sgm_g)

    # loop through all vtx. find possible r-conns (use both directions):


    # segmentation jump weight
    def link_ws_dr(dr):
        return ((sqrt(dr) - drjs_mean) / 1.5 / drjs_std) ** 2 * (0.5 if sqrt(dr) < drjs_mean else 1)

    def link_ws_drv(dr):
        return link_ws_dr(sqrt(dr[0] ** 2 + dr[1] ** 2))

    # flow jump weight
    # ToDo: ensure same weight calc in chi2 for tracks in merger
    def link_wf_sdr_dx_tan(sdr, dx, tn, dt):
        return (((sdr - drjf_mean) / drjf_std) ** 2 + atan(tn) * 4 / np.pi * 9 * sqrt(dt)) if (
                dx < 0 and tn < 1) else 100

    def link_wf_drv(dr, dt):
        return link_wf_sdr_dx_tan(sqrt(dr[0] ** 2 + dr[1] ** 2) / max(1, dt),
                                  dr[0],
                                  abs(dr[1] / dr[0]) if dr[0] != 0 else 1000,
                                  max(1, dt))

    def sdr_dx_tan(dr):
        return sqrt(dr[0] ** 2 + dr[1] ** 2), dr[0], abs(dr[1] / dr[0]) if dr[0] != 0 else 1000

    ws_per_dt = w_nc_0 / dt_mrg_max if dt_mrg_max > 1 else 0  # dt=1:w=0, dt=max: 8, with nll=1 will be thres, 9
    wf_per_dt = w_nc_0 / dt_flw_max if dt_flw_max > 1 else 0  # dt=1:w=0, dt=max: 6, with nll=3 will be thres, 9

    for tc_l in vtx_tc_idx:
        t_idx_l, c_idx_l = tc_l
        cell_l = stack.st[t_idx_l].cells[c_idx_l]
        cell_l_r = cell_l.get_r()
        w_nc = w_nc_fid_vol_corr(w_nc_0, fid_vol, cell_l_r)

        cells_r = []
        dbg_prnt = False  # tc_l in [(83,233), (30,199), (113,262), (110,144)]
        if dbg_prnt:
            print(f'left other_cell: {tc_l}, search rad {dr_max_srch}')

        # find all vertices not connected by segment sequence to tc_l within dt_max and dr_max_srch
        for dt in range(0, dt_max):
            t_idx_r = t_idx_l + dt
            if t_idx_r not in stack.st:
                continue

            cells_dt = stack.st[t_idx_r].obj_around_obj(cell_l, dr_max_srch)  # all arpund
            cells_dt = [cell for cell in cells_dt if ((t_idx_r, cell.idx) in vtx_tc_idx and
                                                      (t_idx_r, cell.idx) != tc_l and
                                                      not vtx_connected_by_segments(tc_l,
                                                                                    (t_idx_r, cell.idx),
                                                                                    sgm_lr_map
                                                                                    )
                                                      )]  # only other vtx

            cells_r.extend(cells_dt)

        for cell_r in cells_r:
            t_idx_r = cell_r.t_idx
            tc_r = (t_idx_r, cell_r.idx)
            cell_r_r = cell_r.get_r()
            link_dr = (cell_r_r[0] - cell_l_r[0], cell_r_r[1] - cell_l_r[1])
            dt = t_idx_r - t_idx_l

            ws = link_ws_drv(link_dr) + ws_per_dt * max(0, dt - 1)  # ndf=2
            ws /= sqrt(2)  # ~ ndf=1
            if t_idx_l + dt_mrg_rng[0] <= t_idx_r < t_idx_l + dt_mrg_rng[1]:
                # if tc_l==(144,6):
                #    print(tc_r, link_dr, link_ws_drv(link_dr), ws_per_dt*max(0, dt-1))
                if ws < w_nc:
                    conns.append((tc_l, tc_r, ws, link_dr, Segment.st_merge_jump))
                    # if cell_r.t_idx > tc_l[0]:
                    #    conns.append((tc_r, tc_l, w+1, link_dr))  # higher w for reversed

            wf = link_wf_drv(link_dr, dt) + wf_per_dt * max(0, dt - 1)  # ndf=2
            wf /= sqrt(2)  # ~ ndf=1

            if t_idx_l + dt_flw_rng[0] <= t_idx_r < t_idx_l + dt_flw_rng[1]:
                if wf < w_nc:
                    conns.append((tc_l, tc_r, wf, link_dr, Segment.st_flow_jump))
            if dbg_prnt:
                print(
                    f'right other_cell: {tc_r}, ws={ws}, wf={wf}[{link_wf_drv(link_dr, dt)} + {wf_per_dt * max(0, dt - 1)}                 {sdr_dx_tan(link_dr)}]')

    return list(set(conns))


def get_conn_segments(vtx_merge_conns, lc_map, stack):
    sgm_c = []
    for vtx_conn in vtx_merge_conns:
        *tc_from_to, w, dr, sgm_type = vtx_conn

        sgm = Segment(lc_map, sgm_type)

        sgm.add_l_link(tc_from_to)
        sgm.add_r_link(tc_from_to)
        sgm.fill_struct(stack)
        sgm.w = w

        sgm_c.append(sgm)
    return sgm_c


def get_disjointed_groups_segments(vtx_g, sgm_g, sgm_c, starts_g, only_connected_sgm):
    gi_to_idx = {}
    tc_to_idx = {}
    idx_to_gi = []
    tc_to_g2i = {}

    idx = 0
    for grp_idx, grp_vtx in enumerate(vtx_g):
        for vtx_idx, vtx in enumerate(grp_vtx):
            gi = (grp_idx, vtx_idx)
            tc = (vtx.t_idx, vtx.c_idx)
            gi_to_idx[gi] = idx
            tc_to_idx[tc] = idx
            idx_to_gi.append(gi)
            idx += 1
    n_nodes = idx
    list_idx = list(range(n_nodes))

    links = []
    for grp_sgm in sgm_g:
        for sgm in grp_sgm:
            tc_from = sgm.l_links[0][0]
            tc_to = sgm.r_links[0][1]
            if -1 not in tc_from and -1 not in tc_to:
                links.append((tc_to_idx[tc_from], tc_to_idx[tc_to]))

    sgm_c_g = sgm_c if (len(sgm_c) == 0 or (sgm_c[0]).__class__ == list) else [
        sgm_c]  # for both grouped and global sgm_c

    for grp_sgm_c in sgm_c_g:
        for sgm in grp_sgm_c:
            if (not only_connected_sgm) or (sgm.flow_slv > 0):
                tc_from = sgm.l_links[0][0]
                tc_to = sgm.r_links[0][1]
                if -1 not in tc_from and -1 not in tc_to:
                    links.append((tc_to_idx[tc_from], tc_to_idx[tc_to]))

    groups = get_disjointed_groups_links(list_idx, links)
    n_gr = max(1, len(groups))
    # print(n_gr)

    idx_to_group = [0 for _ in range(n_nodes)]
    for grp_idx, grp in enumerate(groups):
        for idx in grp:
            idx_to_group[idx] = grp_idx

    vtx_g2 = [[] for _ in range(n_gr)]
    sgm_g2 = [[] for _ in range(n_gr)]
    sgm_c_g2 = [[] for _ in range(n_gr)]
    starts_g2 = [[] for _ in range(n_gr)]

    for grp_idx, grp_vtx in enumerate(vtx_g):
        for vtx_idx, vtx in enumerate(grp_vtx):
            tc = (vtx.t_idx, vtx.c_idx)
            idx = tc_to_idx[tc]
            grp_idx = idx_to_group[idx]
            vtx_g2[grp_idx].append(vtx)
            tc_to_g2i[tc] = grp_idx

    for grp_sgm in sgm_g:
        for sgm in grp_sgm:
            tc_from = sgm.l_links[0][0]
            tc_to = sgm.r_links[0][1]
            idx = tc_to_idx.get(tc_from) or tc_to_idx.get(tc_to)

            grp_idx = 0 if idx is None else idx_to_group[idx]
            sgm_g2[grp_idx].append(sgm)
            for tc in sgm.nodes:
                tc_to_g2i[tc] = grp_idx

    for grp_sgm_c in sgm_c_g:
        for sgm in grp_sgm_c:
            if (not only_connected_sgm) or (
                    sgm.flow_slv > 0):  # if `only_connected_sgm` is set, 0-f segs will be skipped
                tc_from = sgm.l_links[0][0]
                tc_to = sgm.r_links[0][1]
                idx = tc_to_idx.get(tc_from) or tc_to_idx.get(tc_to)

                grp_idx = 0 if idx is None else idx_to_group[idx]
                sgm_c_g2[grp_idx].append(sgm)
                for tc in sgm.nodes:
                    tc_to_g2i[tc] = grp_idx

    for grp_starts in starts_g:
        for start in grp_starts:
            tc = start[0]

            if tc not in tc_to_idx:
                continue  # skip ones that were removed, probably upon shaving

            # print(tc)
            grp_idx = tc_to_g2i[tc]
            starts_g2[grp_idx].append(start)

    return vtx_g2, sgm_g2, sgm_c_g2, starts_g2


# global solve flow (as a whole):
def w_nc_time_fact(t_idx):
    """
    Returns left (appearence) and right (disappearance) factors for not-connected weight.
    accounts for accumulations phase, wash out, etc
    """
    t_accum_on = min(1, cfgm.T_ACCUMULATION_START)
    t_accum_stop = cfgm.T_ACCUMULATION_END
    t_accum_off = cfgm.T_ACCUMULATION_COMPLETE

    f_l = 1.
    f_r = 1.
    if t_idx < t_accum_on:
        f_l = (1 / cfgm.W_NC_0)
    elif t_idx < t_accum_off:
        f_l = 1 / cfgm.W_NC_0 * (1 + (cfgm.W_NC_0 - 1) * (t_idx - t_accum_on) / (t_accum_off - t_accum_on))

    if t_idx < t_accum_stop:
        f_r = 0.5
    elif t_accum_stop <= t_idx <= t_accum_off:
        f_r = 2 / cfgm.W_NC_0
    return f_l, f_r


def solve_groups_global_flow_whole(stack, vtx_g, sgm_g, sgm_c, w_nc, w_f_mult_end, w_f_above_est):
    tc_to_vidx = {}  # v for vertex idx

    vtx_w_nc_lr = []

    segs_ends = []
    segs_f_est = []
    segs_f_hnt = []
    psegs_ends = []
    psegs_w = []

    node_idx_to_grp_vtx_idx = []
    link_idx_to_grp_sgm_idx = []
    plink_idx_to_grp_sgm_idx = []

    vtx_global_idx = 0

    # segment types:  0 - seg_g, 1 - seg_c

    fid_vol = stack.f_fid_vol()
    for grp_idx, (grp_vtx, grp_sgm) in enumerate(zip(vtx_g, sgm_g)):
        for vtx_idx, vtx in enumerate(grp_vtx):
            tidx, cidx = vtx.t_idx, vtx.c_idx
            tcidx = tidx, cidx
            cell = stack.st[tidx].cells[cidx]
            r = cell.get_r()

            w_nc_cell = w_nc_fid_vol_corr(w_nc, fid_vol, r)  # to be corrected with time

            w_nc_time_fact_lr = w_nc_time_fact(tidx)

            vtx_w_nc_lr.append((w_nc_cell * w_nc_time_fact_lr[0], w_nc_cell * w_nc_time_fact_lr[1]))

            tc_to_vidx[tcidx] = vtx_global_idx
            node_idx_to_grp_vtx_idx.append((grp_idx, vtx_idx))
            vtx_global_idx += 1

        for sgm_idx, sgm in enumerate(grp_sgm):
            node_l_tc_idx = sgm.l_links[0][0]
            node_r_tc_idx = sgm.r_links[0][1]
            segs_ends.append((tc_to_vidx[node_l_tc_idx], tc_to_vidx[node_r_tc_idx]))
            segs_f_est.append(sgm.flow_median)
            segs_f_hnt.append(sgm.flow_slv)
            link_idx_to_grp_sgm_idx.append((grp_idx, sgm_idx))
            sgm.flow_slv = 0

    for sgm_idx, sgm in enumerate(sgm_c):
        node_l_tc_idx = sgm.l_links[0][0]
        node_r_tc_idx = sgm.r_links[0][1]
        psegs_ends.append((tc_to_vidx[node_l_tc_idx], tc_to_vidx[node_r_tc_idx]))
        psegs_w.append(sgm.w)
        plink_idx_to_grp_sgm_idx.append(sgm_idx)
        sgm.flow_slv = 0

    print('n_vtx = %d' % len(vtx_w_nc_lr), 'n_segm = %d' % len(segs_ends), 'n_segm_p = %d' % len(psegs_ends))

    resolved_links, *_ = solve_flow(vtx_w_nc_lr, segs_ends, segs_f_est, segs_f_hnt, psegs_ends, psegs_w,
                                    fid_vol, w_nc, w_f_mult_end, w_f_above_est
                                    )

    for link in resolved_links:
        (from_idx, to_idx), lnk_info, f = link
        if lnk_info == -1:
            if from_idx == -2:
                grp_idx, vtx_idx = node_idx_to_grp_vtx_idx[to_idx]
                vtx = vtx_g[grp_idx][vtx_idx]
                vtx.l_nc_f_slv = f
            elif to_idx == -1:
                grp_idx, vtx_idx = node_idx_to_grp_vtx_idx[from_idx]
                vtx = vtx_g[grp_idx][vtx_idx]
                vtx.r_nc_f_slv = f
        else:
            link_type, lnk_idx = lnk_info
            if link_type == 0:
                grp_idx, sgm_idx = link_idx_to_grp_sgm_idx[lnk_idx]
                # print(lnk_idx, grp_idx, sgm_idx)
                sgm = sgm_g[grp_idx][sgm_idx]
                sgm.flow_slv = f
            elif link_type == 1:
                sgm_idx = plink_idx_to_grp_sgm_idx[lnk_idx]
                # print(lnk_idx, sgm_idx)
                sgm = sgm_c[sgm_idx]
                sgm.flow_slv = f


# global solve flow (in disjoint groups):
def solve_groups_global_flow(stack, vtx_g, sgm_g, sgm_gc, w_nc, w_f_mult_end, w_f_above_est, timeout_sec=None):
    fid_vol = stack.f_fid_vol()
    success_g = []
    for grp_idx, (grp_vtx, grp_sgm, grp_sgmc) in enumerate(tqdm(zip(vtx_g, sgm_g, sgm_gc), desc='segment multiplicity global', ascii=True)):
        # print(grp_idx, '/', len(vtx_g), end='\r')

        tc_to_vidx = {}  # v for vertex idx

        vtx_w_nc_lr = []

        segs_ends = []
        segs_f_est = []
        segs_f_hnt = []
        psegs_ends = []
        psegs_w = []

        for vtx_idx, vtx in enumerate(grp_vtx):
            tidx, cidx = vtx.t_idx, vtx.c_idx
            tcidx = tidx, cidx
            cell = stack.st[tidx].cells[cidx]
            r = cell.get_r()

            w_nc_cell = w_nc_fid_vol_corr(w_nc, fid_vol, r)  # to be corrected with time

            w_nc_time_fact_lr = w_nc_time_fact(tidx)

            vtx_w_nc_lr.append((w_nc_cell * w_nc_time_fact_lr[0], w_nc_cell * w_nc_time_fact_lr[1]))

            tc_to_vidx[tcidx] = vtx_idx
            vtx.l_nc_f_slv = 0
            vtx.r_nc_f_slv = 0

        for sgm in grp_sgm:
            node_l_tc_idx = sgm.l_links[0][0]
            node_r_tc_idx = sgm.r_links[0][1]
            segs_ends.append((tc_to_vidx[node_l_tc_idx], tc_to_vidx[node_r_tc_idx]))
            segs_f_est.append(sgm.flow_median)
            segs_f_hnt.append(sgm.flow_slv)
            sgm.flow_slv = 0

        for sgm in grp_sgmc:
            node_l_tc_idx = sgm.l_links[0][0]
            node_r_tc_idx = sgm.r_links[0][1]
            psegs_ends.append((tc_to_vidx[node_l_tc_idx], tc_to_vidx[node_r_tc_idx]))
            psegs_w.append(sgm.w)
            sgm.flow_slv = 0

        # print('n_vtx = %d' % len(vtx_w_nc_lr), 'n_segm = %d'%len(segs_ends), 'n_segm_p = %d'%len(psegs_ends))

        resolved_links, *_ = solve_flow(vtx_w_nc_lr, segs_ends, segs_f_est, segs_f_hnt, psegs_ends, psegs_w,
                                        fid_vol, w_nc, w_f_mult_end, w_f_above_est, timeout_sec=timeout_sec
                                        )

        for link in resolved_links:
            (from_idx, to_idx), lnk_info, f = link
            if lnk_info == -1:
                if from_idx == -2:
                    grp_vtx[to_idx].l_nc_f_slv = f
                elif to_idx == -1:
                    grp_vtx[from_idx].r_nc_f_slv = f
            else:
                link_type, lnk_idx = lnk_info
                if link_type == 0:
                    grp_sgm[lnk_idx].flow_slv = f
                elif link_type == 1:
                    grp_sgmc[lnk_idx].flow_slv = f

        success = False if (len(resolved_links) == 0 and (len(segs_ends) + len(psegs_ends)) > 0) else True
        success_g.append(success)
    print('')
    # success if at least one segment is resolved when somethig was to be resolved for all groups
    return all(success_g)


def remove_small_groups(vtx_g, sgm_g, sgm_c_g, all_starts_g, n_node_min=6):
    vtx_g2, sgm_g2, sgm_c_g2, all_starts_g2 = [], [], [], []

    for grp_vtx, grp_sgm, grp_sgm_c, grp_starts in zip(vtx_g, sgm_g, sgm_c_g, all_starts_g):
        seg_len = [len(sgm.nodes) for sgm in grp_sgm]
        tot_seg_len = np.sum(seg_len)
        if tot_seg_len >= n_node_min:
            vtx_g2.append(grp_vtx)
            sgm_g2.append(grp_sgm)
            sgm_c_g2.append(grp_sgm_c)
            all_starts_g2.append(grp_starts)

    return vtx_g2, sgm_g2, sgm_c_g2, all_starts_g2


def shave_and_solve_groups_global_flow(stack, vtx_g, sgm_g, sgm_c_g):
    t1 = time.time()
    if cfgm.SHAVING_SOLVER_TIMEOUT != 0 and cfgm.SHAVING_SOLVER_MAX_ITER != 0:
        n_grp = len(vtx_g)
        for gr_idx, (grp_vtx, grp_sgm, grp_sgm_c) in enumerate(zip(vtx_g, sgm_g, sgm_c_g)):
            print(f'Shaving group {gr_idx}/{n_grp}')
            vtx_gi, sgm_gi, sgm_c_gi = [grp_vtx], [grp_sgm], [grp_sgm_c]

            itr_idx = -1
            while True if cfgm.SHAVING_SOLVER_MAX_ITER == -1 else (itr_idx < cfgm.SHAVING_SOLVER_MAX_ITER):
                itr_idx += 1

                n_shaved_tot = shave_segments_g(vtx_gi, sgm_gi, sgm_c_gi)
                if n_shaved_tot == 0:
                    break
                print('\tshaved', n_shaved_tot)

                res = False
                # sort the segments in sgm_c_gi by weight, ascending order

                sgm_c_gi0 = [sorted(sgm_c_gi[0], key=lambda sgm: sgm.w)]
                n_seg_c = len(sgm_c_gi0[0])
                for remove_worst_sgm_c_perc in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
                    if remove_worst_sgm_c_perc > 0:
                        print(f'\nSolving failed. trying with removed {remove_worst_sgm_c_perc}% of ' +
                              f'the worst segment candidates')

                    n_seg_remaining = int(n_seg_c * (1 - remove_worst_sgm_c_perc / 100))
                    sgm_c_gi = [sgm_c_gi0[0][:n_seg_remaining]]

                    res = solve_groups_global_flow(stack=stack,
                                                   vtx_g=vtx_gi, sgm_g=sgm_gi, sgm_gc=sgm_c_gi,
                                                   w_nc=cfgm.W_NC_GLOB,
                                                   w_f_mult_end=cfgm.W_F_MULT_END_GLOB_SHAVED,
                                                   w_f_above_est=cfgm.W_F_ABOVE_EST_GLOB,
                                                   timeout_sec=cfgm.SHAVING_SOLVER_TIMEOUT
                                                   )

                    if res:
                        break  # if solved, stop here

                if not res:
                    print(f'\tgroup {gr_idx} not solved after {itr_idx} iterations during shaving (G3)')

                sgm_g[gr_idx], sgm_c_g[gr_idx] = sgm_gi[0], sgm_c_gi[0]

    t2 = time.time()
    print('time spent shaving = %.2fs' % (t2 - t1))
    return sgm_g, sgm_c_g


# method removes segments connecting hanging vtx-es, per group
def shave_segments(grp_vtx, grp_sgm, grp_sgm_c, free_vtx_min_nodes=6):
    stp_n = 0
    stp_c = 1

    tc_to_l_links = {}  # list, (type, idx)
    tc_to_r_links = {}
    tc_to_l_links_act = {}
    tc_to_r_links_act = {}

    tc_to_vtx_idx = {(vtx.t_idx, vtx.c_idx): idx for idx, vtx in enumerate(grp_vtx)}

    grp_sgm_all = [grp_sgm, grp_sgm_c]
    for segs, stp in zip(grp_sgm_all, [stp_n, stp_c]):
        for idx, seg in enumerate(segs):
            tc_from = seg.l_links[0][0]
            tc_to = seg.r_links[0][1]

            if tc_to not in tc_to_l_links:
                tc_to_l_links[tc_to] = []
            tc_to_l_links[tc_to].append((stp, idx))

            if tc_from not in tc_to_r_links:
                tc_to_r_links[tc_from] = []
            tc_to_r_links[tc_from].append((stp, idx))

    for tc, links in tc_to_l_links.items():
        tc_to_l_links_act[tc] = [(stp, idx) for (stp, idx) in links if grp_sgm_all[stp][idx].flow_slv > 0]

    for tc, links in tc_to_r_links.items():
        tc_to_r_links_act[tc] = [(stp, idx) for (stp, idx) in links if grp_sgm_all[stp][idx].flow_slv > 0]

    # print(tc_to_l_links)
    # print(tc_to_l_links_act)

    excluded_stp_idx = set()

    for vtx in grp_vtx:
        tc = vtx.t_idx, vtx.c_idx
        l_links = tc_to_l_links_act.get(tc, [])
        r_links = tc_to_r_links_act.get(tc, [])

        # left: >1 conn, if exist > 1 vtx/>0 node per seg - then remove all with only 1vtx & 0 node/seg

        excl_candidates_l = []
        excl_candidates_r = []
        n_l = len(l_links)
        n_r = len(r_links)

        if n_l > 1:
            for stp, idx in l_links:
                sgm = grp_sgm_all[stp][idx]
                tc_next_node_l = sgm.l_links[0][0]
                n_next_l = len(tc_to_l_links_act.get(tc_next_node_l, []))
                if n_next_l == 0 and len(sgm.nodes) < free_vtx_min_nodes:
                    excl_candidates_l.append((stp, idx))
        if n_r > 1:
            for stp, idx in r_links:
                sgm = grp_sgm_all[stp][idx]
                tc_next_node_r = sgm.r_links[0][1]
                n_next_r = len(tc_to_r_links_act.get(tc_next_node_r, []))
                if n_next_r == 0 and len(sgm.nodes) < free_vtx_min_nodes:
                    excl_candidates_r.append((stp, idx))

        if n_l > 1 and 0 < len(excl_candidates_l) < n_l:  # can't remove all
            excluded_stp_idx.update(excl_candidates_l)
        if n_r > 1 and 0 < len(excl_candidates_r) < n_r:
            excluded_stp_idx.update(excl_candidates_r)

    # print(len(excluded_stp_idx))
    sgm_shaved = [grp_sgm_all[stp][idx] for stp, idx in excluded_stp_idx if stp == stp_n]
    sgm_c_shaved = [grp_sgm_all[stp][idx] for stp, idx in excluded_stp_idx if stp == stp_c]

    grp_sgm_new = [sgm for idx, sgm in enumerate(grp_sgm) if (stp_n, idx) not in excluded_stp_idx]
    grp_sgm_c_new = [sgm for idx, sgm in enumerate(grp_sgm_c) if (stp_c, idx) not in excluded_stp_idx]

    return grp_sgm_new, grp_sgm_c_new, sgm_shaved, sgm_c_shaved


def shave_segments_g(vtx_g, sgm_g, sgm_c_g):
    sgm_shaved_all = []
    sgm_c_shaved_all = []
    for idx, (grp_vtx, grp_sgm, grp_sgm_c) in enumerate(zip(vtx_g, sgm_g, sgm_c_g)):
        grp_sgm_new, grp_sgm_c_new, sgm_shaved, sgm_c_shaved = shave_segments(grp_vtx, grp_sgm, grp_sgm_c)
        sgm_g[idx], sgm_c_g[idx] = grp_sgm_new, grp_sgm_c_new
        sgm_shaved_all.append(sgm_shaved)
        sgm_c_shaved_all.append(sgm_c_shaved)

    n_shaved = np.sum([len(sgm) for sgm in sgm_shaved_all])
    n_shaved_c = np.sum([len(sgm) for sgm in sgm_c_shaved_all])
    n_shaved_tot = n_shaved + n_shaved_c

    return n_shaved_tot


def plot_seg_flow(vtx_g, sgm_g, sgm_c_g, all_starts_g, stack, all_tracks_start_tcidx, all_tracks_xyt, saveto=None):
    for gr_idx in range(len(vtx_g)):
        grp_starts = all_starts_g[gr_idx]

        grp_vtx = vtx_g[gr_idx]
        grp_sgm = sgm_g[gr_idx]
        grp_sgm_c = sgm_c_g[gr_idx]

        plot_seg_flow_grp(grp_vtx, grp_sgm, grp_sgm_c, grp_starts, stack, all_tracks_start_tcidx, all_tracks_xyt, saveto, idx=gr_idx)


def plot_seg_flow_grp(grp_vtx, grp_sgm, grp_sgm_c, grp_starts, stack, all_tracks_start_tcidx, all_tracks_xyt, saveto=None, idx=0):
    has_mflow = False  # len(grp_vtx) > 300

    # colors = ['pink', 'blue', 'green', 'yellow', 'orange', 'red']
    colors = ['k', 'b', 'g', 'y', 'orange', 'r']

    s = 14
    fig, ax = plt.subplots(1, 3, figsize=(3 * s, s))
    all_segments_plots = [[], [], []]

    for start in grp_starts:
        track_tcids_idx = all_tracks_start_tcidx[start[0]]
        track_xyt = all_tracks_xyt[track_tcids_idx]
        track_xyt = np.array(track_xyt)
        x, y, t = track_xyt.transpose()

        ax[0].plot(t, x, 'grey')
        ax[0].set_xlabel('t')
        ax[0].set_ylabel('x')
        ax[1].plot(t, y, 'grey')
        ax[1].set_xlabel('t')
        ax[1].set_ylabel('y')
        ax[2].plot(x, y, 'grey')
        ax[2].set_xlabel('x')
        ax[2].set_ylabel('y')
        ax[2].set_aspect('equal')

    vtx_xyt = [cell_r_ofs_t(stack.st[vtx.t_idx].cells[vtx.c_idx]) for vtx in grp_vtx]
    vtx_xyt = np.array(vtx_xyt)
    # vx, vy, vt = vtx_xyt.transpose()
    #     ax[0].scatter(vt, vx, s = 64, c='k')
    #     ax[1].scatter(vt, vy, s = 64, c='k')
    #     ax[2].scatter(vx, vy, s = 64, c='k')

    for sgm in grp_sgm:
        tc_i = sgm.l_links[0][0]
        tc_o = sgm.r_links[0][1]
        s_i_xyt = cell_r_ofs_t(stack.st[tc_i[0]].cells[tc_i[1]])
        s_o_xyt = cell_r_ofs_t(stack.st[tc_o[0]].cells[tc_o[1]])

        col = colors[sgm.flow_slv]
        all_segments_plots[0].extend([[s_i_xyt[2], s_o_xyt[2]], [s_i_xyt[0], s_o_xyt[0]], col])
        all_segments_plots[1].extend([[s_i_xyt[2], s_o_xyt[2]], [s_i_xyt[1], s_o_xyt[1]], col])
        all_segments_plots[2].extend([[s_i_xyt[0], s_o_xyt[0]], [s_i_xyt[1], s_o_xyt[1]], col])

    for sgm in grp_sgm_c:
        if sgm.flow_slv == 0:
            continue
        tc_i = sgm.l_links[0][0]
        tc_o = sgm.r_links[0][1]
        s_i_xyt = cell_r_ofs_t(stack.st[tc_i[0]].cells[tc_i[1]])
        s_o_xyt = cell_r_ofs_t(stack.st[tc_o[0]].cells[tc_o[1]])

        col = '--' + colors[sgm.flow_slv]
        all_segments_plots[0].extend([[s_i_xyt[2], s_o_xyt[2]], [s_i_xyt[0], s_o_xyt[0]], col])
        all_segments_plots[1].extend([[s_i_xyt[2], s_o_xyt[2]], [s_i_xyt[1], s_o_xyt[1]], col])
        all_segments_plots[2].extend([[s_i_xyt[0], s_o_xyt[0]], [s_i_xyt[1], s_o_xyt[1]], col])

    ax[0].plot(*(all_segments_plots[0]), alpha=0.5)
    ax[1].plot(*(all_segments_plots[1]), alpha=0.5)
    ax[2].plot(*(all_segments_plots[2]), alpha=0.5)
    plt.suptitle('xt, yt plots of intersecting tracks within group %s, n_tr=%d' % (str(idx),
                                                                                   len(grp_starts)
                                                                                   ))
    plt.tight_layout(pad=0.5, h_pad=0, w_pad=0)
    if saveto:
        plt.savefig(saveto + '\\%03d' % idx)
    if has_mflow or saveto is None:
        plt.show()
    plt.close()


def plot_seg_flow_change_dist(sgm_g, saveto):
    # distribution of f_slv-f_est
    df_arr = []
    for segs in sgm_g:
        for sgm in segs:
            # print (len(sgm.nodes), sgm.flow_slv)
            if sgm.flow_slv > 0 and len(sgm.nodes) > 6:
                df_arr.append(sgm.flow_slv - sgm.flow_median)
    _ = plt.hist(df_arr, 100, log=True)
    _ = plt.title('flow segmentation error')
    plt.savefig(saveto)
    plt.close()


def merge_selected_potential_segments(vtx_g, sgm_g, sgm_c_g, all_starts_g):
    skip_grps = set()
    for grp_idx, (grp_sgm, grp_sgm_c) in enumerate(zip(sgm_g, sgm_c_g)):
        # get min and max flow segments in grp_sgm and in grp_sgm_c
        f_gs = [sgm.flow_slv for sgm in grp_sgm]
        f_gsc = [sgm.flow_slv for sgm in grp_sgm_c]

        f_gs_r = min_max(f_gs)
        f_gsc_r = min_max(f_gsc)

        grp_sgm.extend(grp_sgm_c)

        bad_grp = False
        try:
            for sgm in grp_sgm:
                assert (sgm.flow_slv > 0), f'sgm.flow_slv={sgm.flow_slv} > 0'
        except AssertionError as err:
            print(err.args, f'sgm_g[{grp_idx}] flow range={f_gs_r}, sgm_c_g[{grp_idx}] flow range={f_gsc_r}')
            bad_grp = True

        if bad_grp:
            skip_grps.add(grp_idx)

    del sgm_c_g

    print('skipped unresolved groups:', skip_grps)

    vtx_g = [itm for idx, itm in enumerate(vtx_g) if idx not in skip_grps]
    sgm_g = [itm for idx, itm in enumerate(sgm_g) if idx not in skip_grps]
    all_starts_g = [itm for idx, itm in enumerate(all_starts_g) if idx not in skip_grps]
    return vtx_g, sgm_g, None, all_starts_g


# finding long segments and crossings
def get_full_sgm_from(vtx0, sgm0, grp_sgm, xing_vtx_tc, vtx_tc_to_r_sgm_idx_arr, stack):
    sgm: Segment = Segment(sgm0.lc_map, Segment.st_merged)
    flow = sgm0.flow_slv
    sgm.flow_slv = flow

    # 0. join starting vertex
    if vtx0:
        # 1. append to sgm
        tc_v0 = (vtx0.t_idx, vtx0.c_idx)
        sgm.add_node(tc_v0)
        # 2. set input link
        sgm.add_l_link(((-1, -1), tc_v0))  # no left node, this is start
    else:
        # 2. set input link
        for lnk in sgm0.l_links:
            sgm.add_l_link(lnk)

    curr_sgm: Segment or None = sgm0

    while True:
        assert (curr_sgm.flow_slv == flow)
        # 3. append curr_sgm to sgm
        for node_tc in curr_sgm.nodes:
            sgm.add_node(node_tc)
        for mj_t_span in curr_sgm.mj_t_ranges:
            sgm.add_merge_jump_timespan(*mj_t_span)
        for fj_t_span in curr_sgm.fj_t_ranges:
            sgm.add_flow_jump_timespan(*fj_t_span)

        # 4. find next vtx
        next_vtx_tc = curr_sgm.r_links[0][1]
        if next_vtx_tc in xing_vtx_tc or next_vtx_tc == (-1, -1):
            break

        # 5. else - add vtx
        sgm.add_node(next_vtx_tc)

        # 6. contunue to next segment
        next_sgm_idx = vtx_tc_to_r_sgm_idx_arr.get(next_vtx_tc, [])
        n_links = len(next_sgm_idx)
        assert (n_links in [0, 1])

        if n_links == 0:
            sgm.add_r_link((next_vtx_tc, (-1, -1)))
            curr_sgm = None
            break

        curr_sgm = grp_sgm[next_sgm_idx[0]]

    # 7. set out link by r link of curr_sgm
    if curr_sgm is not None:
        for lnk in curr_sgm.r_links:
            sgm.add_r_link(lnk)

    sgm.fill_struct(stack)
    return sgm


# will be fn per group:
def restructure_segments_grp(grp_vtx, grp_sgm, stack,
                             n_node_min_eval=6):  # it's a shit name :/.  restructure group for X-ing resolving
    # 1. vtx connectivity map:
    vtx_tc_to_l_sgm_idx_arr = {}  # idx of all segments connected to a vtx on the left
    vtx_tc_to_r_sgm_idx_arr = {}  # idx of all segments connected to a vtx on the right
    for sidx, sgm in enumerate(grp_sgm):
        l_tc = sgm.l_links[0][0]
        r_tc = sgm.r_links[0][1]

        if r_tc not in vtx_tc_to_l_sgm_idx_arr:
            vtx_tc_to_l_sgm_idx_arr[r_tc] = []
        vtx_tc_to_l_sgm_idx_arr[r_tc].append(sidx)

        if l_tc not in vtx_tc_to_r_sgm_idx_arr:
            vtx_tc_to_r_sgm_idx_arr[l_tc] = []
        vtx_tc_to_r_sgm_idx_arr[l_tc].append(sidx)

    # 2. get vtx belonging to crossings:

    xing_vtx_tc = set()  # tc-s of all vtx belonging to crossings
    start_vtx_idx = []  # idx of all vtx of segment starts (for seg search).
    for vidx, vtx in enumerate(grp_vtx):
        tc = vtx.t_idx, vtx.c_idx
        l_sgm_idx = vtx_tc_to_l_sgm_idx_arr.get(tc, [])
        r_sgm_idx = vtx_tc_to_r_sgm_idx_arr.get(tc, [])

        l_n_sgm = len(l_sgm_idx)
        l_f_sgm = np.sum([grp_sgm[sidx].flow_slv for sidx in l_sgm_idx])
        l_f_nc = vtx.l_nc_f_slv

        r_n_sgm = len(r_sgm_idx)
        r_f_sgm = np.sum([grp_sgm[sidx].flow_slv for sidx in r_sgm_idx])
        r_f_nc = vtx.r_nc_f_slv

        is_start = (l_f_nc > 0 and l_f_sgm == 0)
        has_lost_flow = (l_f_nc > 0 and l_f_sgm > 0) or (r_f_nc > 0 and r_f_sgm > 0)  # isn't end
        is_xing_vtx = l_n_sgm > 1 or r_n_sgm > 1  # merging several segments
        is_xing = is_xing_vtx or has_lost_flow

        if is_xing:
            # print(vidx, tc, l_n_sgm, l_f_sgm, l_f_nc, r_n_sgm, r_f_sgm, r_f_nc)
            xing_vtx_tc.add(tc)

        if is_start or is_xing:
            start_vtx_idx.append(vidx)

    # 3. build new segments starting from start_vtx_idx (including if tc not in xing_vtx_tc)
    grp_full_sgm = []

    for v_idx in start_vtx_idx:
        vtx = grp_vtx[v_idx]
        tc = vtx.t_idx, vtx.c_idx

        if tc in xing_vtx_tc:
            for sidx in vtx_tc_to_r_sgm_idx_arr.get(tc, []):
                sgm0 = grp_sgm[sidx]
                vtx0 = None
                fsgm = get_full_sgm_from(vtx0, sgm0, grp_sgm, xing_vtx_tc, vtx_tc_to_r_sgm_idx_arr, stack)
                grp_full_sgm.append(fsgm)
        else:
            sidxs = vtx_tc_to_r_sgm_idx_arr.get(tc, [])
            # ToDo: check ok w/o:
            # assert len(sidxs)==1, f'{len(sidxs)}'
            if len(sidxs) != 1:
                continue
            sidx = sidxs[0]

            sgm0 = grp_sgm[sidx]
            vtx0 = vtx
            fsgm = get_full_sgm_from(vtx0, sgm0, grp_sgm, xing_vtx_tc, vtx_tc_to_r_sgm_idx_arr, stack)
            grp_full_sgm.append(fsgm)

    # 4. group xing vtx by segments with flow>1 or too short for resolving
    tspan_to_dt = lambda t0t1: t0t1[1] - t0t1[0]

    all_sgm_xing = []
    all_sgm_fis1 = []
    for sgm in grp_full_sgm:
        if (sgm.flow_slv > 1 or
                tspan_to_dt(sgm.get_t_minmax()) < n_node_min_eval or
                len(sgm.node_map) < n_node_min_eval
        ):
            all_sgm_xing.append(sgm)
        else:
            all_sgm_fis1.append(sgm)

    all_vtx_xing = [vtx for vtx in grp_vtx if (vtx.t_idx, vtx.c_idx) in xing_vtx_tc]
    vtx_xing, *_ = get_disjointed_groups_segments([all_vtx_xing], [all_sgm_xing], [], [], only_connected_sgm=False)

    return vtx_xing, all_sgm_xing, all_sgm_fis1


def restructure_segments_grps(vtx_g, sgm_g, stack):
    vtx_xg = []
    sgm_xg = []
    sgm_1g = []
    for grp_vtx, grp_sgm in zip(vtx_g, sgm_g):
        # print('\n', len(grp_vtx), len(grp_sgm))
        vtx_xing, all_sgm_xing, all_sgm_fis1 = restructure_segments_grp(grp_vtx, grp_sgm, stack)
        vtx_xg.append(vtx_xing)
        sgm_xg.append(all_sgm_xing)
        sgm_1g.append(all_sgm_fis1)
        # print(len(vtx_xing), len(all_sgm_xing), len(all_sgm_fis1))
    return vtx_xg, sgm_xg, sgm_1g


def pickle_svs(st_merged, st_full, vtx_xg, sgm_xg, sgm_1g, sfx='', path='.'):
    sfxp = '_' + sfx if len(sfx) else ''

    save_pckl(st_merged, f'st_merged{sfxp}.pckl', path=path)
    save_pckl(st_full, f'st_full{sfxp}.pckl', path=path)
    save_pckl(vtx_xg, f'vtx_xg{sfxp}.pckl', path=path)
    save_pckl(sgm_xg, f'sgm_xg{sfxp}.pckl', path=path)
    save_pckl(sgm_1g, f'sgm_1g{sfxp}.pckl', path=path)


def load_svs(sfx='', path='.'):
    sfxp = '_' + sfx if len(sfx) else ''

    st_merged = load_pckl(f'st_merged{sfxp}.pckl', path=path)
    st_full = load_pckl(f'st_full{sfxp}.pckl', path=path)
    vtx_xg = load_pckl(f'vtx_xg{sfxp}.pckl', path=path)
    sgm_xg = load_pckl(f'sgm_xg{sfxp}.pckl', path=path)
    sgm_1g = load_pckl(f'sgm_1g{sfxp}.pckl', path=path)
    return st_merged, st_full, vtx_xg, sgm_xg, sgm_1g


def plot_xing_segs(vtx_xg, sgm_xg, sgm_1g, all_starts_g, stack, all_tracks_start_tcidx, all_tracks_xyt, saveto=None):
    for gr_idx in range(len(vtx_xg)):
        plot_xing_segs_grp(vtx_xg, sgm_xg, sgm_1g, all_starts_g, stack,
                           all_tracks_start_tcidx, all_tracks_xyt, saveto=saveto, gr_idx=gr_idx)


def plot_xing_segs_grp(vtx_xg, sgm_xg, sgm_1g, all_starts_g,
                       stack, all_tracks_start_tcidx, all_tracks_xyt, saveto=None, gr_idx=0):
    s = 14
    fig, ax = plt.subplots(1, 3, figsize=(3 * s, s))
    all_segments_plots = [[], [], []]

    grp_starts = all_starts_g[gr_idx]
    colors = ['k', 'b', 'g', 'y', 'orange', 'r']

    for start in grp_starts:
        track_tcids_idx = all_tracks_start_tcidx[start[0]]
        track_xyt = all_tracks_xyt[track_tcids_idx]
        track_xyt = np.array(track_xyt)
        x, y, t = track_xyt.transpose()

        ax[0].plot(t, x, 'grey')
        ax[0].set_xlabel('t')
        ax[0].set_ylabel('x')
        ax[1].plot(t, y, 'grey')
        ax[1].set_xlabel('t')
        ax[1].set_ylabel('y')
        ax[2].plot(x, y, 'grey')
        ax[2].set_xlabel('x')
        ax[2].set_ylabel('y')
        ax[2].set_aspect('equal')

    grp_full_sgm = sgm_xg[gr_idx] + sgm_1g[gr_idx]
    grp_full_sgm_idx = [-1] * len(sgm_xg[gr_idx]) + list(range(len(sgm_1g[gr_idx])))
    vtx_xing = vtx_xg[gr_idx]

    n_xing_vtx = 0
    for x_vtx in vtx_xing:
        vtx_xyt = [cell_r_ofs_t(stack.st[vtx.t_idx].cells[vtx.c_idx]) for vtx in x_vtx]
        vtx_xyt = np.array(vtx_xyt)
        n_vtx_x = len(vtx_xyt)
        n_xing_vtx += n_vtx_x
        if n_vtx_x:
            vx, vy, vt = vtx_xyt.transpose()
            ax[0].scatter(vt, vx, s=64)  # , c='k'
            ax[1].scatter(vt, vy, s=64)  # , c='k'
            ax[2].scatter(vx, vy, s=64)  # , c='k'

    tc_nc = (-1, -1)
    for sgm_idx, sgm in zip(grp_full_sgm_idx, grp_full_sgm):  # grp_sgm

        tc_i = [links[1] for links in sgm.l_links if links[0] != tc_nc and links[1] != tc_nc]
        tc_o = [links[0] for links in sgm.r_links if links[0] != tc_nc and links[1] != tc_nc]
        if len(tc_i) == 0 or len(tc_o) == 0:
            continue
        s_i_xyt = np.mean([cell_r_ofs_t(stack.st[tc[0]].cells[tc[1]]) for tc in tc_i], axis=0)
        s_o_xyt = np.mean([cell_r_ofs_t(stack.st[tc[0]].cells[tc[1]]) for tc in tc_o], axis=0)

        col = colors[sgm.flow_slv]
        all_segments_plots[0].extend([[s_i_xyt[2], s_o_xyt[2]], [s_i_xyt[0], s_o_xyt[0]], col])
        all_segments_plots[1].extend([[s_i_xyt[2], s_o_xyt[2]], [s_i_xyt[1], s_o_xyt[1]], col])
        all_segments_plots[2].extend([[s_i_xyt[0], s_o_xyt[0]], [s_i_xyt[1], s_o_xyt[1]], col])

        if sgm_idx >= 0:
            ax[0].text((s_i_xyt[2] + s_o_xyt[2]) / 2, (s_i_xyt[0] + s_o_xyt[0]) / 2, f'{sgm_idx}')
            ax[1].text((s_i_xyt[2] + s_o_xyt[2]) / 2, (s_i_xyt[1] + s_o_xyt[1]) / 2, f'{sgm_idx}')

    ax[0].plot(*(all_segments_plots[0]), alpha=0.5)
    ax[1].plot(*(all_segments_plots[1]), alpha=0.5)
    ax[2].plot(*(all_segments_plots[2]), alpha=0.5)
    plt.suptitle('xt, yt plots of intersecting tracks within group %s, n_tr=%d' % (str(gr_idx),
                                                                                   len(grp_starts)
                                                                                   ))
    plt.tight_layout(pad=0.5, h_pad=0, w_pad=0)

    if saveto is not None:
        plt.savefig('%s\\%03d' % (saveto, gr_idx))
    # if n_xing_vtx > 0:
    #     plt.show()
    plt.close()


def save_tracks_for_display_from_sgm_vtx(stack, vtx_xg, sgm_xg, sgm_1g, fname):
    # sgm_xg - segments of x-ings. not grouped by x-ing
    # sgm_1g - segments entering/exiting x-ings. not grouped by x-ing
    # vtx_xg - vertices of crossings, grouped by xing

    uidx = -1  # only container related

    tracks_xyz_t_id_uidx = []
    tracks_aux_pars = []
    tracks_aux_local_pars = []
    tracks_aux_map = None
    tracks_aux_local_map = None
    tracks_xyz_map = None

    # here we save only coords, dt, n_node & flow.
    tracks_f_tcs = []

    # all single-f segments:
    # all non-single-f/short segments (from xings):

    for sgm_g in [sgm_xg, sgm_1g]:
        for gr_idx, sgm_grp in enumerate(sgm_g):
            for sgm_idx, sgm in enumerate(sgm_grp):
                tracks_tc = []

                for t_idx in sorted(sgm.node_map.keys()):
                    c_arr = sgm.node_map[t_idx]
                    if len(c_arr) > 0:
                        tracks_tc.append((t_idx, c_arr[0]))  # eventually parts can be merged

                if len(tracks_tc):
                    tracks_f_tcs.append((sgm.flow_slv, tracks_tc))

    # ~dummy segments, connecting segments, or rather
    for gr_idx, vtx_x_grp in enumerate(vtx_xg):
        for vtx_x in vtx_x_grp:
            for vtx in vtx_x:
                t_idx, c_idx = vtx.t_idx, vtx.c_idx
                tracks_tc_tmpl = [(t_idx, c_idx), (t_idx, c_idx)]
                for lr, links in zip([0, 1], [vtx.l_links, vtx.r_links]):
                    for lnk in links:
                        if lnk.selected:
                            assert (lnk.m > 0)
                            tracks_tc = tracks_tc_tmpl.copy()
                            tgt_tc = (lnk.tgt_t_idx, lnk.tgt_c_idx)
                            tracks_tc[lr] = tgt_tc
                            tracks_f_tcs.append((0, tracks_tc))  # we use 0 as flow to show connecting segments

    for flow, tc_idxs in tracks_f_tcs:
        n = len(tc_idxs)
        assert (n > 0)
        if n == 1:
            continue

        uidx += 1

        xyztid = []
        aux_local = []

        for t_idx, c_idx in tc_idxs:
            cell = stack.st[t_idx].cells[c_idx]
            xyztid.append([cell.x, cell.y, cell.z, t_idx])
            aux_local.append([cell.w])

        all_t = [v[3] for v in xyztid]
        dt = all_t[-1] - all_t[0]
        n_nodes = len(xyztid)

        aux = [uidx, flow, n_nodes, dt, ]

        tracks_xyz_t_id_uidx.append(xyztid)
        tracks_aux_pars.append(aux)
        tracks_aux_local_pars.append(aux_local)

    tracks_xyz_map = {0: 'x', 1: 'y', 2: 'z', 3: 't_idx'}

    tracks_aux_map = {0: 'uidx', 1: 'flow', 2: 'n_nodes', 3: 'dt'}
    tracks_aux_local_map = {0: 'w'}

    dataset = {
        'tracks_xyz': tracks_xyz_t_id_uidx,
        'tracks_aux': tracks_aux_pars,
        'tracks_aux_local': tracks_aux_local_pars,
        'xyz_map': tracks_xyz_map,
        'aux_map': tracks_aux_map,
        'aux_local_map': tracks_aux_local_map
    }
    with open(fname, 'wb') as f:
        pickle.dump(dataset, f)


class SegmentExtender:
    """
    Group of methous and the interface funtion for extending outer segments (1-segments) into crossings
    by using subcells, if best subcell-to-segment assignment is unambigous.
    All crossings are preserved and no segments are merged.

    In the process all input variables remain unaffected apart form stack, where new subcell grous can be generated.
    Use constructor and extend_all_outer_segments_iter interfaces.

    """

    def __init__(self, stack, vtx_xg, sgm_xg, sgm_1g):
        """
        Args:
            stack(Stack): stack of timeframes containing full other_cell & all their subcell data
            vtx_xg(list(list(list(Vertex)))): vertices per group per xing
            sgm_xg(list(list(Segment))): crossing-segments, per group
            sgm_1g(list(list(Segment))): outer (1-segments), per group


        """
        self.stack = stack

        self.vtx_xg = copy.deepcopy(vtx_xg)
        self.sgm_xg = copy.deepcopy(sgm_xg)
        self.sgm_1g = copy.deepcopy(sgm_1g)

        self.vtx_tc_to_gx = {}

        self.gx_to_sgmxg_idx = {}
        self.vtx_tc_to_sgmxg_idx = {}

        self.gx_to_sgm1g_idx = {}
        self.vtx_tc_to_sgm1g_idx = {}

        self.gs_to_tr = {}
        self.gs_to_mid_pts = {}
        self.gs_to_hs2 = {}
        self.n_fit_nodes = 6

        self.init_intermediate_data()

    def init_intermediate_data(self):
        self.init_vtx_to_gx()
        self.init_gx_to_sgmx()
        self.init_gx_to_sgm1()

        self.init_sgm_chi2_mtr()

    def init_vtx_to_gx(self):
        """
            makes vtx_tc->group,xing indeces
        """
        for grp_idx, vtx_g in enumerate(self.vtx_xg):
            for xing_idx, vtx_x in enumerate(vtx_g):
                for vtx in vtx_x:
                    tc_idx = vtx.t_idx, vtx.c_idx
                    self.vtx_tc_to_gx[tc_idx] = (grp_idx, xing_idx)

    def init_gx_to_sgmx(self):
        """
        makes:
            gx_to_sgmxg_idx: group,xing index -> [[l sgm_xg], [r sgm_xg]] indeces
                (# l is joining xing/vtx, r is leaving)

            vtx_tc_to_sgmxg_idx: tc_idx->x sgm index in group idx
        """
        tc_nc = (-1, -1)
        for grp_idx, sgm_g in enumerate(self.sgm_xg):
            for sgm_idx, sgm in enumerate(sgm_g):
                l_nodes = {tc_l for tc_l, tc_l_own in sgm.l_links}
                r_nodes = {tc_r for tc_r_own, tc_r in sgm.r_links}

                assert (len(l_nodes) == 1)
                assert (len(r_nodes) == 1)

                l_node = l_nodes.pop()
                r_node = r_nodes.pop()

                if l_node == tc_nc and r_node == tc_nc:
                    # print(sgm.flow_slv, len(sgm.nodes))
                    continue

                l_xing_idx = -1
                r_xing_idx = -1
                if r_node != tc_nc:
                    gx = r_grp_idx, r_xing_idx = self.vtx_tc_to_gx[r_node]
                    assert (r_grp_idx == grp_idx)
                    if gx not in self.gx_to_sgmxg_idx:
                        self.gx_to_sgmxg_idx[gx] = [[], []]
                    self.gx_to_sgmxg_idx[gx][0].append(sgm_idx)

                    if r_node not in self.vtx_tc_to_sgmxg_idx:
                        self.vtx_tc_to_sgmxg_idx[r_node] = [[], []]
                    self.vtx_tc_to_sgmxg_idx[r_node][0].append(sgm_idx)

                if l_node != tc_nc:
                    gx = l_grp_idx, l_xing_idx = self.vtx_tc_to_gx[l_node]
                    assert (l_grp_idx == grp_idx)
                    if gx not in self.gx_to_sgmxg_idx:
                        self.gx_to_sgmxg_idx[gx] = [[], []]
                    self.gx_to_sgmxg_idx[gx][1].append(sgm_idx)

                    if l_node not in self.vtx_tc_to_sgmxg_idx:
                        self.vtx_tc_to_sgmxg_idx[l_node] = [[], []]
                    self.vtx_tc_to_sgmxg_idx[l_node][1].append(sgm_idx)

                if l_xing_idx != -1 and r_xing_idx != -1:
                    assert (l_xing_idx == r_xing_idx)

    def init_gx_to_sgm1(self):
        """
        makes:
            gx_to_sgm1g_idx: group,xing-> [[l sgm_1g], [r sgm_1g]] indeces
                (# l is joining xing/vtx, r is leaving)

            vtx_tc_to_sgm1g_idx: tc_idx->1 sgm index in group idx
        """
        tc_nc = (-1, -1)
        for grp_idx, sgm_g in enumerate(self.sgm_1g):
            for sgm_idx, sgm in enumerate(sgm_g):
                l_nodes = {tc_l for tc_l, tc_l_own in sgm.l_links}
                r_nodes = {tc_r for tc_r_own, tc_r in sgm.r_links}

                assert (len(l_nodes) == 1)
                assert (len(r_nodes) == 1)

                l_node = l_nodes.pop()
                r_node = r_nodes.pop()

                l_xing_idx = -1
                r_xing_idx = -1
                if r_node != tc_nc:
                    gx = r_grp_idx, r_xing_idx = self.vtx_tc_to_gx[r_node]
                    assert (r_grp_idx == grp_idx)
                    if gx not in self.gx_to_sgm1g_idx:
                        self.gx_to_sgm1g_idx[gx] = [[], []]
                    self.gx_to_sgm1g_idx[gx][0].append(sgm_idx)

                    if r_node not in self.vtx_tc_to_sgm1g_idx:
                        self.vtx_tc_to_sgm1g_idx[r_node] = [[], []]
                    self.vtx_tc_to_sgm1g_idx[r_node][0].append(sgm_idx)

                if l_node != tc_nc:
                    gx = l_grp_idx, l_xing_idx = self.vtx_tc_to_gx[l_node]
                    assert (l_grp_idx == grp_idx)
                    if gx not in self.gx_to_sgm1g_idx:
                        self.gx_to_sgm1g_idx[gx] = [[], []]
                    self.gx_to_sgm1g_idx[gx][1].append(sgm_idx)

                    if l_node not in self.vtx_tc_to_sgm1g_idx:
                        self.vtx_tc_to_sgm1g_idx[l_node] = [[], []]
                    self.vtx_tc_to_sgm1g_idx[l_node][1].append(sgm_idx)

    # inner methods:

    def get_vtx_1_flow_info(self, vtx_tc):
        """
        Helper function to obtain number of associated 1-segments and their sum of flow
        Args:
            vtx_tc (tuple(t_idx, c_idx))

        Returns:
            tuple(flow_l_1, flow_r_1, c_l, c_r): flow left and right of vtx & number of connections, i.e. segments)
        """

        if vtx_tc not in self.vtx_tc_to_sgm1g_idx:
            return 0, 0, 0, 0

        grp_idx, xing_idx = self.vtx_tc_to_gx[vtx_tc]
        l_sgm_idxs, r_sgm_idxs = self.vtx_tc_to_sgm1g_idx[vtx_tc]

        c_l, c_r = len(l_sgm_idxs), len(r_sgm_idxs)

        flows_l = [self.sgm_1g[grp_idx][idx].flow_slv for idx in l_sgm_idxs]
        flows_r = [self.sgm_1g[grp_idx][idx].flow_slv for idx in r_sgm_idxs]
        f_l, f_r = np.sum(flows_l), np.sum(flows_r)
        return f_l, f_r, c_l, c_r

    def get_vtx_x_flow_info(self, vtx_tc):
        """
        Helper function to obtain number of associated x-segments and their sum of flow
        Args:
            vtx_tc (tuple(t_idx, c_idx))

        Returns:
            tuple(flow_l_x, flow_r_x, c_l, c_r): flow left and right of vtx & number of connections, i.e. segments)
        """

        if vtx_tc not in self.vtx_tc_to_sgmxg_idx:
            return 0, 0, 0, 0

        grp_idx, xing_idx = self.vtx_tc_to_gx[vtx_tc]
        l_sgm_idxs, r_sgm_idxs = self.vtx_tc_to_sgmxg_idx[vtx_tc]

        c_l, c_r = len(l_sgm_idxs), len(r_sgm_idxs)

        flows_l = [self.sgm_xg[grp_idx][idx].flow_slv for idx in l_sgm_idxs]
        flows_r = [self.sgm_xg[grp_idx][idx].flow_slv for idx in r_sgm_idxs]
        f_l, f_r = np.sum(flows_l), np.sum(flows_r)
        return f_l, f_r, c_l, c_r

    def get_num_vtx_subnodes(self, tc_idx):
        t_idx, c_idx = tc_idx
        cell: Cell = self.stack.st[t_idx].cells[c_idx]
        n_sub_nodes = max(1, len(cell.get_subcells_idx()))
        return n_sub_nodes

    def get_vtx_subnodes(self, tc_idx):
        t_idx, c_idx = tc_idx
        cell: Cell = self.stack.st[t_idx].cells[c_idx]
        sc_idxs = cell.get_subcells_idx()

        return [(t_idx, sc_idx) for sc_idx in sc_idxs]

    def cell_coords(self, t_idx, c_idx):
        return self.stack.st[t_idx].cells[c_idx].get_r()

    # linfit of track
    @staticmethod
    def get_linfit_pars(ts, rs):
        # print(len(ts))
        p1s, p0s = np.polyfit(ts, rs, 1)
        return p1s, p0s

    @staticmethod
    def get_dr2(p1s, p0s, t, r):
        dr = p1s * t + p0s - r
        # print(p1s)
        return (dr ** 2).sum()

    @staticmethod
    def get_dr2_for_subsegment(ts, rs, t, r):
        p1s, p0s = SegmentExtender.get_linfit_pars(ts, rs)
        dr2 = SegmentExtender.get_dr2(p1s, p0s, t, r)
        # print(dr2)
        return dr2

    @staticmethod
    def get_endpoint_dr2_for_subsegment(ts, rs):  # pred for first & last
        ts_l = ts[:-1]
        ts_r = ts[1:]
        rs_l = rs[:-1]
        rs_r = rs[1:]

        t_l = ts[0]
        t_r = ts[-1]
        r_l = rs[0]
        r_r = rs[-1]

        dr2_lr = SegmentExtender.get_dr2_for_subsegment(ts_l, rs_l, t_r, r_r)
        dr2_rl = SegmentExtender.get_dr2_for_subsegment(ts_r, rs_r, t_l, r_l)

        return [dr2_lr, dr2_rl]

    @staticmethod
    def get_sgm_middle_jump_points(sgm):
        middle_points = []
        for frm, to in sgm.mj_t_ranges:
            for t in range(frm, to):
                middle_points.append(t + 0.5)
        return middle_points

    @staticmethod
    def middle_point_on_subsegment(ts, mid_pts):
        if len(mid_pts) == 0:
            return False

        t0, t1 = ts[0], ts[-1]
        mid_pnt_on = (t0 < mp < t1 for mp in mid_pts)
        return any(mid_pnt_on)

    @staticmethod
    def get_all_endpoint_dr2_for_segment(ts, rs, mid_pts, n_fit_nodes):
        min_len = n_fit_nodes + 1
        dr2s = []

        while True:  # try removing jump regions if possible
            for ofs in range(len(rs) - n_fit_nodes):
                ts_i = ts[ofs: ofs + min_len]
                rs_i = rs[ofs: ofs + min_len]
                if SegmentExtender.middle_point_on_subsegment(ts_i, mid_pts):
                    continue

                dr2s_i = SegmentExtender.get_endpoint_dr2_for_subsegment(ts_i, rs_i)
                dr2s.extend(dr2s_i)
            if len(dr2s) == 0 and len(mid_pts) > 0:
                # print(ts, mid_pts)
                mid_pts = []
            else:
                break
        # print(dr2s)
        return dr2s

    @staticmethod
    def get_s2_for_segment(ts, rs, mid_pts, n_fit_nodes):
        dr2s = SegmentExtender.get_all_endpoint_dr2_for_segment(ts, rs, mid_pts, n_fit_nodes)
        # print(dr2s)
        # return np.mean(dr2s)
        return np.sqrt(dr2s).mean() ** 2

    def get_tsrs_for_segment(self, sgm):
        ts = []
        rs = []
        for t_idx, c_idxs in sgm.node_map.items():
            ts.append(t_idx)
            r = [self.cell_coords(t_idx, c_idx) for c_idx in c_idxs]
            r_mean = np.mean(r, axis=0)
            rs.append(r_mean)
        rs = np.asarray(rs)
        return ts, rs

    def get_dr2_segment_to_node(self, sgm, cell_tc_idx):
        min_len = self.n_fit_nodes + 1

        t_idx, c_idx = cell_tc_idx
        t = t_idx
        r = self.cell_coords(t_idx, c_idx)

        ts, rs = self.get_tsrs_for_segment(sgm)
        t0, t1 = ts[0], ts[-1]

        if abs(t0 - t) < abs(t1 - t):
            ts_i = ts[: min_len]
            rs_i = rs[: min_len]
        else:
            ts_i = ts[-min_len:]
            rs_i = rs[-min_len:]

        p1s, p0s = SegmentExtender.get_linfit_pars(ts_i, rs_i)
        dr2 = SegmentExtender.get_dr2(p1s, p0s, t, r)
        return dr2

    @staticmethod
    def sgm_hash(sgm):
        return hash(tuple(sorted(sgm.nodes)))

    def get_or_update_sgm_s2(self, gs_idx):
        sgm = self.sgm_1g[gs_idx[0]][gs_idx[1]]

        h, s2 = self.gs_to_hs2.get(gs_idx, (None, 0))

        curr_h = SegmentExtender.sgm_hash(sgm)
        if curr_h != h:
            ts, rs = self.get_tsrs_for_segment(sgm)

            self.gs_to_tr[gs_idx] = (ts, rs)
            mid_pts = self.gs_to_mid_pts[gs_idx]
            s2 = SegmentExtender.get_s2_for_segment(ts, rs, mid_pts, self.n_fit_nodes)
            self.gs_to_hs2[gs_idx] = (curr_h, s2)

        return s2

    def get_chi2_segment_to_node(self, gs_idx, cell_tc_idx):
        s2 = self.get_or_update_sgm_s2(gs_idx)
        sgm = self.sgm_1g[gs_idx[0]][gs_idx[1]]
        dr2 = self.get_dr2_segment_to_node(sgm, cell_tc_idx)
        return dr2 / s2

    @staticmethod
    def get_unique_solution(chi2_mtr):
        # returns column idx for each row or None if no uniques (best for each) solution available
        sol = []
        for row_idx, row in enumerate(chi2_mtr):
            best_col_idx = np.argmin(row)
            col = chi2_mtr[:, best_col_idx]
            if np.argmin(col) == row_idx:
                sol.append(best_col_idx)
            else:
                return None
        return sol

    def init_sgm_chi2_mtr(self):
        # 1. get seq of coords & t for segment

        sgm_lens = []
        for grp_idx, sgm_g in enumerate(self.sgm_1g):
            for sgm_idx, sgm in enumerate(sgm_g):
                ts, rs = self.get_tsrs_for_segment(sgm)
                gs_idx = (grp_idx, sgm_idx)
                self.gs_to_mid_pts[gs_idx] = SegmentExtender.get_sgm_middle_jump_points(sgm)
                sgm_lens.append(len(sgm.node_map))

        # 2. get sigma^2 for each segment
        min_len = min(sgm_lens) if len(sgm_lens) else 1
        self.n_fit_nodes = min_len - 1

        # init all
        for gs_idx in self.gs_to_mid_pts.keys():
            s2 = self.get_or_update_sgm_s2(gs_idx)

    # Remove vtx from vtx container
    def remove_vtx(self, res_dict):
        # since idx is never to the best of present knowledge used
        # we could just remove it:

        grp_idx, xing_idx, vtx_idx, _ = res_dict['vtx']
        removed_vtx = self.vtx_xg[grp_idx][xing_idx][vtx_idx]
        del self.vtx_xg[grp_idx][xing_idx][vtx_idx]

        return removed_vtx

    def extend_1_segments(self, res_dict):
        """Extends 1-segments according to best cells"""

        # get segment objects
        grp_idx, *_ = res_dict['vtx']
        subcell_asgn = res_dict['subcell_asgn']

        g_sgm = self.sgm_1g[grp_idx]
        for sgm_idx, subcell_tc in subcell_asgn.items():
            sgm = g_sgm[sgm_idx]

            # append other_cell.is added on the correct side
            # WARNING: sgm links are not yet updated!
            sgm.add_node(subcell_tc)

    def remove_next_node_from_x_segments(self, res_dict):
        """"""
        grp_idx, xing_idx, vtx_idx, (t_idx, c_idx) = res_dict['vtx']

        all_x_sgm_idx = res_dict['x_sgm_idx']
        xg_sgm = self.sgm_xg[grp_idx]

        x_sgm_on_the_left_from_vtx = res_dict['1-lr']
        remove_node_on_the_right_of_sgm = x_sgm_on_the_left_from_vtx
        t_idx_remove = t_idx

        while True:
            t_idx_remove = t_idx_remove + ((-1) if remove_node_on_the_right_of_sgm else (+1))

            removed_nodes_tc = []
            for sgm_idx in all_x_sgm_idx:
                sgm = xg_sgm[sgm_idx]
                removed_tc = sgm.remove_nodes_at(t_idx_remove)
                removed_nodes_tc += removed_tc

            if len(removed_nodes_tc):
                break

        return removed_nodes_tc

    def make_m_cell_from_subcells(self, t_idx, subcell_c_idxs: list):
        """
        Creates an m-other_cell from several sub-cells according to subcell_tc_idxs
        Returns:
            tc_idx: tuple: tc of the newly added m-other_cell in the stack
        """

        stack_cells = self.stack.st[t_idx].cells
        new_c_idx = len(stack_cells)
        sub_cells = [stack_cells[c_idx] for c_idx in subcell_c_idxs]
        for sc in sub_cells:
            sc.m_idx = [new_c_idx]

        cell = Cell.cell_from_subcells(sub_cells, subcell_c_idxs, new_c_idx)
        stack_cells.append(cell)

        return (t_idx, new_c_idx)

    def make_m_cell_from_parts(self, tc_idxs: list):
        """
        Creates an m-other_cell from several single-, sub-, or m-cells according to tc_idxs
        If c_idxs contains not all sub-cells on an existing m-other_cell Value error is raised
        if m-other_cell is created form several existing m-cells, existing ones are NOT destroyed, new is appended to the list,
        with sub-cells ids pointing to subcells of original m-cells. YET, subcell_idx (showing the m-other_cell idx for subcells)
        is reset to the new m-other_cell
        The caller is responsible for verifying that none if their code expects to reuse the engulfed m-cells

        Returns:
            tc_idx: tuple: tc of the newly added or previously existing other_cell in the stack
        """

        all_t_idxs = {t_idx for t_idx, c_idx in tc_idxs}
        all_c_idxs = [c_idx for t_idx, c_idx in tc_idxs]
        # there should be nodes, and all at same timeframe, otherwise we did smth wrong previously collecting them
        assert len(all_t_idxs) == 1
        assert len(all_c_idxs) == len(set(all_c_idxs))

        t_idx = all_t_idxs.pop()
        st_cells = self.stack.st[t_idx].cells

        subcells_for_merging = []
        all_subcells_of_involved_m_cells = []
        independent_top_level_c_idx = []

        # 1. get all subcells:
        for c_idx in all_c_idxs:
            cell: Cell = st_cells[c_idx]
            if cell.is_subcell():
                subcells_for_merging.append(c_idx)

                subcell_idxs = cell.get_subcells_idx()
                assert len(subcell_idxs) == 1

                # collect all orig subcells, to ensure all collected
                m_cell_idx = subcell_idxs[0]
                independent_top_level_c_idx.append(m_cell_idx)
                m_cell: Cell = st_cells[m_cell_idx]
                subcell_idxs = m_cell.get_subcells_idx()
                all_subcells_of_involved_m_cells.extend(subcell_idxs)

            elif cell.is_single():
                subcells_for_merging.append(c_idx)
                independent_top_level_c_idx.append(c_idx)

            elif cell.is_merged():
                subcell_idxs = cell.get_subcells_idx()
                subcells_for_merging.extend(subcell_idxs)
                independent_top_level_c_idx.append(c_idx)

        subcells_for_merging = set(subcells_for_merging)
        all_subcells_of_involved_m_cells = set(all_subcells_of_involved_m_cells)
        independent_top_level_c_idx = set(independent_top_level_c_idx)

        # assert all subcells of an M-other_cell are collected. Mb this is not required?...
        # if subcells_for_merging.union(all_subcells_of_involved_m_cells) != subcells_for_merging:
        #     raise ValueError('some subcells of original M-other_cell are not in the collection for new M-other_cell')

        if (len(independent_top_level_c_idx) == 1
                # it's only one M-other_cell or a single other_cell. nothing to merge

                and

                subcells_for_merging.union(all_subcells_of_involved_m_cells) == subcells_for_merging
                # All subcells of original M-other_cell are  in the collection for new M-other_cell')
        ):
            return (t_idx, independent_top_level_c_idx.pop())
        else:  # need merging:
            if subcells_for_merging.union(all_subcells_of_involved_m_cells) != subcells_for_merging:
                print('some subcells of original M-other_cell are not in the collection for new M-other_cell')
            return self.make_m_cell_from_subcells(t_idx, subcells_for_merging)

    def add_vtx(self, res_dict, tc_idx):
        """
        Returns:
            idx(int): index in the collection of the added Vtx.
        """

        g, x, *_ = res_dict['vtx']
        lc = LinkedCell(*tc_idx)
        lc.fill_struct()

        vtx = Vertex(lc)
        vtx_x = self.vtx_xg[g][x]
        idx = len(vtx_x)
        vtx_x.append(vtx)

        return idx

    def relink_vtx_sgm(self, res_dict, new_vtx_idx, new_tc_idx, removed_vtx):
        # +update vtx, 1-sgm-s, and x-sgm-s links for all construct.
        # for segments - on the proper side, also update flow map(timerange)
        # and jump map (move to corresponding 1-segment from x segment?)

        x_sgm_on_the_left_from_vtx = res_dict['1-lr']
        g, x, old_vtx_idx, old_tc_idx = res_dict['vtx']

        all_1_sgm_idx = res_dict['1_sgm_idx']
        all_x_sgm_idx = res_dict['x_sgm_idx']

        new_t_idx, new_c_idx = new_tc_idx

        # 1. unlink old vtx
        assert (removed_vtx.l_nc_f_slv == removed_vtx.r_nc_f_slv == 0)

        # 2. update local association containers
        del self.vtx_tc_to_gx[old_tc_idx]
        self.vtx_tc_to_gx[new_tc_idx] = (g, x)  # order matters as c_idx could have been reused

        sgmxg_idx = self.vtx_tc_to_sgmxg_idx[old_tc_idx]
        self.vtx_tc_to_sgmxg_idx[new_tc_idx] = sgmxg_idx
        del self.vtx_tc_to_sgmxg_idx[old_tc_idx]

        sgm1g_idx = self.vtx_tc_to_sgm1g_idx[old_tc_idx]
        self.vtx_tc_to_sgm1g_idx[new_tc_idx] = sgm1g_idx
        del self.vtx_tc_to_sgm1g_idx[old_tc_idx]

        vtx_l_tgts = []
        vtx_r_tgts = []

        # 3. relink 1 - segments
        change_left = x_sgm_on_the_left_from_vtx
        grp_sgm = self.sgm_1g[g]
        for sgm_idx in all_1_sgm_idx:
            sgm = grp_sgm[sgm_idx]

            t_arr = np.array(list(sgm.node_map.keys()))
            t_l, t_r = t_arr.min(), t_arr.max()

            if change_left:
                t = t_l

                nodes = sgm.node_map[t]
                nodes_tc_idxs = [(t, node_c_idx) for node_c_idx in nodes]
                sgm.l_links = [(new_tc_idx, node_tc_idx) for node_tc_idx in nodes_tc_idxs]
                vtx_r_tgts.extend(nodes_tc_idxs)
            else:
                t = t_r

                nodes = sgm.node_map[t]
                nodes_tc_idxs = [(t, node_c_idx) for node_c_idx in nodes]
                sgm.r_links = [(node_tc_idx, new_tc_idx) for node_tc_idx in nodes_tc_idxs]
                vtx_l_tgts.extend(nodes_tc_idxs)

        # 4. relink x - segments
        change_left = not x_sgm_on_the_left_from_vtx
        grp_sgm = self.sgm_xg[g]
        for sgm_idx in all_x_sgm_idx:
            sgm = grp_sgm[sgm_idx]
            t_arr = np.array(list(sgm.node_map.keys()))
            if len(t_arr):
                t_l, t_r = t_arr.min(), t_arr.max()

                if change_left:
                    t = t_l

                    nodes = sgm.node_map[t]
                    nodes_tc_idxs = [(t, node_c_idx) for node_c_idx in nodes]
                    sgm.l_links = [(new_tc_idx, node_tc_idx) for node_tc_idx in nodes_tc_idxs]
                    vtx_r_tgts.extend(nodes_tc_idxs)
                else:
                    t = t_r

                    nodes = sgm.node_map[t]
                    nodes_tc_idxs = [(t, node_c_idx) for node_c_idx in nodes]
                    sgm.r_links = [(node_tc_idx, new_tc_idx) for node_tc_idx in nodes_tc_idxs]
                    vtx_l_tgts.extend(nodes_tc_idxs)
            else:  # no nodes left, relink to other side
                if change_left:
                    # right side valid
                    l_tc = new_tc_idx
                    r_tc = sgm.r_links[0][1]

                    sgm.l_links = [(l_tc, r_tc)]
                    sgm.r_links = [(l_tc, r_tc)]
                    vtx_r_tgts.extend([r_tc])
                else:
                    # left side valid
                    l_tc = sgm.l_links[0][0]
                    r_tc = new_tc_idx

                    sgm.l_links = [(l_tc, r_tc)]
                    sgm.r_links = [(l_tc, r_tc)]
                    vtx_l_tgts.extend([l_tc])

        # 5. fill connections for new vtx
        vtx = self.vtx_xg[g][x][new_vtx_idx]
        vtx.l_links = [LinkedCell.Link(t_idx, c_idx, 0, abs(new_t_idx - t_idx)) for t_idx, c_idx in vtx_l_tgts]
        vtx.r_links = [LinkedCell.Link(t_idx, c_idx, 0, abs(new_t_idx - t_idx)) for t_idx, c_idx in vtx_r_tgts]

        # ToDo: ^ here mb smth missing, needs check
        pass

    def extend_outer_segments(self, res_dict):
        removed_vtx = self.remove_vtx(res_dict)
        self.extend_1_segments(res_dict)
        removed_nodes_tc = self.remove_next_node_from_x_segments(res_dict)

        new_vtx_cell_tc = self.make_m_cell_from_parts(removed_nodes_tc)
        vtx_idx = self.add_vtx(res_dict, new_vtx_cell_tc)

        self.relink_vtx_sgm(res_dict, vtx_idx, new_vtx_cell_tc, removed_vtx)

    def extend_all_outer_segments(self):
        updated = False
        for grp_idx, vtx_g in enumerate(self.vtx_xg):
            for xing_idx, vtx_x in enumerate(vtx_g):
                for vtx_idx, vtx in enumerate(vtx_x):
                    tc_idx = t_idx, c_idx = vtx.t_idx, vtx.c_idx
                    if vtx.l_nc_f_slv == 0 and vtx.r_nc_f_slv == 0:
                        flow_l_1, flow_r_1, c1_l, c1_r = self.get_vtx_1_flow_info(tc_idx)
                        flow_l_x, flow_r_x, cx_l, cx_r = self.get_vtx_x_flow_info(tc_idx)
                        l1 = c1_r == 0 and cx_l == 0
                        r1 = c1_l == 0 and cx_r == 0
                        if l1 or r1:
                            assert (c1_l + c1_r > 0)
                            flow_1 = flow_l_1 + flow_r_1
                            n_vtx_sn = self.get_num_vtx_subnodes(tc_idx)
                            if flow_1 == n_vtx_sn:
                                all_x_sgm_idx = self.vtx_tc_to_sgmxg_idx[tc_idx]
                                all_x_sgm_idx = all_x_sgm_idx[0] + all_x_sgm_idx[1]
                                all_x_sgm = [self.sgm_xg[grp_idx][sgm_idx] for sgm_idx in all_x_sgm_idx]
                                all_x_seg_non0_len = all([len(sgm.node_map) > 0 for sgm in all_x_sgm])
                                if not all_x_seg_non0_len:
                                    continue

                                # print(f'{i}:\t{flow_1}', flow_l_1, flow_r_1, c1_l, c1_r, flow_l_x, flow_r_x, cx_l, cx_r)
                                # i += 1

                                # get all 1-segments:
                                all_1_sgm_idx = self.vtx_tc_to_sgm1g_idx[tc_idx]
                                all_1_sgm_idx = all_1_sgm_idx[0] + all_1_sgm_idx[1]

                                all_1_sgm = [self.sgm_1g[grp_idx][sgm_idx] for sgm_idx in all_1_sgm_idx]
                                # 1-segments can't have zero nodes.

                                all_sn_tc = self.get_vtx_subnodes(tc_idx)
                                # print(all_1_sgm, all_sn_tc)
                                chi2mtx = [[self.get_chi2_segment_to_node((grp_idx, sgm_idx),
                                                                          sn_tc_idx)
                                            for sn_tc_idx in all_sn_tc]
                                           for sgm_idx in all_1_sgm_idx
                                           ]

                                chi2mtx = np.array(chi2mtx)
                                # print(chi2mtx)
                                sol = SegmentExtender.get_unique_solution(chi2mtx)

                                if sol is not None:
                                    # we can extend segment
                                    # for sgm_idx, sol_tgt in zip(all_1_sgm_idx, sol):
                                    #    scell_tc_idx = all_sn_tc[sol_tgt]
                                    #    print(f'{sgm_idx}->{scell_tc_idx}, {["l","r"][r1]}')

                                    element = {
                                        'vtx': [grp_idx, xing_idx, vtx_idx, tc_idx],
                                        '1_sgm_idx': all_1_sgm_idx,
                                        'x_sgm_idx': all_x_sgm_idx,
                                        '1-lr': r1,  # x-rl,
                                        'subcell_asgn': {sgm_idx: all_sn_tc[sol_tgt] for sgm_idx, sol_tgt in
                                                         zip(all_1_sgm_idx, sol)}
                                    }
                                    self.extend_outer_segments(res_dict=element)
                                    updated = True
        return updated

    def extend_all_outer_segments_iter(self):
        while True:
            updated = self.extend_all_outer_segments()
            if not updated:
                break
        return self.vtx_xg, self.sgm_xg, self.sgm_1g


def get_used_tc(vtx_xg, sgm_xg, sgm_1g):
    used_tc = set()

    for grp_idx, vtx_g in enumerate(vtx_xg):
        for xing_idx, vtx_x in enumerate(vtx_g):
            for vtx in vtx_x:
                tc_idx = vtx.t_idx, vtx.c_idx
                used_tc.add(tc_idx)

    for sgm_ig in [sgm_xg, sgm_1g]:
        for grp_idx, sgm_g in enumerate(sgm_ig):
            for sgm_idx, sgm in enumerate(sgm_g):
                used_tc.update(set(sgm.nodes))

    return used_tc


def get_tc_subtypes(stack, used_tc):
    """
    Returns tc_idxs separately for single nodes, merged nodes (+ tc_idx all their associated subcells),
    and nodes which are subcells of merged nodes (+ tc_idx of corresponding merged nodes)
    """
    # get set of used co-nodes: M if any part used, all parts if M used

    used_tc_S = set()  # singles

    used_tc_M = set()  # used M
    used_tc_M_co_SC = set()  # subcells of used M

    used_tc_SC = set()  # used subcells
    used_tc_SC_co_M = set()  # M other_cell of used subcelss

    for tc_idx in used_tc:
        t_idx, c_idx = tc_idx
        st = stack.st[t_idx]
        cell = st.cells[c_idx]

        if cell.is_merged():
            used_tc_M.add(tc_idx)
            subcells_idx = cell.get_subcells_idx()
            for sc_c_idx in subcells_idx:
                sc_tc = (t_idx, sc_c_idx)
                used_tc_M_co_SC.add(sc_tc)
        elif cell.is_subcell():
            used_tc_SC.add(tc_idx)
            subcells_idx = cell.get_subcells_idx()
            assert (len(subcells_idx) == 1)
            mc_c_idx = subcells_idx[0]
            sc_tc = (t_idx, mc_c_idx)
            used_tc_SC_co_M.add(sc_tc)
        elif cell.is_single():
            used_tc_S.add(tc_idx)
        else:
            print('shouldnt be here')

    return used_tc_S, used_tc_M, used_tc_M_co_SC, used_tc_SC, used_tc_SC_co_M


def get_all_tc(stack):
    all_tc = set()
    for t_idx, st in stack.st.items():
        for c_idx, _ in enumerate(st.cells):
            tc_idx = (t_idx, c_idx)
            all_tc.add(tc_idx)
    return all_tc


def get_cell_DoC(stack, tc_idx):
    t_idx, c_idx = tc_idx
    cell = stack.st[t_idx].cells[c_idx]
    return cell.get_DoC()


def get_cells_DoC(stack, tc_idxs):
    return [get_cell_DoC(stack, tc_idx) for tc_idx in tc_idxs]


def explore_unused_cells_draft(vtx_xg, sgm_xg, sgm_1g, st_full):
    # ### Get unused cells
    used_tc = get_used_tc(vtx_xg, sgm_xg, sgm_1g)
    used_tc_S, used_tc_M, used_tc_M_co_SC, used_tc_SC, used_tc_SC_co_M = get_tc_subtypes(st_full, used_tc)

    all_tc = get_all_tc(st_full)
    unused_tc = all_tc.difference(used_tc, used_tc_SC_co_M)  # exclude also subcells of used M nodes

    unused_cells_times = [t_idx for t_idx, _ in unused_tc]

    # # distribution of non-used cells: one can see accumulation phase
    # plt.figure(figsize=(20, 4))
    # plt.hist(unused_cells_times, np.max(unused_cells_times)-np.min(unused_cells_times)+1);


    # filter after accumulation phase:
    # ToDo: one can also filter of somewhere is > c * median in neighbouhood of 5tf or smth
    unused_tc = {tc_idx for tc_idx in unused_tc if tc_idx[0] > cfgm.T_ACCUMULATION_END}

    # distribution of non-used selected cells
    # plt.figure(figsize=(20, 4))
    unused_cells_times = [t_idx for t_idx, _ in unused_tc]
    # plt.hist(unused_cells_times, np.max(unused_cells_times)-np.min(unused_cells_times)+1);


    unused_tc_S, unused_tc_M, unused_tc_M_co_SC, unused_tc_SC, unused_tc_SC_co_M = get_tc_subtypes(st_full, unused_tc)

    # ### Study unused nodes


    # plot DoC
    for used_tcs in [used_tc_S, used_tc_M, used_tc_SC, unused_tc_S, unused_tc_M, unused_tc_SC]:
        docs = get_cells_DoC(st_full, used_tcs)
        # plt.hist(docs, log=1);
        # plt.show()

    test_tc = (149, 222)

    # no nodes in same vtx
    prev_tc_vtx = set()
    for gr_idx, vtx_x_grp in enumerate(vtx_xg):
        for vtx_x in vtx_x_grp:
            curr_tc = {(vtx.t_idx, vtx.c_idx) for vtx in vtx_x}

            if test_tc in curr_tc:
                print(gr_idx)

            i = curr_tc.intersection(prev_tc_vtx)
            if len(i) > 0:
                print(i, gr_idx)

            prev_tc_vtx.update(curr_tc)

    # no nodes in same vtx
    prev_tc_sgm_x = set()
    for gr_idx, sgm_grp in enumerate(sgm_xg):
        for sgm in sgm_grp:
            curr_tc = {tc_idx for tc_idx in sgm.nodes}
            curr_tc_m = {(t_idx, c_idx) for t_idx in sgm.node_map for c_idx in sgm.node_map[t_idx]}

            if (curr_tc != curr_tc_m):
                print(gr_idx, curr_tc, curr_tc_m)
            if test_tc in curr_tc:
                print(gr_idx)

            i_sx = curr_tc.intersection(prev_tc_sgm_x)
            i_v = curr_tc.intersection(prev_tc_vtx)
            if len(i_sx) > 0 or len(i_v) > 0:
                print(i_sx, i_v, gr_idx)

            prev_tc_sgm_x.update(curr_tc)

    # no nodes in same vtx
    prev_tc_sgm_1 = set()
    for gr_idx, sgm_grp in enumerate(sgm_1g):
        for sgm in sgm_grp:
            curr_tc = {tc_idx for tc_idx in sgm.nodes}
            curr_tc_m = {(t_idx, c_idx) for t_idx in sgm.node_map for c_idx in sgm.node_map[t_idx]}

            if (curr_tc != curr_tc_m):
                print(gr_idx, curr_tc, curr_tc_m)
            if test_tc in curr_tc:
                print(gr_idx)

            i_s1 = curr_tc.intersection(prev_tc_sgm_1)
            i_sx = curr_tc.intersection(prev_tc_sgm_x)
            i_v = curr_tc.intersection(prev_tc_vtx)
            if len(i_s1) > 0 or len(i_sx) > 0 or len(i_v) > 0:
                print(i, gr_idx)

            prev_tc_sgm_1.update(curr_tc)

# ### Merger classes
class FlowSolver:
    def __init__(self):
        self.starts = []
        self.ends = []
        self.all_connections = {}  # (id1, id2) -> (conn_id, t1,t2,chi2)
        self.resolved_conns = []  # (conn_id, tr_idx1, t1, tr_idx2, t2, chi2)

        self.all_ids = set()
        self.id_map = {}  # tr_idx->idx
        self.id_rmap = {}  # idx->tr_idx

        self._start_nodes = []
        self._end_nodes = []
        self._capacities = []
        self._costs = []

        self.solved_cost = 0

    def add_start(self, tr_idx):
        self.starts.append(tr_idx)

    def add_end(self, tr_idx):
        self.ends.append(tr_idx)

    def add_weight(self, conn_idx, tr_idx1, t1, tr_idx2, t2, chi2):
        if t2 < t1:
            t2, t1 = t1, t2
            tr_idx2, tr_idx1 = tr_idx1, tr_idx2

        self.all_connections[(tr_idx1, tr_idx2)] = (conn_idx, t1, t2, chi2)

    def solve(self):
        # ids:
        i1 = [id1_id2[0] for id1_id2 in self.all_connections.keys()]
        i2 = [id1_id2[1] for id1_id2 in self.all_connections.keys()]

        chis = [v[3] for v in self.all_connections.values()]
        chi_min, chi_max = min(chis), max(chis)
        d_chi = chi_max - chi_min

        self.all_ids = set(i1 + i2 + self.starts + self.ends)

        for i, tr_idx in enumerate(self.all_ids):
            idx = i + 2
            self.id_map[tr_idx] = idx
            self.id_rmap[idx] = tr_idx

        self._start_nodes = []
        self._end_nodes = []
        self._capacities = []
        self._costs = []

        def fill(idx1, idx2, w):
            self._start_nodes.append(idx1)
            self._end_nodes.append(idx2)
            self._capacities.append(1)
            self._costs.append(w)

        n_active = min(len(self.starts), len(self.ends))
        supplies = [n_active, -n_active] + [0] * n_active

        source = 0
        sink = 1

        # fill values
        for tr_idx in self.starts:
            idx = self.id_map[tr_idx]
            fill(0, idx, 0)

        for tr_idx in self.ends:
            idx = self.id_map[tr_idx]
            fill(idx, 1, 0)

        for tr_idx1_tr_idx2, conn_id_t1_t2_chi2 in self.all_connections.items():
            conn_id, t1, t2, chi2 = conn_id_t1_t2_chi2
            tr_idx1, tr_idx2 = tr_idx1_tr_idx2
            idx1, idx2 = self.id_map[tr_idx1], self.id_map[tr_idx2]

            chi2_i = (chi2 - chi_min) / d_chi * 100000
            chi2_i = int(chi2_i) + 1
            fill(idx1, idx2, chi2_i)

        min_cost_flow = pg.SimpleMinCostFlow()

        # print(self._start_nodes)
        # print(self._end_nodes)
        # print(self._capacities )
        # print(self._costs)
        # Add each arc.
        for i in range(len(self._start_nodes)):
            min_cost_flow.AddArcWithCapacityAndUnitCost(self._start_nodes[i], self._end_nodes[i],
                                                        self._capacities[i], self._costs[i])

        # Add node supplies.

        for i in range(len(supplies)):
            min_cost_flow.SetNodeSupply(i, supplies[i])

        res = min_cost_flow.Solve()
        res_codes = {min_cost_flow.BAD_COST_RANGE: 'BAD_COST_RANGE',
                     min_cost_flow.BAD_RESULT: 'BAD_RESULT',
                     min_cost_flow.FEASIBLE: 'FEASIBLE',
                     min_cost_flow.INFEASIBLE: 'INFEASIBLE',
                     min_cost_flow.NOT_SOLVED: 'NOT_SOLVED',
                     min_cost_flow.OPTIMAL: 'OPTIMAL',
                     min_cost_flow.UNBALANCED: 'UNBALANCED',
                     }
        if res == min_cost_flow.OPTIMAL:
            self.solved_cost = 0  # (min_cost_flow.OptimalCost() / 100000*d_chi) + chi_min

            for arc in range(min_cost_flow.NumArcs()):
                # Can ignore arcs leading out of source or into sink.
                if min_cost_flow.Tail(arc) != source and min_cost_flow.Head(arc) != sink:

                    # Arcs in the solution have a flow value of 1. Their start and end nodes
                    # give an assignment of worker to task.
                    if min_cost_flow.Flow(arc) > 0:
                        idx1 = min_cost_flow.Tail(arc)
                        idx2 = min_cost_flow.Head(arc)
                        chi2_i = min_cost_flow.UnitCost(arc)

                        tr_idx1 = self.id_rmap[idx1]
                        tr_idx2 = self.id_rmap[idx2]
                        conn_id_t1_t2_chi2 = self.all_connections[(tr_idx1, tr_idx2)]

                        conn_id, t1, t2, chi2 = conn_id_t1_t2_chi2
                        self.resolved_conns.append((conn_id, tr_idx1, t1, tr_idx2, t2, chi2))
                        self.solved_cost += chi2
            return True
        else:
            print('There was an issue with the min cost flow input.', res_codes[res])
            return False


class LAPSolver:
    chi2_no_conn = cfgm.W_NC_0

    def __init__(self, in_to_out=True, explicit_nc=False):
        """
        Args:
            in_to_out : if True outcoming tracks will be matched to incomming.
                        Otherwise tracks withing to the ones crossing the intesection
            explicit_nc: if True, the non-connected options must be given explicitly
        """
        self.starts = []
        self.ends = []
        self.in_to_out = in_to_out
        self.explicit_nc = explicit_nc

        self.all_connections = {}  # conn_id -> tr_idx -> [[t_in0,..], [t_out0,...]]

        self.resolved_conns = []  # (conn_id, tr_idx1, t1, tr_idx2, t2, chi2)

        self.all_ids_i = set()  # (conn_id, tr_idx)
        self.all_ids_o = set()  # (conn_id, tr_idx)
        self.id_map_i = {}  # (conn_id, tr_idx)->idx
        self.id_rmap_i = {}  # idx->(conn_id, tr_idx)
        self.id_map_o = {}  # (conn_id, tr_idx)->idx
        self.id_rmap_o = {}  # idx->(conn_id, tr_idx)

        self.conn_ids_io = {}  # conn_idx->(set(in_ids), set(out_ids))  map to local indeces

        self._chi2_f = None

        self._start_nodes = []
        self._end_nodes = []
        self._costs = []

        self.solved_cost = 0

        self.min_cost_asgn = None

        # self.nc_uid = -1

    def add_conn_node(self, conn_idx, tr_idx, t, in0_out1):
        if conn_idx not in self.all_connections:
            self.all_connections[conn_idx] = {}

        c = self.all_connections[conn_idx]

        if tr_idx == -1:
            if not self.explicit_nc:
                raise ValueError('not connected tracks (id==-1) are allowed only if the explicit_nc flag is set')
            else:
                raise ValueError('Sorry, actually you need to create them explicitly, and properly evaluate chi2')

            tr_idx = self.nc_uid
            # self.nc_uid -= 1

        if tr_idx not in c:
            c[tr_idx] = [[], []]

        tr_io = c[tr_idx]
        tr_io[in0_out1].append(t)

    def add_conn_node_in(self, conn_idx, tr_idx, t):
        self.add_conn_node(conn_idx, tr_idx, t, 0)

    def add_conn_node_out(self, conn_idx, tr_idx, t):
        self.add_conn_node(conn_idx, tr_idx, t, 1)

    def set_chi2_fn(self, f):
        """ f(conn_idx, tr_idx1, t1, tr_idx2, t2) returns corresponding chi2 """
        self._chi2_f = f

    def find_suitable_tracks_cross_within(self):
        # finds all tracks that either enter and exit a connection or stay within only
        # and adds them to the all_ids_i/all_ids_o sets

        for conn_idx, c in self.all_connections.items():
            self.conn_ids_io[conn_idx] = [set(), set()]

            for tr_idx, tr_io_t in c.items():
                # sort in/out times per track for easy type check
                ts_i = tr_io_t[0]
                ts_o = tr_io_t[1]
                ts_i.sort()
                ts_o.sort()

                is_in = is_out = False

                if len(ts_i):
                    if len(ts_o) == 0 or (ts_i[0] <= ts_o[0]):
                        is_in = True

                if len(ts_o):
                    if len(ts_i) == 0 or (ts_i[-1] <= ts_o[-1]):
                        is_out = True

                if is_in and is_out:
                    self.all_ids_i.add((conn_idx, tr_idx))
                elif not is_out and not is_in:
                    self.all_ids_o.add((conn_idx, tr_idx))

    def find_suitable_tracks_io(self):
        # finds all tracks that either enter a connection or exit (not both and not stay only within)
        # and adds them to the all_ids_i/all_ids_o sets

        for conn_idx, c in self.all_connections.items():
            self.conn_ids_io[conn_idx] = [set(), set()]

            for tr_idx, tr_io_t in c.items():
                # sort in/out times per track for easy type check
                ts_i = tr_io_t[0]
                ts_o = tr_io_t[1]
                ts_i.sort()
                ts_o.sort()

                is_in = is_out = False

                if len(ts_i):
                    if len(ts_o) == 0 or (ts_i[0] <= ts_o[0] <= ts_o[-1] <= ts_i[-1]):
                        is_in = True

                if len(ts_o):
                    if len(ts_i) == 0 or (ts_o[0] <= ts_i[0] <= ts_i[-1] <= ts_o[-1]):
                        is_out = True

                if is_in and not is_out:
                    self.all_ids_i.add((conn_idx, tr_idx))
                elif is_out and not is_in:
                    self.all_ids_o.add((conn_idx, tr_idx))

    def find_suitable_tracks(self):
        return (self.find_suitable_tracks_io() if self.in_to_out else
                self.find_suitable_tracks_cross_within())

    def fill_local_info_implicit_nc(self):
        self.id_map_i = {}
        self.id_rmap_i = {}
        self.id_map_o = {}
        self.id_rmap_o = {}

        for uid, conn_tr in enumerate(self.all_ids_i):
            conn_idx, tr_idx = conn_tr
            self.id_map_i[conn_tr] = uid
            self.id_rmap_i[uid] = conn_tr
            self.conn_ids_io[conn_idx][0].add(uid)

        for uid, conn_tr in enumerate(self.all_ids_o):
            conn_idx, tr_idx = conn_tr
            self.id_map_o[conn_tr] = uid
            self.id_rmap_o[uid] = conn_tr
            self.conn_ids_io[conn_idx][1].add(uid)

        n_i = len(self.all_ids_i)
        n_o = len(self.all_ids_o)

        for conn_idx, io_sets in self.conn_ids_io.items():
            i_ids, o_ids = io_sets
            conn = self.all_connections[conn_idx]

            # fill real conn values & corresponding dummies
            for i in i_ids:
                conn_i, tr_idx_i = self.id_rmap_i[i]
                t_i = conn[tr_idx_i][0][0]  # first in time for the track in this connection

                for o in o_ids:
                    conn_o, tr_idx_o = self.id_rmap_o[o]
                    t_o = conn[tr_idx_o][1][-1]  # last out time for the track in this connection

                    assert (conn_i == conn_o == conn_idx)

                    chi2 = self._chi2_f(conn_idx, tr_idx_i, t_i, tr_idx_o, t_o)  # conn_idx, tr_idx1, t1, tr_idx2, t2
                    if chi2 < 0:
                        chi2 = LAPSolver.chi2_no_conn * 100

                    # normal connection:
                    self._start_nodes.append(i)
                    self._end_nodes.append(o)
                    self._costs.append(chi2)

                    # complementary:
                    self._start_nodes.append(o + n_i)
                    self._end_nodes.append(i + n_o)
                    self._costs.append(0)

            # not connected ins
            for i in i_ids:
                self._start_nodes.append(i)
                self._end_nodes.append(i + n_o)
                self._costs.append(LAPSolver.chi2_no_conn)

                # not connected outs
            for o in o_ids:
                self._start_nodes.append(o + n_i)
                self._end_nodes.append(o)
                self._costs.append(LAPSolver.chi2_no_conn)

    def fill_local_info_explicit_nc(self):
        self.id_map_i = {}
        self.id_rmap_i = {}
        self.id_map_o = {}
        self.id_rmap_o = {}

        for uid, conn_tr in enumerate(self.all_ids_i):
            conn_idx, tr_idx = conn_tr
            self.id_map_i[conn_tr] = uid
            self.id_rmap_i[uid] = conn_tr
            self.conn_ids_io[conn_idx][0].add(uid)

        for uid, conn_tr in enumerate(self.all_ids_o):
            conn_idx, tr_idx = conn_tr
            self.id_map_o[conn_tr] = uid
            self.id_rmap_o[uid] = conn_tr
            self.conn_ids_io[conn_idx][1].add(uid)

        n_i = len(self.all_ids_i)
        n_o = len(self.all_ids_o)


        assert n_i == n_o, f'fill_local_info_explicit_nc: n_in={n_i}, n_out={n_o}'

        for conn_idx, io_sets in self.conn_ids_io.items():
            i_ids, o_ids = io_sets
            conn = self.all_connections[conn_idx]

            # fill real conn values for track to track
            # and corresponding dummies for track-nc & nc-nc
            for i in i_ids:
                conn_i, tr_idx_i = self.id_rmap_i[i]
                t_i = conn[tr_idx_i][0][0]  # first in time for the track in this connection

                # i_is_nc = tr_idx_i < 0

                for o in o_ids:
                    conn_o, tr_idx_o = self.id_rmap_o[o]
                    t_o = conn[tr_idx_o][1][-1]  # last out time for the track in this connection

                    # o_is_nc = tr_idx_o < 0

                    assert (conn_i == conn_o == conn_idx)  # make sure we didn't mess up
                    if tr_idx_i == tr_idx_o:
                        continue

                    # if i_is_nc and o_is_nc:
                    #     #chi2 = LAPSolver.chi2_no_conn*2
                    #     continue  # no connection of nc to nc directly

                    # elif i_is_nc or o_is_nc:
                    #     chi2 = LAPSolver.chi2_no_conn
                    # else:
                    chi2 = self._chi2_f(conn_idx,
                                        tr_idx_i, t_i,
                                        tr_idx_o, t_o)  # conn_idx, tr_idx1, t1, tr_idx2, t2

                    if chi2 < 0:
                        chi2 = LAPSolver.chi2_no_conn * 100

                    # normal connection:
                    self._start_nodes.append(i)
                    self._end_nodes.append(o)
                    self._costs.append(chi2)

    def fill_local_info(self):
        return self.fill_local_info_explicit_nc() if self.explicit_nc else self.fill_local_info_implicit_nc()

    def solve(self):
        self.find_suitable_tracks()
        self.fill_local_info()

        if len(self._costs) == 0:
            self.resolved_conns = []
            return True

        chi_min, chi_max = min(self._costs), max(self._costs)
        d_chi = max(1, chi_max - chi_min)

        # chi2_i = (chi2 - chi_min) / d_chi * 100000

        min_cost_asgn = pg.LinearSumAssignment()
        self.min_cost_asgn = min_cost_asgn

        # print(self._start_nodes)
        # print(self._end_nodes)
        # print(self._capacities )
        # print(self._costs)
        # Add each arc.
        for i, o, chi2 in zip(self._start_nodes, self._end_nodes, self._costs):
            cost = (chi2 - chi_min) / d_chi * 100000
            cost = int(cost)
            min_cost_asgn.AddArcWithCost(i, o, cost)

        res = min_cost_asgn.Solve()

        if res == min_cost_asgn.OPTIMAL:
            self.solved_cost = 0  # (min_cost_asgn.OptimalCost() / 100000*d_chi) + chi_min

            n_i = len(self.all_ids_i)
            n_o = len(self.all_ids_o)
            for i in range(n_i):  # we don't care of not-connected o-s, w/ indeces [n_i,...n_i+n_o-1]
                o = min_cost_asgn.RightMate(i)
                if o < n_o:  # only real connections:
                    conn_i, tr_idx_i = self.id_rmap_i[i]
                    conn_o, tr_idx_o = self.id_rmap_o[o]

                    assert (conn_i == conn_o)
                    conn = self.all_connections[conn_i]
                    t_i = conn[tr_idx_i][0][0]  # first in time for the track in this connection
                    t_o = conn[tr_idx_o][1][-1]  # last out time for the track in this connection

                    chi2 = self._chi2_f(conn_i, tr_idx_i, t_i, tr_idx_o, t_o)
                    self.solved_cost += chi2

                    self.resolved_conns.append((conn_i, tr_idx_i, t_i, tr_idx_o, t_o, chi2))
            return True
        elif res == min_cost_asgn.INFEASIBLE:
            print('No assignment is possible.')
            arr = []
            for i, o, chi2 in zip(self._start_nodes, self._end_nodes, self._costs):
                cost = (chi2 - chi_min) / d_chi * 100000
                cost = int(cost)
                arr.append((i, o, cost))
            print(arr)
        elif res == min_cost_asgn.POSSIBLE_OVERFLOW:
            print('Some input costs are too large and may cause an integer overflow.')


def F_stat_prob(m1, m2, s1, s2, n1, n2):
    delta = m1 - m2
    sum_var = s1 ** 2 + s2 ** 2
    dfd = n1 + n2 - 2

    if dfd > 0 and n1 > 0 and n2 > 0:
        t2 = delta ** 2 / sum_var * n1 * n2 / (n1 + n2)
        p = (1 - stats.f.cdf(t2, 1, dfd)) if isfinite(t2) else 1e-10
    else:
        p = 1 / sqrt(np.e)
    return p


def F_stat_NLL(m1, m2, s1, s2, n1, n2):
    p = F_stat_prob(m1, m2, s1, s2, n1, n2)
    return -2 * log(p)


def chi2_ndf_1(chi2, ndf, chi2_inf=75):
    p = stats.chi2.cdf(chi2, ndf)
    return stats.chi2.ppf(p, 1) if p != 1 else chi2_inf


class Node:
    # x, y, z, w, phi, eccentr
    def __init__(self, x: float, y: float, z: float, w: float, phi: float, eccentr: float, *args):
        self.r = (x, y, z)
        self.w = w
        self.phi = phi
        self.eccentr = eccentr
        self.pars = args

    def __repr__(self):
        return 'Node: x:%.1f y:%.1f z:%.1f w:%.1f phi:%.1f° e:%.2f' % (*self.r,
                                                                       self.w,
                                                                       self.phi / np.pi * 180,
                                                                       self.eccentr
                                                                       )


class TSegment:  # track segment, valid parts of a track
    _NC_TP_START = 1  # virtual segment goes to the future, node to the past
    _NC_TP_END = -1  # virtual segment goes to the past, node to the future

    def __init__(self, nc_type=None):
        self.nodes = []
        self.times = []
        self.updated = False
        self.n = 0

        self.local_v = []  # sqrt velocity between nodes [i] & [i+1]
        self.local_lin_v = []  # sqrt linear velocity between nodes [i] & [i+n_lin+1]. n_lin==6
        self.local_lin_dir = []  # unit-vector of linear velocity (i->i+n_lin nodes)
        self.local_ecc_phi = []  # eliptic fit of nodes[i:i+n_lin+1] positions. proxy to directionality (eccentricitet and angle of big axis, [0,\pi])
        self.local_MI = []  # meandering index
        self.local_kink = []  # angle between vectors (n_(i)->n_(i+1)) and (n_(i+1)->n_(i+2))
        self.w = []  # area/weight
        self.r_min = [0, 0, 0]
        self.r_max = [0, 0, 0]

        self.is_nc = nc_type is not None  # is a virtual segment, to represent missing (not connected) tracks
        self.nc_type = nc_type

        self.info = None

    def add_node(self, n: Node, t: float):
        self.nodes.append(n)
        self.times.append(t)

        self.updated = True

    @staticmethod
    def node_distance(n1: Node, n2: Node):
        r1, r2 = n1.r, n2.r
        dr = [r2i - r1i for r1i, r2i in zip(r1, r2)]
        dr2 = np.dot(dr, dr)
        return sqrt(dr2)

    @staticmethod
    def node_dr(n1: Node, n2: Node):
        r1, r2 = n1.r, n2.r
        dr = np.array([r2i - r1i for r1i, r2i in zip(r1, r2)])
        return dr

    @staticmethod
    def node_kink(n1: Node, n2: Node, n3: Node):
        r1, r2, r3 = n1.r, n2.r, n3.r
        a = [r2i - r1i for r1i, r2i in zip(r1, r2)]  # dr12 = a
        b = [r3i - r2i for r2i, r3i in zip(r2, r3)]  # dr23 = b

        a2 = np.dot(a, a)
        b2 = np.dot(b, b)
        ab = np.dot(a, b)

        cos_phi = ab / sqrt(a2 * b2) if (a2 > 0 and b2 > 0) else 1
        cos_phi = min(1, max(-1, cos_phi))
        return acos(cos_phi)

    def update(self, n_lin=6):
        if self.updated:
            self.updated = False

            # sort
            self.n = len(self.times)
            if self.n > 0:
                sorted_idx_t = sorted(enumerate(self.times), key=lambda x: x[1])
                self.times = [t for i, t in sorted_idx_t]
                self.nodes = [self.nodes[i] for i, t in sorted_idx_t]

                rs = np.array([n.r for n in self.nodes])
                self.r_min = rs.min(axis=0)
                self.r_max = rs.max(axis=0)

                self.w = [n.w for n in self.nodes]

                if self.n > 1:  # we can evaluate velocities
                    t_from, t_to = self.times[:-1], self.times[1:]
                    n_from, n_to = self.nodes[:-1], self.nodes[1:]

                    dr_dts = [(self.node_distance(n1, n2), t2 - t1) for t1, n1, t2, n2 in
                              zip(t_from, n_from, t_to, n_to)]
                    self.local_v = [(sqrt(dr / dt) if dt > 0 else 0, dr, dt) for dr, dt in dr_dts]
                self.local_v.append(None)  #

                if self.n > 2:
                    t_from, t_through, t_to = self.times[:-2], self.times[1:-1], self.times[2:]
                    n_from, n_through, n_to = self.nodes[:-2], self.nodes[1:-1], self.nodes[2:]

                    self.local_kink = [(self.node_kink(n1, n2, n3),
                                        t2 - t1,
                                        t3 - t2)
                                       for t1, n1, t2, n2, t3, n3 in
                                       zip(t_from, n_from, t_through, n_through, t_to, n_to)]
                self.local_kink = [None] + self.local_kink + [None]

                assert (n_lin > 0)
                assert (n_lin % 2 == 0)  # even number
                if self.n > n_lin:  # we can evaluate ~ linear velocities
                    t_0 = self.times[:self.n - n_lin]
                    n_0 = self.nodes[:self.n - n_lin]
                    t_n = self.times[n_lin:]
                    n_n = self.nodes[n_lin:]

                    dr_0_nlin = [self.node_dr(n_nlin, n0) for n0, n_nlin in zip(n_0, n_n)]
                    dt_0_nlin = [t_nlin - t0 for t0, t_nlin in zip(t_0, t_n)]
                    ds_0_nlin = [self.node_distance(n_nlin, n0) for n0, n_nlin in zip(n_0, n_n)]

                    ds_i_ip1 = [self.node_distance(n_ip1, n_i) for n_i, n_ip1 in zip(self.nodes[:-1], self.nodes[1:])]
                    ds_sum_nlin = [np.sum(ds_i_ip1[i:i + n_lin]) for i in range(self.n - 1 - n_lin)]

                    drv_dr_dst_dt = [(dr_0_nlin_j,
                                      ds_0_nlin_j,
                                      ds_sum_nlin_j,
                                      dt_0_nlin_j) \
                                     for dr_0_nlin_j, ds_0_nlin_j, ds_sum_nlin_j, dt_0_nlin_j \
                                     in zip(dr_0_nlin, ds_0_nlin, ds_sum_nlin, dt_0_nlin)]

                    if (False):  # old version, while new remains TBT
                        t_0, n_0 = self.times[0:-6], self.nodes[0:-6]
                        t_1, n_1 = self.times[1:-5], self.nodes[1:-5]
                        t_2, n_2 = self.times[2:-4], self.nodes[2:-4]
                        t_3, n_3 = self.times[3:-3], self.nodes[3:-3]
                        t_4, n_4 = self.times[4:-2], self.nodes[4:-2]
                        t_5, n_5 = self.times[5:-1], self.nodes[5:-1]
                        t_6, n_6 = self.times[6:], self.nodes[6:]

                        drv_dr_dst_dt = [(self.node_dr(n6, n0),
                                          self.node_distance(n6, n0),
                                          self.node_distance(n1, n0) +
                                          self.node_distance(n2, n1) +
                                          self.node_distance(n3, n2) +
                                          self.node_distance(n4, n3) +
                                          self.node_distance(n5, n4) +
                                          self.node_distance(n6, n5),
                                          t6 - t0) \
                                         for t0, t6, n0, n1, n2, n3, n4, n5, n6 in
                                         zip(t_0, t_6, n_0, n_1, n_2, n_3, n_4, n_5, n_6)]

                    self.local_MI = [((dst / dr) if dr > 0 else 1, dr, dt) for drv, dr, dst, dt in drv_dr_dst_dt]
                    self.local_lin_v = [(sqrt(dr / (dt if dt > 0 else 1)), dr, dt) for drv, dr, dst, dt in
                                        drv_dr_dst_dt]

                    self.local_lin_dir = [drv / (dr if dr > 0 else 1) for drv, dr, dst, dt in drv_dr_dst_dt]

                    # todo: elips fit 7 nodes
                    nodes_xyz = [ni.r for ni in self.nodes]
                    nodes_xyz_local = [nodes_xyz[i:i + n_lin + 1] for i in range(self.n - n_lin)]  # sh==(nt, 7, 3,)
                    nodes_xyz_local = np.array(nodes_xyz_local)
                    # print(nodes_xyz_local.shape, self.n)
                    nodes_xyz_local = nodes_xyz_local.transpose([0, 2, 1])  # sh==(nt, 3, 7)
                    self.local_ecc_phi = [ellipse_fit(nodes_x, nodes_y) for nodes_x, nodes_y, nodes_z in
                                          nodes_xyz_local]

                n_pad = n_lin // 2
                self.local_MI = [None] * n_pad + self.local_MI + [None] * n_pad
                self.local_lin_v = [None] * n_pad + self.local_lin_v + [None] * n_pad
                self.local_lin_dir = [None] * n_pad + self.local_lin_dir + [None] * n_pad
                self.local_ecc_phi = [None] * n_pad + self.local_ecc_phi + [None] * n_pad

    def aggregate(self, f, arg):
        self.update()
        for n, t in zip(self.nodes, self.times):
            arg = f(n, t, arg)
        return arg

    def aggregate_nodes_in_time_interval(self, f, arg, t1, t2):
        self.update()
        for n, t in zip(self.nodes, self.times):
            if t1 <= t <= t2:
                f(n, t, arg)

    def aggregate_pars_in_time_interval(self, f, arg, t1, t2):
        self.update()

        all_pars = (self.local_v,
                    self.local_lin_v,
                    self.local_lin_dir,
                    self.local_ecc_phi,
                    self.local_MI,
                    self.local_kink,
                    self.w)

        for t_v_linv_lindir_eccphi_MI_kink_w in zip(self.times, *all_pars):
            t = t_v_linv_lindir_eccphi_MI_kink_w[0]
            if t1 <= t <= t2:
                f(t_v_linv_lindir_eccphi_MI_kink_w, arg)

    def get_last_time(self):
        self.update()
        return self.times[-1] if self.n > 0 else -1

    def get_first_time(self):
        self.update()
        return self.times[0] if self.n > 0 else -1

    def get_last_pos(self):
        self.update()
        return self.nodes[-1].r if self.n > 0 else (0, 0, 0)

    def get_first_pos(self):
        self.update()
        return self.nodes[0].r if self.n > 0 else (0, 0, 0)

    def get_closest_time(self, t):
        if self.n == 0:
            return t

        t_c = self.times[0]
        dt = abs(t - t_c)
        for ti in self.times[1:]:
            dti = abs(t - ti)
            if dti < dt:
                dt = dti
                t_c = ti

            if dti > dt:  # Since times are sorted, after interval increases, it won't decrease ever again
                break

    def get_num_nodes(self):
        return len(self.times)


class Track:
    UID = 0

    # t_v_linv_lindir_eccphi_MI_kink_w
    priors = copy.deepcopy(cfgm.TRACK_PRIORS)

    _chi2_pars = ['s_linv',
                  's_move_v',
                  'ecc',
                  # 'kink',
                  'directionality',
                  'w',
                  'vector'
                  ]

    # per parameter: population mean, population sigma^2, per instance mean sigma^2
    # 'v' is sqrt(velocity)

    # fiducial volume correction of NLL
    _l1 = cfgm.BORDER_ATT_DIST  # um from boundary
    _l2 = cfgm.CELL_RADIUS
    _v1 = 0.01  # value of NLL at _l1
    _v2 = 100  # value of NLL at _l2
    _NLL_alpha = _v1 / ((_v1 / _v2) ** (_l1 / (_l1 - _l2)))
    _NLL_lambda = np.log(_v1 / _v2) / (_l1 - _l2)

    _boundary_x0x1y0y1 = None

    @staticmethod
    def get_fiducial_NLL(pos):
        if Track._boundary_x0x1y0y1 is None:
            return 0

        x, y = pos[:2]
        x0, x1, y0, y1 = Track._boundary_x0x1y0y1

        delta = np.min(np.abs([x0 - x, x1 - x, y0 - y, y1 - y]))
        nll = Track._NLL_alpha * np.exp(Track._NLL_lambda * delta)

        return nll

    @staticmethod
    def in_fiducial_volume(pos):
        if Track._boundary_x0x1y0y1 is None:
            return True

        x, y = pos[:2]
        x0, x1, y0, y1 = Track._boundary_x0x1y0y1

        delta = np.min([x - x0, x1 - x, y - y0, y1 - y])
        return delta >= Track._l1

    @staticmethod
    def set_fiducial_boundary(boundary_x0x1y0y1):
        Track._boundary_x0x1y0y1 = boundary_x0x1y0y1

    def __init__(self, segment: TSegment = None):
        self.segments = []
        # self.connection_ids = set()  # in connections
        if segment is not None:
            self.segments.append(segment)

        self.updated = True
        self.uid = Track.UID
        self.merged_into = None
        self.in_fid_vol = True

        self.merged_uids = {self.uid}

        self.ds_id = None

        Track.UID += 1

    def set_ds_id(self, ds_id):
        self.ds_id = ds_id

    def sort_segments_by_time(self):
        self.segments.sort(key=lambda seg: seg.get_first_time())

    def add_segment(self, segment: TSegment):
        self.segments.append(segment)
        self.sort_segments_by_time()
        self.updated = True

    def merge_track(self, track):
        assert (track.merged_into == None)

        self.segments.extend(track.segments)
        self.updated = True
        self.sort_segments_by_time()

        track.clear()
        track.merged_into = self.uid

        self.merged_uids.add(track.uid)
        self.merged_uids.update(track.merged_uids)

    def reset_updated(self):
        self.updated = False

    def is_updated(self):
        return self.updated

    def aggregate(self, f, arg):
        n_seg = len(self.segments)
        for s in self.segments:
            if (not s.is_nc) or n_seg == 1:
                arg = f(s, arg)
        return arg

    def aggregate_nodes(self, f, arg):
        n_seg = len(self.segments)
        for s in self.segments:
            if (not s.is_nc) or n_seg == 1:
                arg = s.aggregate(f, arg)
        return arg

    def aggregate_nodes_in_time_interval(self, f, arg, t1, t2):
        """
        Params:
            f(n,t,arg) (method): function to be called on each element
            arg (list): list whose elements will be updated. arg[0] = num aggregated, arg[1] - main collection.
            t1 (float): starting time from which on items will be iterated, inclusive
            t2 (float): end time to which items will be iterated, inclusive
        """
        n_seg = len(self.segments)

        suitable_segs = [s for s in self.segments if (t1 <= s.get_last_time() and
                                                      s.get_first_time() <= t2 and
                                                      ((not s.is_nc) or n_seg == 1)
                                                      )]
        for s in suitable_segs:
            s.aggregate_nodes_in_time_interval(f, arg, t1, t2)
        return arg

    def aggregate_pars_in_time_interval(self, f, arg, t1, t2):
        """
        Params:
            f(pars,arg) (method): function to be called on each element. last known pars - (t,sqrt(v),mi,phi,w).
                                  (t, sqrt(v), sqrt(linv), lindir, eccphi, MI, kink, w),
                                  s_v, s_linv, lindir, eccphi, MI, and kink be None in which case should be ignored
            arg (list): list whose elements will be updated. arg[0] = num aggregated, arg[1] - main collection.
            t1 (float): starting time from which on items will be iterated, inclusive
            t2 (float): end time to which items will be iterated, inclusive
        """
        n_seg = len(self.segments)
        suitable_segs = [s for s in self.segments if (t1 <= s.get_last_time() and
                                                      s.get_first_time() <= t2 and
                                                      ((not s.is_nc) or n_seg == 1)
                                                      )]
        for s in suitable_segs:
            s.aggregate_pars_in_time_interval(f, arg, t1, t2)
        return arg

    def aggregate_pars_around_time(self, f, arg, t0, dt):
        return self.aggregate_pars_in_time_interval(f, arg, t0 - dt, t0 + dt)

    def get_closest_node_time(self, t):
        ints = self.get_segmment_intervals()

        closest_t = -1
        for s in self.segments:
            closest_t = s.get_first_time()
            if closest_t != -1:
                break

        dt = abs(t - closest_t)

        for s_idx, (interval, s) in enumerate(ints, self.segments):
            t1, t2 = interval
            if t1 <= t <= t2:
                closest_t = s.get_closest_time(t)
                break
            elif t2 < t:
                dti = t - t2
                if dti < dt:
                    dt = dti
                    closest_t = t2
            elif t < t1:
                dti = t1 - t
                if dti <= dt:
                    dt = dti
                    closest_t = t1
                else:
                    break  # keeps increasing, since segs are sorted
        return closest_t

    def overlaps_with(self, track, max_overlap=0, max_gap=0):
        assert (track.merged_into == None)
        assert (self.merged_into == None)

        # if both tracks are NC, speciall case applies
        if self.is_nc() and track.is_nc():
            seg0_s = self.segments[0]
            seg0_t = track.segments[0]
            nc_s = seg0_s.nc_type
            nc_t = seg0_t.nc_type
            if nc_s == nc_t:
                return True
            else:
                if nc_s == TSegment._NC_TP_END:
                    if seg0_s.times[0] > seg0_t.times[0]:
                        return True
                else:
                    if seg0_s.times[0] < seg0_t.times[0]:
                        return True
            return False

        own_times = set()
        track_times = set()

        def get_times_set(n, t, timeset: set):
            timeset.add(t)
            return timeset

        own_times = self.aggregate_nodes(get_times_set, own_times)
        track_times = track.aggregate_nodes(get_times_set, track_times)

        overlap_times = own_times.intersection(track_times)
        overlap_times = list(overlap_times)
        overlap_times.sort()

        # count longest chain
        t_none = -1  # no negative times assumed
        t_prev = t_none
        max_len = 0
        seq_len = 0
        for t in overlap_times:
            if t_prev == t_none:
                seq_len = 1

            step = t - t_prev
            if step <= 1 + max_gap:
                seq_len += step
                max_len = max(max_len, seq_len)
            else:
                seq_len = 1

            t_prev = t

        max_len = max(max_len, seq_len)

        return max_len > max_overlap

    def connection_possible(self, track, print_dbg=False):
        """
        The distance between end-points does not prohibit connection,
        i.e. is 'not too far'. Is used to skip unnecessary chi2 computations
        """

        # find all conn points
        endpoints = []  # (s_e, tr1(self)_tr2(track), t, r)
        #         for seg in self.segments:
        #             endpoints.append((0, 0, seg.get_first_time(), seg.get_first_pos()))
        #             endpoints.append((1, 0, seg.get_last_time(), seg.get_last_pos()))
        #         for seg in track.segments:
        #             endpoints.append((0, 1, seg.get_first_time(), seg.get_first_pos()))
        #             endpoints.append((1, 1, seg.get_last_time(), seg.get_last_pos()))

        n_segs = len(self.segments)
        for seg in self.segments:
            not_s_nc = not seg.is_nc
            if n_segs == 1 or not_s_nc:
                if not_s_nc or seg.nc_type == TSegment._NC_TP_START:
                    endpoints.append((0, 0, seg.get_first_time(), seg.get_first_pos()))
                if not_s_nc or seg.nc_type == TSegment._NC_TP_END:
                    endpoints.append((1, 0, seg.get_last_time(), seg.get_last_pos()))

        n_segt = len(track.segments)
        for seg in track.segments:
            not_s_nc = not seg.is_nc
            if n_segt == 1 or not_s_nc:
                if not_s_nc or seg.nc_type == TSegment._NC_TP_START:
                    endpoints.append((0, 1, seg.get_first_time(), seg.get_first_pos()))
                if not_s_nc or seg.nc_type == TSegment._NC_TP_END:
                    endpoints.append((1, 1, seg.get_last_time(), seg.get_last_pos()))

        endpoints.sort(key=lambda se_t1t2_t_r: se_t1t2_t_r[2])

        if print_dbg:
            print(endpoints)

        # fill nearest pairs
        conn_point_pairs = []
        for pt1, pt2 in zip(endpoints[:-1], endpoints[1:]):
            se_1, tr1, *_ = pt1
            se_2, tr2, *_ = pt2

            if se_1 == 1 and se_2 == 0 and tr1 != tr2:
                conn_point_pairs.append([pt1, pt2])

        if print_dbg:
            print(conn_point_pairs)

        n_par = 0
        nll_sum = 0

        # evaluate chi2 by params as well as the distance around each connections (disabled)
        tr1_tr2 = [self, track]

        closest_pos = None

        if len(conn_point_pairs) == 0:
            return False
        conn_point_pairs.sort(key=lambda pt1_pt2: abs(pt1_pt2[0][2] - pt1_pt2[1][2]))

        pt1, pt2 = conn_point_pairs[0]
        _, tr1_idx, t1, pos1 = pt1  # end of track segment
        _, tr2_idx, t2, pos2 = pt2  # start of track segment
        r = Track.r(pos1, pos2)

        sv_m, sv_s2, _ = Track.priors['s_linv']
        v_max = (sv_m + 3 * sqrt(sv_s2)) ** 2
        dt = abs(t2 - t1)
        if print_dbg:
            print(r, dt, v_max)

        return r < (dt * v_max + cfgm.MAX_TRACK_CONN_RAD_OFS) and (dt <= cfgm.MAX_TRACK_CONN_RAD_DT)

    def get_mean_std(self, bayesian_estimators=True, excl_outliers=True,
                     return_arr=False, t0=0,
                     t_range=1000000, cmp_dt=1):

        # cmp_dt - dt of 2 tracks for comparission of which the mean/std are being evaluated
        assert (self.merged_into == None)

        # ToDo: use cached values & the updated flag

        if len(self.segments) == 1 and self.segments[0].is_nc:
            res = {}
            for k, (m, s2, t2) in Track.priors.items():
                res[k] = [m, sqrt(s2), 2]
            if return_arr:
                return res, []
            else:
                return res

        all_vals = [0, {'s_v': [],  #
                        's_v_t': [],  #
                        's_linv': [],  #
                        's_linv_t': [],  #
                        'lindir': [],  #
                        'lindir_t': [],  #
                        'ecc': [],  #
                        'ecc_t': [],  #
                        'phi': [],  #
                        'phi_t': [],  #
                        'MI': [],  #
                        'MI_t': [],  #
                        'kink': [],  #
                        'kink_t': [],  #
                        'w': [],  #
                        'w_t': [],  #
                        }
                    ]

        def get_all_pars(pars, all_vals: List):
            t, s_v_p, s_linv_p, lindir, eccphi_p, MI_p, kink_p, w = pars
            all_vals[0] += 1

            dt = None
            dt_lin = None

            if (s_v_p):
                sv, dr, dt = s_v_p
                if cfgm.TRACK_PARS_DT_RANGE_SV[0] < dt < cfgm.TRACK_PARS_DT_RANGE_SV[1]:  # 0->3
                    all_vals[1]['s_v'].append(sv)
                    all_vals[1]['s_v_t'].append(t)

            if (s_linv_p):
                sv, dr, dt_lin = s_linv_p
                if cfgm.TRACK_PARS_DT_RANGE_LINV[0] < dt_lin < cfgm.TRACK_PARS_DT_RANGE_LINV[1]:  # 5->10
                    all_vals[1]['s_linv'].append(sv)
                    all_vals[1]['s_linv_t'].append(t)

            if (lindir is not None and dt_lin is not None):
                if cfgm.TRACK_PARS_DT_RANGE_LINV[0] < dt_lin < cfgm.TRACK_PARS_DT_RANGE_LINV[1]:  # 5->10
                    all_vals[1]['lindir'].append(lindir)
                    all_vals[1]['lindir_t'].append(t)

            if (eccphi_p and dt_lin):
                ecc, phi = eccphi_p
                if cfgm.TRACK_PARS_DT_RANGE_LINV[0] < dt_lin < cfgm.TRACK_PARS_DT_RANGE_LINV[1]:  # 5->10
                    all_vals[1]['phi'].append(phi)
                    all_vals[1]['phi_t'].append(t)
                    all_vals[1]['ecc'].append(ecc)
                    all_vals[1]['ecc_t'].append(t)

            if (MI_p):
                MI, dr, dt_lin = MI_p
                if cfgm.TRACK_PARS_DT_RANGE_LINV[0] < dt_lin < cfgm.TRACK_PARS_DT_RANGE_LINV[1]:  # 5->10
                    all_vals[1]['MI'].append(MI)
                    all_vals[1]['MI_t'].append(t)

            if (kink_p):
                kink_phi, dt1, dt2 = kink_p
                if dt1 == 1 and dt2 == 1:
                    all_vals[1]['kink'].append(kink_phi)
                    all_vals[1]['kink_t'].append(t)

            all_vals[1]['w'].append(w)
            all_vals[1]['w_t'].append(t)

        all_vals = self.aggregate_pars_around_time(get_all_pars, all_vals, t0, t_range)
        # print(all_vals)
        n = all_vals[0]

        res = {'s_linv': [0, 0, 0],
               's_move_v': [0, 0, 0],
               'ecc': [0, 0, 0],
               'kink': [0, 0, 0],
               'w': [0, 0, 0],
               'directionality': [0, 0, 0],
               'vector': [(1, 1, 1), 0, 0]  # value closes to
               }

        if n == 0:
            return res

        for param, msn in res.items():
            arr = None
            if param == 's_move_v':  # not original param names
                s_linv_arr = all_vals[1]['s_linv']
                MI_arr = all_vals[1]['MI']
                arr = [s_linv * sqrt(MI) for s_linv, MI in zip(s_linv_arr, MI_arr)]
            elif param == 'directionality':
                lindir_arr = all_vals[1]['lindir']
                phi_arr = all_vals[1]['phi']
                lindir_phi_2D = [atan2(y, x) for x, y, z in lindir_arr]
                arr = [angle_to_mpi2ppi2(phi - lin_phi) for phi, lin_phi in zip(phi_arr, lindir_phi_2D)]
            elif param == 'vector':
                lindir_arr = all_vals[1]['lindir']
                lindir_t_arr = all_vals[1]['lindir_t']
                if len(lindir_t_arr) == 0:
                    msn[:] = 0, 0, n
                    continue

                lindir_dt_arr = [abs(t - t0) for t in lindir_t_arr]
                closest_idx = np.argmin(lindir_dt_arr)

                mean = lindir_arr[closest_idx]  # closest lindir_arr in time

                n_points = len(lindir_arr)
                n = n_points - cmp_dt
                if n > 1:
                    # try evaluating form variance of lindir across cmp_dt timesteps
                    # 1. get idx pairs
                    idx_pairs = []
                    for i1, t1 in enumerate(lindir_t_arr):
                        for i2_, t2 in enumerate(lindir_t_arr[i1 + 1:]):
                            i2 = i2_ + i1 + 1
                            dt = t2 - (t1 + cmp_dt)
                            if dt == 0:
                                idx_pairs.append((i1, i2))
                            if dt > 0:
                                break
                    # 2. if too little items - do differently
                    n = len(idx_pairs)
                    if n < 2:
                        n = 0
                    else:
                        # 3. collect delta_phi-s for std estimation
                        d_phi_lindir = [angle_3d(lindir_arr[i], lindir_arr[j]) for i, j in idx_pairs]
                        std = sqrt((np.asarray(d_phi_lindir) ** 2).sum() / (n - 1))

                if n <= 1:
                    # ToDo:
                    # estimate from all pairs as: std_1 * sqrt(dt) + std_0== d_phi(dt)
                    # then std=std1 * sqrt(cmp_dt) + std_0

                    # if failed (not enough points) - estimate std from eccentricitet:
                    ecc_arr = all_vals[1]['ecc']
                    dphi_1sigma = [atan(1 / ecc * 1.81899) for ecc in ecc_arr if ecc > 0]  # 1.81899==tan(pi*0.68/2)
                    n = len(dphi_1sigma)
                    if n > 1:
                        std = sqrt((np.asarray(dphi_1sigma) ** 2).sum() / (n - 1))
                    else:
                        n = 0
                        std = 0

                msn[:] = mean, std, n
                continue

            else:
                arr = all_vals[1][param]

            arr = [el for el in arr if el is not None]
            n = len(arr)
            if not n:
                continue
            if excl_outliers:
                arr_excl = exclude_outliers(arr)
                if len(arr_excl):
                    arr = arr_excl
            n = len(arr)
            if not n:
                continue

            m = np.mean(arr)

            if not bayesian_estimators:
                s = np.std(arr, ddof=1) if n > 2 else 0  # np.sqrt(5**2+2**2)
            else:
                prior = Track.priors[param]
                mu_pop, s2_pop, s2_instance = prior

                s2_pop_n = s2_pop * n
                denom_factor = 1. / (s2_pop_n + s2_instance)

                s2 = s2_instance * s2_pop * denom_factor
                mu = (s2_pop_n * m + s2_instance * mu_pop) * denom_factor

                m, s = mu, sqrt(s2 * n)

            msn[:] = m, s, n

        if return_arr:
            return res, all_vals
        else:
            return res

    @staticmethod
    def r2(pos1, pos2):
        dr2 = [(p1 - p2) ** 2 for p1, p2 in zip(pos1, pos2)]
        return np.sum(dr2)

    @staticmethod
    def r(pos1, pos2):
        return sqrt(Track.r2(pos1, pos2))

    @staticmethod
    def angle(pos_from, pos_to):
        dx = pos_to[0] - pos_from[0]
        dy = pos_to[1] - pos_from[1]
        cos_phi = (dx / sqrt(dx ** 2 + dy ** 2)) if (dx != 0 or dy != 0) else 0
        cos_phi = max(-1, min(1, cos_phi))  # correct possible numerical error
        return acos(cos_phi)  # 0..pi

    def is_nc(self):
        return np.all([seg.is_nc for seg in self.segments])
        n_seg = len(self.segments)
        return (n_seg == 1 and self.segments[0].is_nc)

    @staticmethod
    def _endpoints_consistent(sorted_endpoints):
        """
        Checks if tsegments of two tracks are consistent: ins and outs must alternate.
        In the same sequence can start and end on any type as NC segments have only one end
        """
        if len(sorted_endpoints) == 0:
            return False

        last_io, last_tr, *_ = sorted_endpoints[0][:2]

        for ep in sorted_endpoints[1:]:
            curr_io, curr_tr, *_ = ep

            if last_io == curr_io:
                return False

            last_io, last_tr = curr_io, curr_tr

        return True

    def get_connection_chi2(self, track, time_range=50,
                            print_dbg_nll=False, allow_overlapping=False, disable_dist=False
                            ):
        """
        Calculates nll of possible track to tracks connection
        """
        # s_m, s_s, s_n = self.get_mean_std()
        # t_m, t_s, t_n = track.get_mean_std()
        #
        # if s_n==0 or t_n==0:
        #    return -1

        # nll = F_stat_NLL(s_m, t_m, s_s, t_s, s_n, t_n)

        # find all conn points
        endpoints = []  # (s_e, tr1(self)_tr2(track), t, r)

        n_segs = len(self.segments)
        for seg in self.segments:
            not_s_nc = not seg.is_nc
            if n_segs == 1 or not_s_nc:
                if not_s_nc or seg.nc_type == TSegment._NC_TP_START:
                    nc_start_dt = 0.1 if seg.nc_type == TSegment._NC_TP_START else 0
                    endpoints.append((0, 0, seg.get_first_time() + nc_start_dt, seg.get_first_pos()))
                if not_s_nc or seg.nc_type == TSegment._NC_TP_END:
                    nc_end_dt = -0.1 if seg.nc_type == TSegment._NC_TP_END else 0
                    endpoints.append((1, 0, seg.get_last_time() + nc_end_dt, seg.get_last_pos()))

        n_segt = len(track.segments)
        for seg in track.segments:
            not_s_nc = not seg.is_nc
            if n_segt == 1 or not_s_nc:
                if not_s_nc or seg.nc_type == TSegment._NC_TP_START:
                    nc_start_dt = 0.1 if seg.nc_type == TSegment._NC_TP_START else 0
                    endpoints.append((0, 1, seg.get_first_time() + nc_start_dt, seg.get_first_pos()))
                if not_s_nc or seg.nc_type == TSegment._NC_TP_END:
                    nc_end_dt = -0.1 if seg.nc_type == TSegment._NC_TP_END else 0
                    endpoints.append((1, 1, seg.get_last_time() + nc_end_dt, seg.get_last_pos()))

        endpoints.sort(key=lambda se_t1t2_t_r: se_t1t2_t_r[2])
        if print_dbg_nll:
            print(endpoints)

        endpoints_consistent = self._endpoints_consistent(endpoints)
        if not endpoints_consistent:
            if not allow_overlapping:
                if print_dbg_nll:
                    print('endpoints inconsistent')
                return -1

        # fill nearest pairs
        conn_point_pairs = []
        if endpoints_consistent and not allow_overlapping:
            for pt1, pt2 in zip(endpoints[:-1], endpoints[1:]):
                se_1, tr1_idx, *_ = pt1
                se_2, tr2_idx, *_ = pt2

                if se_1 == 1 and se_2 == 0 and tr1_idx != tr2_idx:
                    conn_point_pairs.append([pt1, pt2])
        else:
            # all nearest pairs
            min_dt = np.min([abs(t2 - t1) for (_, tr1_idx, t1, _), (_, tr2_idx, t2, _)
                             in zip(endpoints[:-1], endpoints[1:]) if tr1_idx != tr2_idx])
            for pt1, pt2 in zip(endpoints[:-1], endpoints[1:]):
                se_1, tr1_idx, t1, _ = pt1
                se_2, tr2_idx, t2, _ = pt2

                if tr1_idx != tr2_idx and abs(t2 - t1) <= min(time_range, 1.3 * min_dt):
                    conn_point_pairs.append([pt1, pt2])

        if len(conn_point_pairs) == 0:
            if print_dbg_nll:
                print('connection not feasible')
            return -1
        else:
            if print_dbg_nll:
                print(conn_point_pairs)

        n_par = 0
        nll_sum = 0

        # evaluate chi2 by params as well as the distance around each connections (disabled)
        tr1_tr2 = [self, track]
        for pt1, pt2 in conn_point_pairs:
            _, tr1_idx, t1, pos1 = pt1  # end of track segment
            _, tr2_idx, t2, pos2 = pt2  # start of track segment

            tr1 = tr1_tr2[tr1_idx]
            tr2 = tr1_tr2[tr2_idx]

            adt = abs(t2 - t1)
            mean_std_dict_1 = tr1.get_mean_std(t0=t1, t_range=time_range, cmp_dt=adt)
            mean_std_dict_2 = tr2.get_mean_std(t0=t2, t_range=time_range, cmp_dt=adt)

            v_mean = 0
            dv_mean = 0
            # v_std_min = 0
            v_std_mean = 0

            r = Track.r(pos1, pos2)

            if (not tr1.is_nc()) and (not tr2.is_nc()):
                for k_msn1, k2_msn2 in zip(mean_std_dict_1.items(), mean_std_dict_2.items()):
                    k, msn1 = k_msn1
                    k2, msn2 = k2_msn2

                    assert (k == k2)

                    if k not in Track._chi2_pars:
                        continue

                    s_m, s_s, s_n = msn1
                    t_m, t_s, t_n = msn2

                    if k == 's_linv':
                        if s_n > 0 and t_n > 0:
                            v_mean = (s_m + t_m) * 0.5
                            dv_mean = np.abs(s_m - t_m)
                            v_std_mean = sqrt((s_n * s_s ** 2 + t_n * t_s ** 2) / (s_n + t_n))
                        elif s_n > 0:
                            v_mean = s_m
                            v_std_mean = s_s
                        elif t_n > 0:
                            v_mean = t_m
                            v_std_mean = t_s

                        # v_std_min = min(s_s,t_s)

                    if k == 'vector':
                        s_m = angle_3d(s_m, t_m) / 2
                        t_m = -s_m
                        if s_n > 0 and t_n > 0:
                            s_n = t_n = 2

                    if s_n > 0 and t_n > 0:
                        coef = 1  # 2 if k in ['s_linv', 's_move_v'] else 1
                        n_par += coef

                        n_max = 2 if k in ['ecc', 'w'] else time_range

                        s_n_, t_n_ = min(s_n, n_max), min(t_n, n_max)
                        nll = F_stat_NLL(s_m, t_m, s_s, t_s, s_n_, t_n_) * coef
                        if print_dbg_nll:
                            print(
                                f'{k}:\t\t nll={nll:.3f}, means=({s_m:.3f}, {t_m:.3f}), std=({s_s:.3f}, {t_s:.3f}), num=({s_n_}, {t_n_}, max(n_max)), fact={coef}')
                        nll_sum += nll
            else:
                if tr1.is_nc() and tr2.is_nc():  # both NC
                    nll_sum += 21.58 - 4  # ~18 if nlld==0 (and exists) + bias, prefering nc to nc, to suppress of noise
                else:
                    nll_sum += 11.9  # ~9 if nlld==0 (and exists)
                n_par += 1

                if print_dbg_nll:
                    print('nc nll: ', nll_sum, 'n_nc=', tr1.is_nc() + tr2.is_nc())

                msn1 = mean_std_dict_1['s_linv']
                msn2 = mean_std_dict_2['s_linv']

                s_m, s_s, s_n = msn1
                t_m, t_s, t_n = msn2

                if s_n > 0 and t_n > 0:
                    v_mean = (s_m + t_m) * 0.5
                    dv_mean = np.abs(s_m - t_m)
                    v_std_mean = sqrt((s_n * s_s ** 2 + t_n * t_s ** 2) / (s_n + t_n))
                elif s_n > 0:
                    v_mean = s_m
                    v_std_mean = s_s
                elif t_n > 0:
                    v_mean = t_m
                    v_std_mean = t_s

            r_norm = r  # linear displacement

            dt = abs(t2 - t1)
            dt_norm = max(1, dt)
            v_expected = v_mean
            v_norm = sqrt(r_norm / dt_norm)

            dv = abs(v_expected - v_norm)

            phi = Track.angle(pos1, pos2)
            coef = 1

            if abs(pos2[0] - pos1[0]) < cfgm.MAX_JUMP_DX:
                coef = 1 - np.exp(-abs(np.pi - phi) / (np.pi / 8))  # allow big jumps in direction of -x
                # if nlld*coef<16:
                #    print(coef, phi, pos2[0]-pos1[0], nll, nll*coef)
            if print_dbg_nll:
                print(f'dx={pos2[0] - pos1[0]:.1f}, dy={pos2[1] - pos1[1]:.1f}, phi^={(np.pi - phi) / np.pi * 180:.1f}')

            no_jump_max_dv = sqrt(max(cfgm.DT / dt_norm, cfgm.NO_JUMP_DR))  # 2um/tf
            dv = dv if dv < no_jump_max_dv else (v_std_mean + (dv - v_std_mean) * coef)

            sdv2 = dv ** 2

            # kappa = 5/(5 + dt)
            # sigma_v2 = kappa * v_std_mean**2 + (1-kappa) * v_std_min**2

            sigma_v2 = v_std_mean ** 2 * max(0.1, 1 / max(1, dt_norm - 1))  # s^2(n_tf) = s^2(1_tf)/n_tf, 0.1 s minimum
            sigma_v2 = max(sigma_v2, 1e-6)  # prevent 0-division

            nlld = sdv2 / sigma_v2  # max(v_std_min**2+10/max(1, dt), 1e-6)
            # nlld *= coef

            if nlld > 20:  # ~ 4.5 sigma -> disable conn
                nlld = 16 * 20  # big value, enough when corrected for ndf
            # else:
            #    if r>150:
            #        print('\n r=', r)
            #        print_dbg_nll = True

            if print_dbg_nll:
                print(
                    f'distance: dt={dt}, v_norm={v_norm}, v_expected={v_expected}, dv_extra={dv}, njmax={no_jump_max_dv}')
                print(
                    f'distance: sdv2={sdv2}, v_std_mean2={v_std_mean ** 2}, sigma_v2={sigma_v2}, nlld_raw={sdv2 / sigma_v2}')
                print(f'distance: nll={nlld:.3f}, coef={coef:.3f}')

            if dt <= cfgm.TRACK_DIST_CHI2_MAX_DT and not disable_dist:
                # disabled distance chi2
                n_par += 1
                nll_sum += nlld

            # not usefull when flow is confirmed
            # nll = (dt / 15)**2
            # n_par += 1
            # nll_sum += nll

            # fuducial effect on connection already taken into account upon graph building
            # nll_sum += Track.get_fiducial_NLL(pos1)
            # nll_sum += Track.get_fiducial_NLL(pos2)

        if print_dbg_nll:
            print('total ndf:', nll_sum, n_par)
        nll = chi2_ndf_1(nll_sum, n_par) if n_par > 0 else 16.  # propagate from solver?
        if print_dbg_nll:
            print('total:', nll)

        return nll

    def get_last_time(self):
        last_times = [s.get_last_time() for s in self.segments]
        return np.max(last_times)

    def get_first_time(self):
        first_times = [s.get_first_time() for s in self.segments]
        return np.min(first_times)

    def time_on_track(self, t):
        return self.get_first_time() <= t <= self.get_last_time()

    def get_segmment_intervals(self):
        return [(s.get_first_time(), s.get_last_time()) for s in self.segments]

    def time_within_segment(self, t):
        intervals = self.get_segmment_intervals()
        for t1, t2 in intervals:
            if t1 <= t <= t2:
                return True
        return False

    def get_closest_intervals(self, t):
        prev = curr = next = None

        intervals = self.get_segmment_intervals()
        for t1, t2 in intervals:
            if t1 <= t <= t2:
                curr = (t1, t2)
                return None, curr, None
            else:
                if t2 < t and (prev is None or prev[1] < t2):
                    prev = (t1, t2)
                if t1 > t and (next is None or next[0] > t1):
                    next = (t1, t2)

        return prev, curr, next

    def get_num_nodes(self):
        n = 0
        for s in self.segments:
            n += s.get_num_nodes()
        # n = self.aggregate_nodes(lambda n,t,arg: arg+1, 0)
        return n

    def clear(self):
        self.segments.clear()
        self.updated = True

    def was_merged(self):
        return self.merged_into is not None

    def contained_in_fiducial_volume(self):
        def node_out_of_fid(n, t, arg):
            return arg or (not Track.in_fiducial_volume(n.r))

        any_node_out = self.aggregate_nodes(node_out_of_fid, False)
        self.in_fid_vol = not any_node_out
        return self.in_fid_vol

    def info(self):
        inf = [seg.info for seg in self.segments]
        return inf


class Xing:
    def __init__(self):  # , prob_map
        # self.children = []
        # self.parent = None

        # same track cam join and leave again and again,
        # once crossings are partially resolved
        self.in_tracks_idxtime = set()
        self.out_tracks_idxtime = set()

        # self.prob_map = prob_map

    def exclude(self, track_idx_time):
        self.in_tracks_idxtime.remove(track_idx_time)
        self.out_tracks_idxtime.remove(track_idx_time)

    def exclude_track(self, track_idx):
        excl = [track_idx_time for track_idx_time in self.in_tracks_idxtime if track_idx_time[0] == track_idx]
        for e in excl:
            self.in_tracks_idxtime.remove(e)

        excl = [track_idx_time for track_idx_time in self.out_tracks_idxtime if track_idx_time[0] == track_idx]
        for e in excl:
            self.out_tracks_idxtime.remove(e)

    def add_in(self, track_idx, time_in):
        self.in_tracks_idxtime.add((track_idx, time_in))

    def add_out(self, track_idx, time_out):
        self.out_tracks_idxtime.add((track_idx, time_out))

    def full(self):
        return len(self.in_tracks_idxtime) == len(self.out_tracks_idxtime)

    def exclude_many(self, tracks_idx_time):
        excl = set(tracks_idx_time)
        self.in_tracks_idxtime = self.in_tracks_idxtime.difference(excl)
        self.out_tracks_idxtime = self.out_tracks_idxtime.difference(excl)

    def merge_tracks_get_conn_pts_pairs(self, tr_idx_tgt, tr_idx_src, prob_map_mtr, chi2):
        # returns list(tuple(tr_idx1, t1, tr_idx2, t2))

        # 1. find all gap intervals (possible connections of two tracks): (t,id,in/out)-tuple
        ins = {}
        outs = {}
        for tr_idx, t in self.in_tracks_idxtime:
            if tr_idx not in ins:
                ins[tr_idx] = []
            ins[tr_idx].append(t)

        for tr_idx, t in self.out_tracks_idxtime:
            if tr_idx not in outs:
                outs[tr_idx] = []
            outs[tr_idx].append(t)

        merged_io = [(t, tr_idx_tgt, 0) for t in (ins.get(tr_idx_tgt, []))]  # in
        merged_io += [(t, tr_idx_src, 0) for t in (ins.get(tr_idx_src, []))]  # in
        merged_io += [(t, tr_idx_tgt, 1) for t in (outs.get(tr_idx_tgt, []))]  # out
        merged_io += [(t, tr_idx_src, 1) for t in (outs.get(tr_idx_src, []))]  # out

        merged_io.sort(key=lambda x: x[0])  # sort by time
        # print(ins)
        # print(outs)
        # print(merged_io)

        intervals = []
        i = 0
        n = len(merged_io)
        while i < n:
            p1 = merged_io[i]
            if p1[2] == 0:  # in, search for out
                if i < n - 1:
                    p2 = merged_io[i + 1]
                    if p2[2] == 1:  # found out, merge
                        i += 1
                        intervals.append((p1[1], p1[0], p2[1], p2[0]))

            i += 1

        # 2. find all ins/outs compatible with tgt track
        other_ins = [tr_idx for tr_idx in ins if tr_idx != tr_idx_tgt and tr_idx != tr_idx_src]
        other_outs = [tr_idx for tr_idx in outs if tr_idx != tr_idx_tgt and tr_idx != tr_idx_src]

        compatible_ins = [tr_idx for tr_idx in other_ins if prob_map_mtr[tr_idx_tgt, tr_idx] != -1]
        compatible_outs = [tr_idx for tr_idx in other_outs if prob_map_mtr[tr_idx_tgt, tr_idx] != -1]

        # 3. exclude intervals containing compatible track times
        connectable_intervals = []
        for i in intervals:
            t1, t2 = i[0], i[2]

            point_in_interval = False

            for tr_idx in compatible_ins:
                if point_in_interval:
                    break
                for t in ins[tr_idx]:
                    if t1 < t < t2:
                        point_in_interval = True
                        break

            for tr_idx in compatible_outs:
                if point_in_interval:
                    break
                for t in outs[tr_idx]:
                    if t1 < t < t2:
                        point_in_interval = True
                        break

            if not point_in_interval:
                connectable_intervals.append((*i, chi2))

        # 4. make tr_idx/t pairs closing left intervals
        # print(intervals)
        # print(connectable_intervals)
        return connectable_intervals

    def update_track_id(self, tr_idx_tgt, tr_idx_src, prob_map_mtr):
        # Returns bool: flag showing if the target track is still attached to connection
        # 1. if has src - replace by tgt

        all_tr = set()

        excl = []
        repl = []
        for tr_idx_t in self.in_tracks_idxtime:
            tr_idx, t = tr_idx_t
            all_tr.add(tr_idx)
            if tr_idx == tr_idx_src:
                excl.append(tr_idx_t)
                repl.append((tr_idx_tgt, t))

        is_dbg = False  # tr_idx_tgt in [0, 865, 2, 1, 3, 4]
        self.in_tracks_idxtime = self.in_tracks_idxtime.difference(excl).union(repl)

        excl = []
        repl = []
        for tr_idx_t in self.out_tracks_idxtime:
            tr_idx, t = tr_idx_t
            all_tr.add(tr_idx)
            if tr_idx == tr_idx_src:
                excl.append(tr_idx_t)
                repl.append((tr_idx_tgt, t))

        self.out_tracks_idxtime = self.out_tracks_idxtime.difference(excl).union(repl)
        if is_dbg:
            print(self.in_tracks_idxtime, self.out_tracks_idxtime)

        # 2. find compatible
        all_tr = list(all_tr.difference([tr_idx_tgt, tr_idx_src]))
        compatible_map = ((prob_map_mtr[tr_idx_tgt, tr_idx] != -1) for tr_idx in all_tr)
        if is_dbg:
            compatible_map = list(compatible_map)
            print([prob_map_mtr[tr_idx_tgt, tr_idx] for tr_idx in all_tr])

        # 3. of not compatible with any in/out - exclude
        new_track_compatible_with_some = any(compatible_map)

        if not new_track_compatible_with_some:
            excl = [tit for tit in self.in_tracks_idxtime if tit[0] == tr_idx_tgt]
            self.in_tracks_idxtime.difference_update(excl)

            excl = [tit for tit in self.out_tracks_idxtime if tit[0] == tr_idx_tgt]
            self.out_tracks_idxtime.difference_update(excl)

        return new_track_compatible_with_some

    def tracks_consistent_in_xing(self, trid_a, trid_b):  # not used
        n_i_a = len([t for idx, t in self.in_tracks_idxtime if idx == trid_a])
        n_o_a = len([t for idx, t in self.out_tracks_idxtime if idx == trid_a])
        n_i_b = len([t for idx, t in self.in_tracks_idxtime if idx == trid_b])
        n_o_b = len([t for idx, t in self.out_tracks_idxtime if idx == trid_b])

        if n_i_a + n_i_b == n_o_a + n_o_b:
            return True

    def exclude_not_belonging(self, prob_map_mtr):
        xing_tracks = {idx for idx, t in self.in_tracks_idxtime}.union({idx for idx, t in self.out_tracks_idxtime})
        xing_tracks = list(xing_tracks)
        if len(xing_tracks) == 0:
            return []

        tracks_compatible_map = {trid: False for trid in xing_tracks}

        # mark tracks compatible with any other ones
        for i, trid_i in enumerate(xing_tracks[:-1]):
            for trid_j in xing_tracks[i + 1:]:
                compatible = prob_map_mtr[trid_i, trid_j] != -1

                # f compatible:
                # additionally: either n_ins

                if compatible:
                    tracks_compatible_map[trid_i] = True
                    tracks_compatible_map[trid_j] = True

        tracks_to_be_removed = {trid for trid, comp in tracks_compatible_map.items() if not comp}
        if len(tracks_to_be_removed) == 0:
            return []

        # remove tracks from xing's IO sets:
        in_excl = {(idx, t) for idx, t in self.in_tracks_idxtime if idx in tracks_to_be_removed}
        out_excl = {(idx, t) for idx, t in self.out_tracks_idxtime if idx in tracks_to_be_removed}
        self.in_tracks_idxtime.difference_update(in_excl)
        self.out_tracks_idxtime.difference_update(out_excl)

        # if xing's io map is left inconsistent - remove track excess by highest
        n_in = len(self.in_tracks_idxtime)
        n_out = len(self.out_tracks_idxtime)
        if n_in != n_out:
            excees_side, shortage_side = self.in_tracks_idxtime, self.out_tracks_idxtime
            if n_in < n_out:
                excees_side, shortage_side = shortage_side, excees_side

            n_exc = abs(n_in - n_out)
            # find n_exc worst elements

            idx_t_nll = []
            for idx_e, t_e in excees_side:
                pair_nlls = [prob_map_mtr[idx_e, idx] for idx, t in shortage_side]
                pair_nlls = [(nll if nll >= 0 else 1000) for nll in pair_nlls]
                lowest_nll = np.min(pair_nlls) if len(pair_nlls) else 1000
                idx_t_nll.append(((idx_e, t_e), lowest_nll))

            idx_t_nll.sort(key=lambda e: e[1])
            print(f'additionally removing tracks with highest nll:', idx_t_nll[-n_exc:])

            excl_extra = {(idx, t) for ((idx, t), nll) in idx_t_nll[-n_exc:]}
            excees_side.difference_update(excl_extra)

            tracks_to_be_removed.update({idx for (idx, t) in excl_extra})

        return list(tracks_to_be_removed)

    def get_track_conns_in_out(self, tr_idx):
        # return all list of in and out connections for given track
        tr_ins = [track_idx_time for track_idx_time in self.in_tracks_idxtime if track_idx_time[0] == tr_idx]
        tr_outs = [track_idx_time for track_idx_time in self.out_tracks_idxtime if track_idx_time[0] == tr_idx]

        return tr_ins, tr_outs


class Xings:
    def __init__(self, tracks: List[Track]):
        self.tracks = tracks
        self.xings = []
        # self.xings_id_at_time = {}  # time-> {track_id->xing_id}
        self.track_xing_dict = {}  # track id -> list of xing ids

        self.get_boundary()

    def get_boundary(self):
        def get_boudary_range(n: Node, t, x0x1y0y1_n: list):
            x, y = n.r[:2]
            if x0x1y0y1_n[-1] == 0:
                x0x1y0y1_n[0] = x0x1y0y1_n[1] = x
                x0x1y0y1_n[2] = x0x1y0y1_n[3] = y
            else:
                x0x1y0y1_n[0] = min(x0x1y0y1_n[0], x)
                x0x1y0y1_n[1] = max(x0x1y0y1_n[1], x)
                x0x1y0y1_n[2] = min(x0x1y0y1_n[2], y)
                x0x1y0y1_n[3] = max(x0x1y0y1_n[3], y)

            x0x1y0y1_n[-1] += 1

            return x0x1y0y1_n

        x0x1y0y1_n = [0.] * 4 + [0]

        # t1 = timer()
        for tr in self.tracks:
            x0x1y0y1_n = tr.aggregate_nodes(get_boudary_range, x0x1y0y1_n)
        # t2 = timer()
        # print('boundary search took %.3fs' % (t2-t1), x0x1y0y1_n)

        x0x1y0y1 = x0x1y0y1_n[:4]
        Track.set_fiducial_boundary(x0x1y0y1)

    def track_get_xing_in_out(self, tr_id, t):
        tr = self.tracks[tr_id]
        prev, curr, next = tr.get_closest_intervals(t)  # intervals

        c_in = c_out = -1
        if curr is not None:
            return c_in, c_out  # here not connected

        tr_xing = self.track_xing_dict.get(tr_id, {})
        if len(tr_xing) == 0:
            return c_in, c_out

        if prev is not None:  # track existed before t, thus was joined somewhere
            in_xing_trid_t = [(c_id, self.xings[c_id].in_tracks_idxtime) for c_id in tr_xing]
            in_xing_t = [(c_id, tim) for c_id, c in in_xing_trid_t for trid, tim in c if trid == tr_id]
            if len(in_xing_t) != 0:
                track_in_xing_t = np.array([(c_id, tim)
                                            for c_id, tim in in_xing_t
                                            if (tim <= t and tim >= prev[1])])  # conn after last seg, before t
                if len(track_in_xing_t) != 0:
                    idx = track_in_xing_t[:, 1].argmax()  # latest in time, but not more than t
                    c_in = track_in_xing_t[idx, 0]

        if next is not None:  # track exists after t, thus goes out of somewhere
            out_xing_trid_t = [(c_id, self.xings[c_id].out_tracks_idxtime) for c_id in tr_xing]
            out_xing_t = [(c_id, tim) for c_id, c in out_xing_trid_t for trid, tim in c if trid == tr_id]

            if len(out_xing_t) != 0:
                track_out_xing_t = np.array([(c_id, tim)
                                             for c_id, tim in out_xing_t
                                             if (tim >= t and tim <= next[0])])  # conn before next seg, after t
                if len(track_out_xing_t) != 0:
                    idx = track_out_xing_t[:, 1].argmin()  # earliest in time, but not less than t
                    c_out = track_out_xing_t[idx, 0]

        return c_in, c_out

    def remove_track_merge_xings(self, tr_id):
        if tr_id not in self.track_xing_dict:
            return

        track_xings = list(self.track_xing_dict[tr_id])

        n_xing = len(track_xings)

        if n_xing > 0:
            xing_idx1 = track_xings[0]

            if n_xing > 1:
                for xing_idx2 in track_xings[1:]:
                    self.merge_xings(xing_idx1, xing_idx2)

            xing = self.xings[xing_idx1]
            xing.exclude_track(tr_id)

        self.track_xing_dict[tr_id].clear()

    def merge_xings(self, xing_id1, xing_id2):
        c1 = self.xings[xing_id1]  # tgt
        c2 = self.xings[xing_id2]  # src

        all_conns = c2.in_tracks_idxtime.union(c2.out_tracks_idxtime)
        for tr_id, time in all_conns:  # upate track conn dict in c2
            self.track_xing_dict[tr_id].difference_update([xing_id2])
            self.track_xing_dict[tr_id].add(xing_id1)

        c1.in_tracks_idxtime.update(c2.in_tracks_idxtime)  # merge c2->c1
        c1.out_tracks_idxtime.update(c2.out_tracks_idxtime)
        c2.in_tracks_idxtime.clear()  # clear everythging from c2
        c2.out_tracks_idxtime.clear()

    def add_full_xing(self, xing):
        """
        Params:
            xing ([ins:set(idx,t), outs:set(idx,t)])
        """
        c = Xing()
        xing_idx = len(self.xings)

        for idx, t in xing[0]:  # joining tracks
            c.add_in(idx, t)

            if idx not in self.track_xing_dict:
                self.track_xing_dict[idx] = set()
            self.track_xing_dict[idx].add(xing_idx)

        for idx, t in xing[1]:  # leaving tracks
            c.add_out(idx, t)

            if idx not in self.track_xing_dict:
                self.track_xing_dict[idx] = set()
            self.track_xing_dict[idx].add(xing_idx)

        self.xings.append(c)

    def add_xing_in(self, tr1_id, tr2_id, t):  # if t==-1 t = tr1.last_time
        tr1 = self.tracks[tr1_id]
        tr2 = self.tracks[tr2_id]

        tr1_xing_id = -1
        if tr1_id in self.track_xing_dict:
            tr1_xing_id, _ = self.track_get_xing_in_out(tr1_id, t)

        tr2_xing_id = -1
        if tr2_id in self.track_xing_dict:
            tr2_xing_id, _ = self.track_get_xing_in_out(tr2_id, t)

        if tr1_xing_id == -1 and tr2_xing_id == -1:  # new xing
            c = Xing()
            c.add_in(tr1_id, t)
            c.add_in(tr2_id, t)
            xing_idx = len(self.xings)
            self.xings.append(c)

            if tr1_id not in self.track_xing_dict:
                self.track_xing_dict[tr1_id] = set()
            self.track_xing_dict[tr1_id].add(xing_idx)

            if tr2_id not in self.track_xing_dict:
                self.track_xing_dict[tr2_id] = set()
            self.track_xing_dict[tr2_id].add(xing_idx)

        elif tr1_xing_id != -1 and tr2_xing_id != -1:  # both are already in xing: merge
            if tr1_xing_id != tr2_xing_id:
                self.merge_xings(tr1_xing_id, tr2_xing_id)

        elif tr1_xing_id != -1 or tr2_xing_id != -1:  # only one already in xing: add to that one
            tr_to_add = tr1 if tr1_xing_id == -1 else tr2  # will be added
            tr_in_xing = tr2 if tr1_xing_id == -1 else tr1  # already in a xing
            tr_to_add_id = tr1_id if tr1_xing_id == -1 else tr2_id
            tr_in_xing_id = tr2_id if tr1_xing_id == -1 else tr1_id
            tgt_xing_id = tr1_xing_id if tr1_xing_id != -1 else tr2_xing_id

            c = self.xings[tgt_xing_id]
            c.add_in(tr_to_add_id, t)

            if tr_to_add_id not in self.track_xing_dict:
                self.track_xing_dict[tr_to_add_id] = set()
            self.track_xing_dict[tr_to_add_id].add(tgt_xing_id)

    def add_xing_out(self, tr1_id, tr2_id, t):  # if t==-1 t = tr1.last_time
        tr1 = self.tracks[tr1_id]
        tr2 = self.tracks[tr2_id]

        tr1_xing_id = -1
        if tr1_id in self.track_xing_dict:
            _, tr1_xing_id = self.track_get_xing_in_out(tr1_id, t)

        tr2_xing_id = -1
        if tr2_id in self.track_xing_dict:
            _, tr2_xing_id = self.track_get_xing_in_out(tr2_id, t)

        if tr1_xing_id == -1 and tr2_xing_id == -1:  # new xing
            c = Xing()
            c.add_out(tr1_id, t)
            c.add_out(tr2_id, t)
            xing_idx = len(self.xings)
            self.xings.append(c)

            if tr1_id not in self.track_xing_dict:
                self.track_xing_dict[tr1_id] = set()
            self.track_xing_dict[tr1_id].add(xing_idx)

            if tr2_id not in self.track_xing_dict:
                self.track_xing_dict[tr2_id] = set()
            self.track_xing_dict[tr2_id].add(xing_idx)

        elif tr1_xing_id != -1 and tr2_xing_id != -1:  # only one already in xing: add to that one
            if tr1_xing_id != tr2_xing_id:
                self.merge_xings(tr1_xing_id, tr2_xing_id)

        elif tr1_xing_id != -1 or tr2_xing_id != -1:  # only one already in xing: add to that one
            tr_to_add = tr1 if tr1_xing_id == -1 else tr2  # will be added
            tr_in_xing = tr2 if tr1_xing_id == -1 else tr1  # already in a xing
            tr_to_add_id = tr1_id if tr1_xing_id == -1 else tr2_id
            tr_in_xing_id = tr2_id if tr1_xing_id == -1 else tr1_id
            tgt_xing_id = tr1_xing_id if tr1_xing_id != -1 else tr2_xing_id

            c = self.xings[tgt_xing_id]
            c.add_out(tr_to_add_id, t)

            if tr_to_add_id not in self.track_xing_dict:
                self.track_xing_dict[tr_to_add_id] = set()
            self.track_xing_dict[tr_to_add_id].add(tgt_xing_id)

    def add_xing_out_to_in(self, tr_in_id, tr_out_id, t):  # if t==-1 t = tr1.last_time
        tr_in = self.tracks[tr_in_id]
        tr_out = self.tracks[tr_out_id]

        tr_in_xing_id = -1
        if tr_in_id in self.track_xing_dict:
            tr_in_xing_id, _ = self.track_get_xing_in_out(tr_in_id, t)

        tr_out_xing_id = -1
        if tr_out_id in self.track_xing_dict:
            _, tr_out_xing_id = self.track_get_xing_in_out(tr_out_xing_id, t)

        # print(tr_in_xing_id,tr_out_xing_id)

        if tr_in_xing_id == -1 and tr_out_xing_id == -1:  # new xing
            c = Xing()
            c.add_in(tr_in_id, t)
            c.add_out(tr_out_id, t)
            xing_idx = len(self.xings)
            self.xings.append(c)

            if tr_in_id not in self.track_xing_dict:
                self.track_xing_dict[tr_in_id] = set()
            self.track_xing_dict[tr_in_id].add(xing_idx)

            if tr_out_id not in self.track_xing_dict:
                self.track_xing_dict[tr_out_id] = set()
            self.track_xing_dict[tr_out_id].add(xing_idx)

        elif tr_out_xing_id != -1 and tr_in_xing_id != -1:  # both already in crossing: merge
            if tr_out_xing_id != tr_in_xing_id:
                self.merge_xings(tr_out_xing_id, tr_in_xing_id)

        elif tr_in_xing_id != -1:  # in track in crossing, out not
            c = self.xings[tr_in_xing_id]
            c.add_out(tr_out_id, t)

            if tr_out_id not in self.track_xing_dict:
                self.track_xing_dict[tr_out_id] = set()
            self.track_xing_dict[tr_out_id].add(tr_in_xing_id)

        elif tr_out_xing_id != -1:  # out track in crossing, out not
            c = self.xings[tr_out_xing_id]
            c.add_in(tr_in_id, t)

            if tr_in_id not in self.track_xing_dict:
                self.track_xing_dict[tr_in_id] = set()
            self.track_xing_dict[tr_in_id].add(tr_out_xing_id)

    def merge_tracks_global(self, tr_idx_tgt, tr_idx_src, prob_map_mtr, chi2):

        xings_of_tracks_src = self.track_xing_dict.get(tr_idx_src, set())
        xings_of_tracks_tgt = self.track_xing_dict.get(tr_idx_tgt, set())
        xings_of_tracks = xings_of_tracks_src.union(xings_of_tracks_tgt)

        # 7. curr conn: remove this two ends
        # 8. all conns: (merged,t*) replace by (target,t*)
        for idx in xings_of_tracks:
            xing = self.xings[idx]

            still_connected = xing.update_track_id(tr_idx_tgt, tr_idx_src, prob_map_mtr)

            if tr_idx_tgt not in self.track_xing_dict:
                self.track_xing_dict[tr_idx_tgt] = set()

            if still_connected:
                self.track_xing_dict[tr_idx_tgt].add(idx)
            else:
                self.track_xing_dict[tr_idx_tgt].difference_update([idx])

        if tr_idx_src in self.track_xing_dict:
            self.track_xing_dict[tr_idx_src].clear()

        # 4. fill crossing for drawing table
        return []

    def merge_tracks(self, xing_idx, tr_idx_tgt, tr_idx_src, prob_map_mtr, chi2):
        """ returns list(tuple(tr_idx1, t1, tr_idx2, t2))"""

        if xing_idx == -1:
            print('some global merging happened')
            return self.merge_tracks_global(tr_idx_tgt, tr_idx_src, prob_map_mtr, chi2)

        xing = self.xings[xing_idx]

        conns = xing.merge_tracks_get_conn_pts_pairs(tr_idx_tgt, tr_idx_src, prob_map_mtr, chi2)

        dbg_prnt = False  # tr_idx_tgt==83
        xing_ids_src = self.track_xing_dict[tr_idx_src]
        xing_ids_tgt = self.track_xing_dict[tr_idx_tgt]
        xings_of_tracks = xing_ids_src.union(xing_ids_tgt)

        # 7. curr conn: remove this two ends
        # 8. all conns: (merged,t*) replace by (target,t*)
        for idx in xings_of_tracks:
            xing = self.xings[idx]

            still_connected = xing.update_track_id(tr_idx_tgt, tr_idx_src, prob_map_mtr)
            if dbg_prnt:
                print(xing_idx, idx, still_connected, xing_ids_src, xing_ids_tgt, tr_idx_tgt, tr_idx_src)

            if still_connected:
                self.track_xing_dict[tr_idx_tgt].add(idx)
            else:
                self.track_xing_dict[tr_idx_tgt].difference_update([idx])

            if dbg_prnt:
                print(xing.in_tracks_idxtime, xing.out_tracks_idxtime)
            removed_tracks = xing.exclude_not_belonging(prob_map_mtr)

            if dbg_prnt:
                if len(removed_tracks):
                    print(f'extra removed from xing {idx}: {removed_tracks}')
                print(xing.in_tracks_idxtime, xing.out_tracks_idxtime)

            for tr_idx in removed_tracks:
                self.track_xing_dict[tr_idx].difference_update([idx])

        self.track_xing_dict[tr_idx_src].clear()

        # 4. fill crossing for drawing table
        return conns

    def track_has_start_end_xings(self, tr_idx):
        ins = []
        outs = []

        xings = self.track_xing_dict.get(tr_idx, [])

        for xing_idx in xings:
            lins, louts = self.xings[xing_idx].get_track_conns_in_out(tr_idx)
            ins.extend(lins)
            outs.extend(louts)

            if len(ins) > 0 and len(outs) > 0:
                return True, True

        return len(outs) > 0, len(ins) > 0  # outs: track start connected, ins: track end connected

    def tracks_in_crossing(self, tr_idx1, tr_idx2):
        xings_idxs_1 = self.track_xing_dict.get(tr_idx1, set())
        xings_idxs_2 = self.track_xing_dict.get(tr_idx2, set())
        return len(xings_idxs_1.intersection(xings_idxs_2)) > 0


class ProbMap:
    def __init__(self, tracks: List[Track]):
        n = len(tracks)
        self.tracks = tracks
        self.mtr = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            self.mtr[i, i] = -1  ## no self crossings

        self.max_gap = 0
        self.max_overlap = 0

    def _is_incompatible(self, ij):
        i, j = ij
        # if i==0 and j<10:
        #    print(len(self.tracks))
        overlaps = self.tracks[i].overlaps_with(track=self.tracks[j],
                                                max_overlap=self.max_overlap,
                                                max_gap=self.max_gap)
        # if overlaps, than they are incompatible
        conn_impossible = overlaps or (not self.tracks[i].connection_possible(track=self.tracks[j]))
        return (overlaps or conn_impossible)

    def fill_incompatible_MT_draft(self):
        # n = len(self.tracks)
        # n_thr = max(1, multiprocessing.cpu_count() - 1)
        # list_pairs = []
        #
        # ts = []
        #
        # ts.append(time.time())
        # for j in range(n - 1):
        #     for i in range(j + 1, n):  # upper right triangle
        #         list_pairs.append((i, j))
        # print(len(list_pairs))
        # ts.append(time.time())
        # chunksize = ceil(len(list_pairs) / n_thr)
        # with ProcessPoolExecutor(max_workers=n_thr) as executor:
        #     ts.append(time.time())
        #     incompatible = executor.map(self._is_incompatible, list_pairs, chunksize=chunksize)
        #     for inc, (i, j) in zip(incompatible, list_pairs):
        #         if incompatible:
        #             self.mtr[i, j] = self.mtr[j, i] = -1
        #
        #     ts.append(time.time())
        #
        # ts.append(time.time())
        # ts = np.asarray(ts)
        # print(ts[1:] - ts[:-1])
        pass

    def fill_incompatible(self, xings: Xings):
        n = len(self.tracks)
        for j in range(n - 1):
            for i in range(j + 1, n):  # upper right triangle
                # print(i,j)
                tr_i, tr_j = [self.tracks[k] for k in [i, j]]
                overlaps = tr_i.overlaps_with(track=tr_j,
                                              max_overlap=self.max_overlap,
                                              max_gap=self.max_gap)
                # if overlaps, than they are incompatible

                tracks_in_crossing = xings.tracks_in_crossing(i, j)
                conn_impossible = overlaps or (not (tr_i.connection_possible(track=tr_j) or tracks_in_crossing))
                if overlaps or conn_impossible:
                    self.mtr[i, j] = self.mtr[j, i] = -1

    def fill_prob(self):
        n = len(self.tracks)
        for j in range(n - 1):
            t_j: Track = self.tracks[j]
            for i in range(j + 1, n):  # upper right triangle
                if self.mtr[i, j] != -1:
                    t_i: Track = self.tracks[i]
                    chi2 = t_i.get_connection_chi2(t_j)
                    self.mtr[i, j] = self.mtr[j, i] = chi2

    def reset_prob_merging(self, tgt_tr_idx, src_tr_idx):
        # probability of tgt->i set to 0 where src->i is not -1
        n = len(self.tracks)
        for i in range(n):  # upper right triangle
            if self.mtr[src_tr_idx, i] != -1:
                self.mtr[tgt_tr_idx, i] = self.mtr[i, tgt_tr_idx] = 0

    def update_prob(self, track_idx):
        n = len(self.tracks)
        j = track_idx
        t_j: Track = self.tracks[j]
        assert (not t_j.was_merged())

        dbg_prnt = False

        for i in range(n):  # upper right triangle
            t_i: Track = self.tracks[i]
            if t_i.was_merged():
                continue
            if self.mtr[i, j] != -1:

                overlaps = t_i.overlaps_with(track=t_j,
                                             max_overlap=self.max_overlap,
                                             max_gap=self.max_gap)
                if overlaps:
                    chi2 = -1
                else:
                    chi2 = t_i.get_connection_chi2(t_j)

                self.mtr[i, j] = self.mtr[j, i] = chi2

    def remove_track(self, track_idx):
        self.mtr[:, track_idx] = self.mtr[track_idx] = -1


class Solver:
    max_chi2_unique = 6  # 9?
    max_chi2 = 9

    max_chi2_significant = 1
    min_chi2_nonsignificant = 4  # 3? 4?

    draw_each_iteration = True
    print_each_iteration = False

    min_track_len = 6

    def __init__(self, crossings: Xings):
        self.tracks: List[Track] = crossings.tracks

        self.tracks_orig = copy.deepcopy(self.tracks)
        self.pm = ProbMap(self.tracks)
        self.crossings = crossings
        self.resolved_conns = []  # (tr_idx1, t1, tr_idx2, t2, chi2)

        self.LSE_hist = []
        self.merge_n1_n2_hist = []

        self.orig_tr_idx_map = {}
        self.orig_mu_sgm = None

    def active_track_id(self, track_id):
        d_uid = self.tracks[0].uid
        while True:
            t = self.tracks[track_id]
            if not t.merged_into:
                return track_id
            track_id = t.merged_into - d_uid

    def fill_priors(self, min_len):
        t1 = time.time()
        priors = Track.priors.copy()

        prior_arr = {}
        for key in priors:
            prior_arr[key] = [[], []]  # arrays of mu and sigma

        for tr in self.tracks:
            if tr.get_num_nodes() >= min_len:
                res = tr.get_mean_std(bayesian_estimators=False)

                for key, msn in res.items():
                    prior = prior_arr[key]
                    m, s, n = msn
                    if n > min_len / 2:
                        prior[0].append(m)
                        prior[1].append(s)
        # print(prior_arr)

        for key, ms_ss in prior_arr.items():
            ms, ss = ms_ss
            if len(ms) <= 1:
                continue
            mu_pop = np.mean(ms, axis=0)
            if iterable(mu_pop):
                mu_pop = list(mu_pop)

            sigma_pop = np.std(ms, ddof=1) * 1.75
            sigma2_pop = sigma_pop ** 2

            sigmas2_inst = np.array(ss) ** 2
            sigma2_inst = max(np.mean(sigmas2_inst) * 1.75, np.percentile(sigmas2_inst, 85))
            # print(np.mean(sigmas2_inst), np.percentile(sigmas2_inst, 75), np.percentile(sigmas2_inst, 85))

            sigma2_pop = sigma2_inst if sigma2_pop == 0 else sigma2_pop

            Track.priors[key] = [mu_pop, sigma2_pop, sigma2_inst]
            #  print(key, [mu_pop, sigma2_pop, sigma2_inst])

        t2 = time.time()
        print(f'filling priors: {(t2 - t1):.2f}s')

    def init_pm(self, pm_mtr=None):
        if pm_mtr is not None:
            self.pm.mtr = copy.deepcopy(pm_mtr)
            return
        print('init_pm:fill_incompatible...')
        ts = []

        ts.append(time.time())

        self.pm.fill_incompatible(self.crossings)
        print('init_pm:fill_prob...')

        ts.append(time.time())
        self.pm.fill_prob()

        ts.append(time.time())
        ts = np.array(ts)
        print('init_pm:done.', ts[1:] - ts[:-1])
        # self.clip_pm()
        pass

    def clip_pm(self):
        self.pm.mtr[self.pm.mtr > Solver.max_chi2] = -1

    def resolve_best_unique(self):  # this function makes no sence?
        n = len(self.tracks)
        mtr = self.pm.mtr
        conn_map = (self.pm.mtr != -1)
        n_conn = conn_map.sum(axis=1)

        single_conn_map = n_conn == 1

        if single_conn_map.sum() == 0:  # no unique connections available.
            return False

        unique_tr_idx = np.arange(n)[single_conn_map]  # indeces of tracks with unique connection
        unique_conn_idx = [np.array([np.argmax(conns)]) for conns in mtr[unique_tr_idx]]  # connection track idx
        #                               ^ argmax since it's unique, and '-1's are less

        unique_conns_chi2 = mtr[unique_tr_idx, unique_conn_idx]
        best_option = unique_conns_chi2.argmax()

        best_chi2 = unique_conns_chi2[best_option]

        if best_chi2 > Solver.max_chi2_unique:  # there are unique, but all not good enough
            return False

        src_idx = unique_tr_idx
        tgt_idx = unique_conn_idx
        src = self.tracks[src_idx]
        tgt = self.tracks[tgt_idx]

        tgt.merge_track(src)
        mtr[src_idx] = mtr[:, src_idx] = -1
        self.pm.update_prob(tgt_idx)

        return True

    def resolve_unique(self, draw_fn=None):
        while True:
            if not self.resolve_best_unique():
                break
            if draw_fn and Solver.draw_each_iteration:
                draw_fn(self)

    def remove_short_tracks(self):
        for tr_idx, tr in enumerate(self.tracks):
            if (tr.get_num_nodes() < Solver.min_track_len) and (not tr.is_nc()):
                print(tr_idx, tr.segments, tr.segments[0].nodes, tr.segments[-1].nodes)
                self.crossings.remove_track_merge_xings(tr_idx)
                self.pm.remove_track(tr_idx)

                tr.clear()

    def get_LSE(self):
        lse = 0
        for t in self.tracks:
            if t.merged_into is None:
                # m,s,n = t.get_mean_std()

                # v = s**2
                # se = v*(n-1) if n>2 else 0

                se = 0  # random stuff
                lse += se
        return lse

    def add_lse_history(self):
        self.LSE_hist.append(self.get_LSE())

    def merge_tracks(self, coon_id, tr_idx_tgt, tr_idx_src):
        # 1. connect
        t_tgt: Track = self.tracks[tr_idx_tgt]
        t_src: Track = self.tracks[tr_idx_src]

        n1, n2 = t_tgt.get_num_nodes(), t_src.get_num_nodes()
        n1, n2 = (n1, n2) if n1 < n2 else (n2, n1)
        self.merge_n1_n2_hist.append([n1, n2])

        self.add_lse_history()

        t_tgt.merge_track(t_src)

        # 2. set chi for merged as -1
        dbg_prnt = False  # tr_idx_src==962 or tr_idx_tgt==962 or 119 in self.crossings.track_xing_dict[tr_idx_src] or 119 in self.crossings.track_xing_dict[tr_idx_tgt]
        chi2 = self.pm.mtr[tr_idx_src, tr_idx_tgt]

        if dbg_prnt:
            print(f'merge_tracks {tr_idx_tgt}<-{tr_idx_src}')
            # print(f'{self.pm.mtr[962, 881]}, \
            # {self.tracks[962].get_connection_chi2(self.tracks[881], print_dbg_nll=True)}, \
            # {self.tracks[962].overlaps_with(self.tracks[881])}, \
            #  ')

        self.pm.reset_prob_merging(tr_idx_tgt, tr_idx_src)
        if dbg_prnt:
            print(f'{self.pm.mtr[962, 881]}, rmp')

        self.pm.remove_track(tr_idx_src)

        # 3. set chi for target (where track was merged into)
        self.pm.update_prob(tr_idx_tgt)
        if dbg_prnt:
            print(f'{self.pm.mtr[962, 881]} up')

        # 4. update crossings
        self.resolved_conns += self.crossings.merge_tracks(coon_id, tr_idx_tgt, tr_idx_src, self.pm.mtr, chi2)

    def resolve_best_unique_in_xings(self):
        # 1. collect all coon_id, from, to, n_from, n_to,chi2 where for at least one of (from,to) this is the only conn
        conns = self.crossings.xings

        coll = set()

        for conn_id, conn in enumerate(conns):
            in_tr = conn.in_tracks_idxtime
            out_tr = conn.out_tracks_idxtime

            all_ins = np.array(list({it[0] for it in in_tr}))
            all_outs = np.array(list({it[0] for it in out_tr}))

            if len(all_ins) == 0 or len(all_outs) == 0:
                continue

            mtr = self.pm.mtr[all_ins][:, all_outs]
            mtr = mtr.copy()
            mtr[mtr > Solver.max_chi2] = -1  # exclude all too bad
            mtr_map = mtr != -1

            # find ins with unique conn:
            in_n_con = mtr_map.sum(axis=1)
            out_n_con = mtr_map.sum(axis=0)

            in_single_con_map = in_n_con == 1
            out_single_con_map = out_n_con == 1

            in_single_tr_idx = all_ins[in_single_con_map]
            out_single_tr_idx = all_outs[out_single_con_map]

            in_single_pair_tr_idx = all_outs[mtr[in_single_con_map].argmax(axis=1)] if len(in_single_con_map) else []
            out_single_pair_tr_idx = all_ins[mtr[:, out_single_con_map].argmax(axis=0)] if len(
                out_single_con_map) else []

            all_single_starts = np.concatenate((in_single_tr_idx, out_single_tr_idx))
            all_single_pairs = np.concatenate((in_single_pair_tr_idx, out_single_pair_tr_idx))

            for t_idx_in, t_idx_out in zip(all_single_starts, all_single_pairs):
                if t_idx_in > t_idx_out:
                    t_idx_in, t_idx_out = t_idx_out, t_idx_in

                t_in = self.tracks[t_idx_in]
                t_out = self.tracks[t_idx_out]
                n_in = t_in.get_num_nodes()
                n_out = t_out.get_num_nodes()

                chi2 = self.pm.mtr[t_idx_in, t_idx_out]
                if chi2 > Solver.max_chi2_unique:
                    continue

                item = (conn_id, t_idx_in, t_idx_out, n_in, n_out, chi2)
                coll.add(item)

        coll = list(coll)
        # 2. sort by chi2 + f(n), n = min(n_from, n_to), f(n) eg= 5/n
        n_min = [max(min(c[3], c[4]), 1) for c in coll]
        score = [c[5] + 5 / n for c, n in zip(coll, n_min)]

        if len(score) == 0:
            return False

        idxs = np.argsort(score)

        # 3. find best, for it:
        for idx in idxs:
            conn = coll[idx]
            coon_id, tr_idx_1, tr_idx_2, n_1, n_2, chi2 = conn
            if chi2 < Solver.max_chi2_unique:
                # 4. merge_tracks(), ret true

                # test:
                if Solver.print_each_iteration:
                    print('UC: coon_id: %d, tr_idx_1: %d, tr_idx_2: %d, n_1: %d, n_2: %d, chi2: %.3f' %
                          (coon_id, tr_idx_1, tr_idx_2, n_1, n_2, chi2))

                self.merge_tracks(coon_id, tr_idx_1, tr_idx_2)
                return True

        return False

    def resolve_unique_in_xings(self, draw_fn):
        connected_some = False
        while True:
            if not self.resolve_best_unique_in_xings():
                break

            connected_some = True
            if draw_fn and Solver.draw_each_iteration:
                draw_fn(self)

        if Solver.print_each_iteration:
            if connected_some:
                print('resolve_unique_in_xings result:')
                print(self.resolved_conns)
                print('________________________________\n')
            else:
                print('resolve_unique_in_xings: nothing new')

        return connected_some

    def resolve_best_significantly_unique_in_xings(self):
        # 1. collect all coon_id, from, to, n_from, n_to,chi2 where for both of
        # (from,to) this is the best connection, and for at leat one
        # of (from,to) this is a significantly unique connection, i.e.
        # this chi2 <= Solver.max_chi2_significant, while
        # all other chi2 >= Solver.min_chi2_nonsignificant

        xings = self.crossings.xings

        coll = set()

        for xing_id, xing in enumerate(xings):
            in_tr = xing.in_tracks_idxtime
            out_tr = xing.out_tracks_idxtime

            all_ins = np.array(list({it[0] for it in in_tr}))
            all_outs = np.array(list({it[0] for it in out_tr}))

            n_ins = len(all_ins)
            n_outs = len(all_outs)
            more_ins = n_ins > 1
            more_outs = n_outs > 1

            if len(all_ins) == 0 or len(all_outs) == 0:
                continue

            mtr = self.pm.mtr[all_ins][:, all_outs]
            mtr = mtr.copy()
            mtr[mtr == -1] = Solver.max_chi2  # prepare for sorting

            # for each track find indexes of paris, sorted by quality
            pairs_to_ins = [np.argsort(ins_conn) for ins_conn in mtr]
            pairs_to_outs = [np.argsort(outs_conn) for outs_conn in mtr.transpose()]

            # find pairs for which other is best solution
            for in_lidx, in_idx in enumerate(all_ins):  # local index in all_ins and actual track index
                best_out_lidx = pairs_to_ins[in_lidx][0]
                best_in_lidx = pairs_to_outs[best_out_lidx][0]

                if best_in_lidx == in_lidx:  # best is same from both sides
                    # check that pair is significant

                    chi2_best = mtr[best_in_lidx, best_out_lidx]
                    if chi2_best <= Solver.max_chi2_significant:  # one may be significant
                        second_best_pair_for_in_chi2 = mtr[
                            best_in_lidx, pairs_to_ins[best_in_lidx][1]] if more_outs else Solver.max_chi2
                        second_best_pair_for_out_chi2 = mtr[
                            pairs_to_outs[best_out_lidx][1], best_out_lidx] if more_ins else Solver.max_chi2

                        in_is_significant = second_best_pair_for_in_chi2 >= Solver.min_chi2_nonsignificant
                        out_is_significant = second_best_pair_for_out_chi2 >= Solver.min_chi2_nonsignificant

                        if in_is_significant or out_is_significant:
                            t_idx_in = in_idx
                            t_idx_out = all_outs[best_out_lidx]

                            t_in = self.tracks[t_idx_in]
                            t_out = self.tracks[t_idx_out]
                            n_in = t_in.get_num_nodes()
                            n_out = t_out.get_num_nodes()

                            item = (xing_id, t_idx_in, t_idx_out, n_in, n_out, chi2_best)
                            coll.add(item)

        coll = list(coll)

        # 2. sort by chi2 + f(n), n = min(n_from, n_to), f(n) eg= 5/n
        n_min = [max(min(c[3], c[4]), 1) for c in coll]
        score = [c[5] + 5 / n for c, n in zip(coll, n_min)]

        if len(score) == 0:
            return False

        # 3. find best, for it:
        idx = np.argmin(score)

        conn = coll[idx]
        xing_id, tr_idx_1, tr_idx_2, n_1, n_2, chi2 = conn

        # 4. merge_tracks(), ret true
        if Solver.print_each_iteration:
            print('SUC: coon_id: %d, tr_idx_1: %d, tr_idx_2: %d, n_1: %d, n_2: %d, chi2: %.3f' %
                  (xing_id, tr_idx_1, tr_idx_2, n_1, n_2, chi2))

        self.merge_tracks(xing_id, tr_idx_1, tr_idx_2)
        return True

    def resolve_significantly_unique_in_xings(self, draw_fn):
        connected_some = False
        while True:
            if not self.resolve_best_significantly_unique_in_xings():
                break

            connected_some = True
            if draw_fn:
                draw_fn(self)

        if Solver.print_each_iteration:
            print('resolve_unique_in_xings result:')
            print(self.resolved_conns)
            print('________________________________\n')

        return connected_some

    def resolve_best_significantly_unique_all(self):
        # 1. collect all from, to, n_from, n_to, chi2 where for both of
        # (from,to) this is the best connection, and for at leat one
        # of (from,to) this is a significantly unique connection, i.e.
        # this chi2 <= Solver.max_chi2_significant, while
        # all other chi2 >= Solver.min_chi2_nonsignificant

        xings = self.crossings.xings

        coll = set()

        mtr = self.pm.mtr.copy()
        mtr[mtr == -1] = Solver.max_chi2  # prepare for sorting

        n_tr = len(mtr)
        more_ins = more_outs = n_tr > 1
        all_ins = np.arange(n_tr)
        all_outs = all_ins.copy()

        # for each track find indexes of paris, sorted by quality
        pairs_to_ins = [np.argsort(ins_conn) for ins_conn in mtr]
        pairs_to_outs = [np.argsort(outs_conn) for outs_conn in mtr.transpose()]

        # find pairs for which other is best solution
        for in_idx in all_ins:  # local index in all_ins and actual track index
            best_out_idx = pairs_to_ins[in_idx][0]
            best_in_idx = pairs_to_outs[best_out_idx][0]

            if best_in_idx == in_idx:  # best is same from both sides
                # check that pair is significant

                chi2_best = mtr[best_in_idx, best_out_idx]
                if chi2_best <= Solver.max_chi2_significant:  # one may be significant
                    second_best_pair_for_in_chi2 = mtr[
                        best_in_idx, pairs_to_ins[best_in_idx][1]] if more_outs else Solver.max_chi2
                    second_best_pair_for_out_chi2 = mtr[
                        pairs_to_outs[best_out_idx][1], best_out_idx] if more_ins else Solver.max_chi2

                    in_is_significant = second_best_pair_for_in_chi2 >= Solver.min_chi2_nonsignificant
                    out_is_significant = second_best_pair_for_out_chi2 >= Solver.min_chi2_nonsignificant

                    if in_is_significant or out_is_significant:
                        t_idx_in = min(best_in_idx, best_out_idx)
                        t_idx_out = max(best_in_idx, best_out_idx)

                        t_in = self.tracks[t_idx_in]
                        t_out = self.tracks[t_idx_out]
                        n_in = t_in.get_num_nodes()
                        n_out = t_out.get_num_nodes()

                        item = (t_idx_in, t_idx_out, n_in, n_out, chi2_best)
                        coll.add(item)

        coll = list(coll)

        # 2. sort by chi2 + f(n), n = min(n_from, n_to), f(n) eg= 5/n
        n_min = [max(min(c[2], c[3]), 1) for c in coll]
        score = [c[4] + 5 / n for c, n in zip(coll, n_min)]

        if len(score) == 0:
            return False

        # 3. find best, for it:
        idx = np.argmin(score)

        conn = coll[idx]
        tr_idx_1, tr_idx_2, n_1, n_2, chi2 = conn

        # 4. merge_tracks(), ret true
        if Solver.print_each_iteration:
            print('SUA: coon_id: %d, tr_idx_1: %d, tr_idx_2: %d, n_1: %d, n_2: %d, chi2: %.3f' %
                  (tr_idx_1, tr_idx_2, n_1, n_2, chi2))

        self.merge_tracks(-1, tr_idx_1, tr_idx_2)
        return True

    def resolve_significantly_unique_all(self, draw_fn):
        connected_some = False
        while True:
            if not self.resolve_best_significantly_unique_all():
                break

            connected_some = True
            if draw_fn:
                draw_fn(self)

        if Solver.print_each_iteration:
            print('resolve_unique_in_xings result:')
            print(self.resolved_conns)
            print('________________________________\n')

        return connected_some

    def resolve_significantly_unique_in_xings_and_unique_in_xings(self, draw_fn):
        connected_some = False
        j = -1
        while True:
            j += 1
            if j != 0:
                if not self.resolve_best_significantly_unique_in_xings():  # no new
                    break
                connected_some = True
                if draw_fn and Solver.draw_each_iteration:
                    draw_fn(self)

            res = self.resolve_unique_in_xings(draw_fn)
            connected_some = connected_some or res

        while False:  # former code
            connected_some = self.resolve_unique_in_xings(draw_fn)
            while True:
                if not self.resolve_best_significantly_unique_in_xings():
                    break

                connected_some = True
                if draw_fn and Solver.draw_each_iteration:
                    draw_fn(self)

                self.resolve_unique_in_xings(draw_fn)

        if draw_fn and not Solver.draw_each_iteration:
            draw_fn(self)

        if Solver.print_each_iteration:
            print('resolve_significantly_unique_in_xings result:')
            print(self.resolved_conns)
            print('________________________________\n')

        return connected_some

    def resolve_significantly_unique_all_and_significantly_unique_in_xings_and_unique_in_xings(self, draw_fn):
        connected_some = False

        i = -1
        while True:
            i += 1
            print('itr:', i)
            if i != 0:
                if not self.resolve_best_significantly_unique_all():  # no new connection
                    break

                connected_some = True
                if draw_fn and Solver.draw_each_iteration:
                    draw_fn(self)

            j = -1
            while True:
                j += 1

                if j != 0:
                    if not self.resolve_best_significantly_unique_in_xings():  # no new
                        break

                    connected_some = True
                    if draw_fn and Solver.draw_each_iteration:
                        draw_fn(self)

                res = self.resolve_unique_in_xings(draw_fn)
                connected_some = connected_some or res

        if draw_fn and not Solver.draw_each_iteration:
            draw_fn(self)

        if Solver.print_each_iteration:
            print('resolve_significantly_unique_all result:')
            print(self.resolved_conns)
            print('________________________________\n')

        return connected_some

    def get_minflow_connections(self):
        f = FlowSolver()
        self.f = f
        for tr_idx, tr in enumerate(self.tracks):
            if tr.merged_into is not None or tr.get_num_nodes() == 0:
                continue

            has_start_xing, has_end_xing = self.crossings.track_has_start_end_xings(tr_idx)

            if not has_start_xing:
                # print('s', tr_idx)
                f.add_start(tr_idx)

            if not has_end_xing:
                # print('e', tr_idx)
                f.add_end(tr_idx)

        xings = self.crossings.xings
        chi2_mtr = self.pm.mtr
        for xing_idx, xing in enumerate(xings):
            in_tr_t = xing.in_tracks_idxtime
            out_tr_t = xing.out_tracks_idxtime

            for tr_in, t_in in in_tr_t:
                for tr_out, t_out in out_tr_t:
                    tr_in = self.active_track_id(tr_in)
                    tr_out = self.active_track_id(tr_out)

                    # print('xing', tr_in, tr_out)
                    chi2 = chi2_mtr[tr_in, tr_out]
                    chi2 = Solver.max_chi2 * 100 if chi2 < 0 else chi2

                    f.add_weight(xing_idx, tr_in, t_in, tr_out, t_out, chi2)

        if not f.solve():
            return []

        resolved = f.resolved_conns
        return resolved

    def resolve_remaining_minflow(self):
        resolved = self.get_minflow_connections()

        # merge
        for res_conn in resolved:
            conn_id, tr_idx1, t1, tr_idx2, t2, chi2 = res_conn
            tr_idx1 = self.active_track_id(tr_idx1)
            tr_idx2 = self.active_track_id(tr_idx2)
            if tr_idx1 == tr_idx2:
                continue

            self.merge_tracks(conn_id, tr_idx1, tr_idx2)

            self.resolved_conns.append((tr_idx1, t1, tr_idx2, t2, chi2))

    def get_LAP_connections(self):
        lap = LAPSolver(in_to_out=True, explicit_nc=True)
        self.lap = lap

        xings = self.crossings.xings
        chi2_mtr = self.pm.mtr

        for xing_idx, xing in enumerate(xings):
            in_tr_t = xing.in_tracks_idxtime
            out_tr_t = xing.out_tracks_idxtime

            for tr_in, t_in in in_tr_t:
                tr_in = self.active_track_id(tr_in)
                lap.add_conn_node_in(xing_idx, tr_in, t_in)

            for tr_out, t_out in out_tr_t:
                tr_out = self.active_track_id(tr_out)
                lap.add_conn_node_out(xing_idx, tr_out, t_out)

            chi2_flip = lambda chi2: Solver.max_chi2 * 100 if chi2 < 0 else chi2
            chi2_f = lambda xing_idx, tr_idx1, t1, tr_idx2, t2: chi2_flip(chi2_mtr[tr_idx1, tr_idx2])

            lap.set_chi2_fn(chi2_f)

        if not lap.solve():
            return []

        resolved = lap.resolved_conns.copy()
        return resolved

    def get_LAP_connections_individual_xings(self):  # For debug purposes
        xings = self.crossings.xings
        chi2_mtr = self.pm.mtr

        resolved = []

        for xing_idx, xing in enumerate(xings):
            lap = LAPSolver(in_to_out=True, explicit_nc=True)
            self.lap = lap

            in_tr_t = xing.in_tracks_idxtime
            out_tr_t = xing.out_tracks_idxtime

            for tr_in, t_in in in_tr_t:
                tr_in = self.active_track_id(tr_in)
                lap.add_conn_node_in(xing_idx, tr_in, t_in)

            for tr_out, t_out in out_tr_t:
                tr_out = self.active_track_id(tr_out)
                lap.add_conn_node_out(xing_idx, tr_out, t_out)

            chi2_flip = lambda chi2: Solver.max_chi2 * 100 if chi2 < 0 else chi2
            chi2_f = lambda xing_idx, tr_idx1, t1, tr_idx2, t2: chi2_flip(chi2_mtr[tr_idx1, tr_idx2])

            lap.set_chi2_fn(chi2_f)
            if lap.solve():
                resolved += [v for v in lap.resolved_conns.copy()]
            else:
                print(f'crossing {xing_idx} cannot be resolved')

        return resolved

    def resolve_remaining_LAP_best(self, lim=None):
        lim = lim or Solver.max_chi2
        resolved = self.get_LAP_connections()

        # sort to strat merging from best ones
        resolved.sort(key=lambda item: item[-1])
        # merge
        merged_some = False
        for res_conn in resolved:
            conn_id, tr_idx1, t1, tr_idx2, t2, chi2 = res_conn
            tr_idx1 = self.active_track_id(tr_idx1)
            tr_idx2 = self.active_track_id(tr_idx2)
            if tr_idx1 == tr_idx2:
                continue

            chi2 = self.pm.mtr[tr_idx1, tr_idx2]
            if chi2 >= lim or chi2 < 0:  # skip too bad
                continue

            merged_some = True
            self.merge_tracks(conn_id, tr_idx1, tr_idx2)

            self.resolved_conns.append((tr_idx1, t1, tr_idx2, t2, chi2))

        return merged_some

    def resolve_remaining_LAP(self, lim=None):
        lim = lim or Solver.max_chi2
        merged_some = False
        itr = 0
        print(f'resolve_remaining_LAP: lim={lim}')
        while True:
            if not self.resolve_remaining_LAP_best(lim):
                break
            merged_some = True
            if Solver.print_each_iteration:
                print('resolve_remaining_LAP: %d' % itr)
            itr += 1
        return merged_some

    def get_LAP_associations(self):
        lap = LAPSolver(in_to_out=False)
        self.lap = lap

        xings = self.crossings.xings
        chi2_mtr = self.pm.mtr

        for xing_idx, xing in enumerate(xings):
            in_tr_t = xing.in_tracks_idxtime
            out_tr_t = xing.out_tracks_idxtime

            for tr_in, t_in in in_tr_t:
                tr_in = self.active_track_id(tr_in)
                lap.add_conn_node_in(xing_idx, tr_in, t_in)

            for tr_out, t_out in out_tr_t:
                tr_out = self.active_track_id(tr_out)
                lap.add_conn_node_out(xing_idx, tr_out, t_out)

            chi2_f = lambda xing_idx, tr_idx1, t1, tr_idx2, t2: Solver.max_chi2 * 100 if chi2_mtr[
                                                                                             tr_idx1, tr_idx2] < 0 else \
            chi2_mtr[tr_idx1, tr_idx2]

            lap.set_chi2_fn(chi2_f)

        if not lap.solve():
            return []

        resolved = lap.resolved_conns.copy()
        return resolved

    def resolve_remaining_associations_LAP_best(self):
        resolved = self.get_LAP_associations()

        # sort to strat merging from best ones
        resolved.sort(key=lambda item: item[-1])

        # merge
        merged_some = False

        for res_conn in resolved:
            conn_id, tr_idx1, t1, tr_idx2, t2, chi2 = res_conn
            tr_idx1 = self.active_track_id(tr_idx1)
            tr_idx2 = self.active_track_id(tr_idx2)
            if tr_idx1 == tr_idx2:
                continue

            chi2 = self.pm.mtr[tr_idx1, tr_idx2]
            # if chi2 >= Solver.max_chi2 or chi2<0: # skip too bad
            #    continue
            if chi2 < 0:  # all valid allowed
                continue

            merged_some = True
            self.merge_tracks(conn_id, tr_idx1, tr_idx2)

            self.resolved_conns.append((tr_idx1, t1, tr_idx2, t2, chi2))

        return merged_some

    def resolve_remaining_associations_LAP(self):
        merged_some = False
        itr = 0
        print(f'resolve_remaining_associations_LAP')
        while True:
            if not self.resolve_remaining_associations_LAP_best():
                break
            merged_some = True
            if Solver.print_each_iteration:
                print('resolve_remaining_associations_LAP: %d' % itr)
            itr += 1
        return merged_some

    def get_quality_info(self):
        if not self.orig_tr_idx_map:
            return None

        # all connected components: new tracks.
        tracks_segment_efficiency = [[], []]  # (orig_tr_id, eff),  num_seg_own / num_seg_tot
        tracks_node_efficiency = [[], []]  # (orig_tr_id, eff), num_node_own/num_note_tot
        tracks_means = [[], []]  # (orig_tr_id, estimate mean)
        tracks_sigma = [[], []]  # (orig_tr_id, estimate sigma)

        tracks_length = [[], []]  # (orig_tr_id, length of segment)

        corr_nodes = 0  # global efficiency count, correct identified nodes
        all_nodes = 0

        d_uid = self.tracks[0].uid
        for tr in self.tracks:
            if tr.merged_into is not None:
                continue

            m, s, n = tr.get_mean_std()

            if n == 0:
                continue

            uids = tr.merged_uids

            # sort by time:
            orig_idxs = np.array([uid - d_uid for uid in uids])
            tr_segs_first_times = np.array([self.tracks_orig[o_idx].get_first_time() for o_idx in orig_idxs])
            sorted_t_idx = np.argsort(tr_segs_first_times)

            orig_idxs = orig_idxs[sorted_t_idx]
            tr_segs_first_times = tr_segs_first_times[sorted_t_idx]
            tr_segs_n_nodes = [self.tracks_orig[o_idx].get_num_nodes() for o_idx in orig_idxs]

            orig_segs_ids = [self.orig_tr_idx_map.get(o_idx, -1) for o_idx in
                             orig_idxs]  # true track ID, before cutting
            orig_id_n_nodes = {o_id: 0 for o_id in set(orig_segs_ids)}

            main_id = orig_segs_ids[0]
            for o_id, n in zip(orig_segs_ids, tr_segs_n_nodes):  # number of nodes for each original track
                orig_id_n_nodes[o_id] += n

                if main_id == o_id:
                    corr_nodes += n
                all_nodes += n

            orig_ids_uniq = list(orig_id_n_nodes.keys())
            n_nodes_per_orig_ids_uniq = list(orig_id_n_nodes.values())
            likely_id_idx = np.argmax(n_nodes_per_orig_ids_uniq)  # index of element with most nodes
            track_likely_orig_id = orig_ids_uniq[likely_id_idx]  # track ID wirth most nodes
            n_nodes_from_orig_id = n_nodes_per_orig_ids_uniq[likely_id_idx]
            n_nodes_tot = np.sum(n_nodes_per_orig_ids_uniq)

            n_seg_likely_id = orig_segs_ids.count(track_likely_orig_id)
            n_seg_total = len(orig_segs_ids)

            # continuous length of each orig ID, nodes
            curr_id = -2
            curr_len = 0
            for i, tr_id_n in enumerate(zip(orig_segs_ids, tr_segs_n_nodes)):
                tr_id, n = tr_id_n
                if curr_id != tr_id:
                    if i != 0:
                        tracks_length[0].append(curr_id)
                        tracks_length[1].append(curr_len)
                    curr_id = tr_id
                    curr_len = 0
                curr_len += n
            tracks_length[0].append(curr_id)
            tracks_length[1].append(curr_len)

            tracks_segment_efficiency[1].append(n_seg_likely_id / n_seg_total)
            tracks_node_efficiency[1].append(n_nodes_from_orig_id / n_nodes_tot)
            tracks_means[1].append(m)
            tracks_sigma[1].append(s)

            tracks_segment_efficiency[0].append(track_likely_orig_id)
            tracks_node_efficiency[0].append(track_likely_orig_id)
            tracks_means[0].append(track_likely_orig_id)
            tracks_sigma[0].append(track_likely_orig_id)

        print('eff=%.5f' % (corr_nodes / all_nodes))
        return tracks_segment_efficiency, tracks_node_efficiency, tracks_means, tracks_sigma, tracks_length

    def plot_history(self):
        fig, ax = plt.subplots(1, 6, figsize=(25, 7))

        ax[0].plot(self.LSE_hist)
        ax[0].set_title('LSE evolution')

        ns = np.array(self.merge_n1_n2_hist).transpose()
        if len(ns):
            ax[1].plot(ns[0], 'g')
            ax[1].plot(ns[1], 'b')
            ax[1].set_title('size of merges segments')

        q = self.get_quality_info()
        if q is not None:
            t_seg_eff, t_nod_eff, t_m, t_s, t_l = q

            ax[2].plot(t_seg_eff[0], t_seg_eff[1], 'k*')
            ax[2].set_title('segment efficiency')

            ax[3].plot(t_nod_eff[0], t_nod_eff[1], 'k*')
            ax[3].set_title('nodes efficiency')

            ax[4].errorbar(t_m[0], t_m[1], yerr=t_s[1], fmt='or', label='estimate', capsize=5)
            if self.orig_mu_sgm is not None:
                musgm = np.array(self.orig_mu_sgm).transpose()
                ax[4].errorbar(np.arange(len(musgm[0])), musgm[0], yerr=musgm[1], fmt='og', label='GT', capsize=5)
            ax[4].set_title('per track mean & std')
            ax[4].legend()

            ax[5].scatter(t_l[0], t_l[1], marker='o')
            ax[5].set_title('lengthes of continuous\ntrack segments')
        plt.tight_layout(pad=0, h_pad=0, w_pad=0)
        plt.show()

    def eval_mean_tr_len(self):
        tr_lens = [tr.get_num_nodes() for tr in self.tracks]
        n_mean = np.mean(tr_lens)
        print(n_mean)
        # _ = plt.hist(tr_lens, 30)
        # plt.show()

    def solve(self, draw_fn=None, hist_fn=None,
              orig_tr_idx_map=None, orig_mu_sgm=None,
              lap_only=False, priors_set=False,
              stop_after_init=False,
              pm_mtr=None,
              remove_short=False
              ):
        proc_t = -timer()
        draw_t = 0

        if orig_tr_idx_map is not None:
            self.orig_tr_idx_map = orig_tr_idx_map
        if orig_mu_sgm is not None:
            self.orig_mu_sgm = orig_mu_sgm.copy()

        if not priors_set:
            self.fill_priors(min_len=20)

        self.init_pm(pm_mtr)

        if stop_after_init:
            return

        if draw_fn:
            draw_t -= timer()
            draw_fn(self)
            draw_t += timer()

        if remove_short:
            print('removing short tracks')
            self.remove_short_tracks()

        draw_t -= timer()
        self.eval_mean_tr_len()
        draw_t += timer()

        self.tracks_orig = copy.deepcopy(self.tracks)

        if hist_fn:
            draw_t -= timer()
            hist_fn(self)
            draw_t += timer()
        # if draw_fn:
        #    draw_fn(self)

        if not lap_only:
            # self.resolve_unique(draw_fn)

            # self.resolve_unique_in_xings(draw_fn)

            # self.resolve_significantly_unique_in_xings(draw_fn)
            # self.resolve_significantly_unique_in_xings_and_unique_in_xings(draw_fn)
            self.resolve_significantly_unique_all_and_significantly_unique_in_xings_and_unique_in_xings(draw_fn)

            self.add_lse_history()
            draw_t -= timer()
            # self.plot_history()
            draw_t += timer()

            # self.resolve_remaining_minflow()

            # Debug commented self.resolve_remaining_LAP()
        else:
            self.resolve_remaining_LAP(lim=Solver.max_chi2 / 9)
            self.resolve_remaining_LAP(lim=Solver.max_chi2 / 3)
            self.resolve_remaining_LAP(lim=Solver.max_chi2 / 3 * 2)
            self.resolve_remaining_LAP(lim=Solver.max_chi2)
            self.resolve_remaining_LAP(lim=Solver.max_chi2 * 200)

        draw_t -= timer()
        # self.plot_history()
        draw_t += timer()
        self.resolve_remaining_associations_LAP()

        self.add_lse_history()

        if hist_fn:
            draw_t -= timer()
            hist_fn(self)
            draw_t += timer()

        draw_t -= timer()
        # self.plot_history()
        draw_t += timer()

        proc_t += timer()

        proc_t -= draw_t

        print(f'proc time: {proc_t:.2f}s, draw time: {draw_t:.2f}s')


def plot_state(s: Solver, orig=False):
    if not orig:
        return
    xing = s.crossings
    cols = ["#76d88d", "#b958d8", "#78d646", "#d4689d", "#d3d254", "#7989d6", "#dc5644", "#57c6bc", "#c88c48",
            "#759649"]

    n_t = 180
    n_tr = len(s.tracks)

    plt.figure(figsize=(n_t // 5, (n_tr + 1) / 5))
    # for tr_id in range(n_tr):
    #    plt.plot(np.zeros(n_t)+tr_id, '.', markersize=8, color=cols[tr_id])

    for tr_idx in range(n_tr):
        track = s.tracks_orig[tr_idx] if orig else xing.tracks[tr_idx]
        n = track.get_num_nodes()
        if track.merged_into is not None:
            # print(track.merged_into)
            pass
        if n == 0:
            continue
        for seg in track.segments:
            for t in seg.times:
                plt.plot([t], [tr_idx], '.', markersize=10, color=cols[tr_idx % len(cols)])
        x = (track.get_first_time() + track.get_last_time()) / 2
        y = tr_idx + 0.05
        plt.text(x, y, str(tr_idx), color='b')

    for c_idx, c in enumerate(xing.xings):
        x = y = 0
        n = 0
        for idx_in, time_in in c.in_tracks_idxtime:
            for idx_out, time_out in c.out_tracks_idxtime:
                if time_out < time_in:
                    continue
                tr_in = idx_in
                tr_out = idx_out
                x += (time_in + time_out) / 2
                y += (tr_in + tr_out) / 2
                n += 1

                chi2 = s.pm.mtr[idx_in, idx_out]
                if chi2 == -1 or chi2 > 90:
                    continue

                qual = min(chi2, 6) / 6 if chi2 >= 0 else 1
                col = (0, 1 - qual, 0)
                plt.plot([time_in - 0.5, time_out + 0.5], [tr_in, tr_out], linewidth=1, color=col)
        if n == 0:
            pass  # print ('xing', c_idx, 'is strange')
        else:
            x /= n
            y /= n
            plt.text(x + 0.5, y + 0.2, str(c_idx), color='r')

    for c in s.resolved_conns:
        tr_idx1, t1, tr_idx2, t2, chi2 = c

        tr_1 = tr_idx1
        tr_2 = tr_idx2

        x = (t1 + t2) / 2
        y = (tr_1 + tr_2) / 2

        if t2 < t1:
            t2, t1 = t1, t2

        plt.plot([t1 - 0.5, t2 + 0.5], [tr_1, tr_2], linewidth=1, color='red')
        plt.text(x, y + 0.2, '%.2f' % chi2, color='r', fontdict={'ha': 'center', 'va': 'center'})

    plt.show()


def plot_tracks(solver, show_line=False, min_len=0, t_max=190, show_num=False):
    f = 30  # per 1000 um
    x0, x1, y0, y1 = Track._boundary_x0x1y0y1 or (0, 1000, 0, 1000)

    w, h = abs(x1 - x0) / 1000, abs(y1 - y0) / 1000
    plt.figure(figsize=(w * f, h * f))

    tracks_c = [tr for tr in solver.tracks if tr.get_num_nodes() >= min_len]
    tracks_idx = [idx for idx, tr in enumerate(solver.tracks) if tr.get_num_nodes() >= min_len]
    excluded = {idx for idx, tr in enumerate(solver.tracks) if tr.get_num_nodes() < min_len}

    x_all = []
    y_all = []
    t_all = []
    for idx, tr in zip(tracks_idx, tracks_c):
        x = []
        y = []
        t = []
        for seg in tr.segments:
            for time, node in zip(seg.times, seg.nodes):
                x.append(node.r[0])
                y.append(node.r[1])
                t.append(time)

        if show_line:
            plt.plot(x, y, color='k', alpha=0.12)

        if show_num:
            xm, ym = np.mean(x), np.mean(y)
            plt.text(xm, ym, '%d' % idx, horizontalalignment='center')

        x_all.extend(x)
        y_all.extend(y)
        t_all.extend(t)

    t_all = np.array(t_all)
    plt.scatter(x_all, y_all, color=cm.rainbow(t_all / t_max), marker='.')

    # for es in conns:
    #    e_tr_idx_e, e_tr_idx_s, ee_t, es_t, ex, ey, sx, sy = es
    #    if e_tr_idx_e in excluded or e_tr_idx_s in excluded:
    #        continue
    #    plt.plot([ex,sx], [ey, sy], color=cm.rainbow(ee_t/180))

    plt.grid(False)


def get_tr_to_tr_chi2(solver, i, j, tr=20, dbg=True):
    tri = solver.tracks[i]
    trj = solver.tracks[j]
    return tri.get_connection_chi2(trj, time_range=tr, print_dbg_nll=dbg)


# ### saving/conversion interface
def save_tracks_for_display_from_solver(slv: Solver, fname):
    # 1. fill dictionary
    uidx = -1  # only container related

    tracks_xyz_t_id_uidx = []
    tracks_aux_pars = []
    tracks_aux_local_pars = []
    tracks_aux_map = None
    tracks_aux_local_map = None
    tracks_xyz_map = None

    def _get_xyztiduidx_aux(trk: Track, uidx, oidx):
        trk: Track = trk

        def nodes_aggregator(node: Node, t_idx, arr):
            x, y, z = node.r

            arr.append([x, y, z, t_idx])
            return arr

        xyztid = []
        xyztid = trk.aggregate_nodes(nodes_aggregator, xyztid)

        uidx += 1
        all_t = [v[3] for v in xyztid]
        if len(all_t) == 0:
            print(len(xyztid), trk.get_num_nodes())

        dt = all_t[-1] - all_t[0]

        mean_std, (n_nodes, pars_dict) = trk.get_mean_std(return_arr=True)
        flow = 1  # by construction for current solver
        linv = mean_std['s_linv'][0] ** 2
        v = mean_std['s_move_v'][0] ** 2
        ecc = mean_std['ecc'][0]
        kink = mean_std['kink'][0]
        w = mean_std['w'][0]
        directionality = mean_std['directionality'][0]

        aux = [uidx, oidx, flow, n_nodes, dt,
               linv, v, ecc, kink, w, directionality,
               ]

        square = lambda x: [xi ** 2 for xi in x]
        d_v = dict(zip(pars_dict['s_v_t'], square(pars_dict['s_v'])))
        d_linv = dict(zip(pars_dict['s_linv_t'], square(pars_dict['s_linv'])))
        d_ecc = dict(zip(pars_dict['ecc_t'], pars_dict['ecc']))
        d_phi = dict(zip(pars_dict['phi_t'], pars_dict['phi']))
        d_MI = dict(zip(pars_dict['MI_t'], pars_dict['MI']))
        d_kink = dict(zip(pars_dict['kink_t'], pars_dict['kink']))
        d_w = dict(zip(pars_dict['w_t'], pars_dict['w']))

        all_dict_pars = [d_v, d_linv, d_ecc, d_phi, d_MI, d_kink, d_w]
        all_dflt_pars = [v, linv, ecc, 0, 1, kink, w]

        def _get_val_nearest(par_dict, t, def_par):
            if len(par_dict) == 0:
                return def_par
            if t in par_dict:
                return par_dict[t]
            else:
                tp = t
                tm = t
                while True:
                    tp += 1
                    tm -= 1
                    # print(t, tp, tm, list(par_dict.keys()))
                    vp = par_dict.get(tp, None)
                    if vp is not None:
                        return vp
                    vm = par_dict.get(tm, None)
                    if vm is not None:
                        return vm

        aux_local = [[_get_val_nearest(par, t, def_par) for par, def_par in zip(all_dict_pars, all_dflt_pars)] for t in
                     all_t]
        return xyztid, aux, aux_local, uidx

    trk: Track
    for trk_idx, trk in enumerate(slv.tracks):
        if trk.get_num_nodes() == 0:
            continue
        if trk.is_nc():
            continue
        n_non_nc = np.sum([not seg.is_nc for seg in trk.segments])
        if n_non_nc == 0:
            continue

        xyztiduidx, aux, aux_local, uidx = _get_xyztiduidx_aux(trk, uidx, trk_idx)
        tracks_xyz_t_id_uidx.append(xyztiduidx)
        tracks_aux_pars.append(aux)
        tracks_aux_local_pars.append(aux_local)

    tracks_xyz_map = {0: 'x', 1: 'y', 2: 'z', 3: 't_idx'}

    tracks_aux_map = {0: 'uidx',
                      1: 'oidx',
                      2: 'flow',
                      3: 'n_nodes',
                      4: 'dt',
                      5: 'linv',
                      6: 'v',
                      7: 'ecc',
                      8: 'kink',
                      9: 'w',
                      10: 'directionality',
                      }
    tracks_aux_local_map = {0: 'v', 1: 'linv', 2: 'ecc', 3: 'phi',
                            4: 'MI', 5: 'kink', 6: 'w'
                            }

    dataset = {
        'tracks_xyz': tracks_xyz_t_id_uidx,
        'tracks_aux': tracks_aux_pars,
        'tracks_aux_local': tracks_aux_local_pars,
        'xyz_map': tracks_xyz_map,
        'aux_map': tracks_aux_map,
        'aux_local_map': tracks_aux_local_map
    }
    with open(fname, 'wb') as f:
        pickle.dump(dataset, f)


def convert_to_xings(stack, vtx_xg, sgm_xg, sgm_1g, get_tracks=False):
    # sgm_xg - segments of x-ings. not grouped by x-ing
    # sgm_1g - segments entering/exiting x-ings. not grouped by x-ing
    # vtx_xg - vertices of crossings, grouped by xing

    tracks = []

    vtx_tc_to_l_tr_idx = {}
    vtx_tc_to_r_tr_idx = {}

    sgm_guid = -1
    for gr_idx, sgm_1_grp in enumerate(sgm_1g):
        for sgm_idx, sgm in enumerate(sgm_1_grp):
            sgm_guid += 1
            segments_tc = []
            curr_seg_tc = []

            sgm_jump_start_tidx = [tb for tb, te in sgm.mj_t_ranges + sgm.fj_t_ranges]

            for t_idx in sorted(sgm.node_map.keys()):
                c_arr = sgm.node_map[t_idx]
                t_in_jump = np.any([tb < t_idx < te for tb, te in sgm.mj_t_ranges + sgm.fj_t_ranges])
                if len(c_arr) == 1 and not t_in_jump:  # skip nodes that are parts of other_cell or in a 'forbidden' timerange
                    curr_seg_tc.append((t_idx, c_arr[0]))  # eventually parts can be merged
                else:
                    if len(curr_seg_tc):
                        segments_tc.append(curr_seg_tc)
                        curr_seg_tc = []

                if t_idx in sgm_jump_start_tidx and len(curr_seg_tc):
                    segments_tc.append(curr_seg_tc)
                    curr_seg_tc = []

            if len(curr_seg_tc):
                segments_tc.append(curr_seg_tc)

            if len(segments_tc):
                tr: Track or None = None

                for seg_idx, seg_tc in enumerate(segments_tc):
                    tseg = TSegment()
                    tseg.info = (gr_idx, sgm_idx)  # maintains original indexes for inspection and backtracing

                    for t_idx, c_idx in seg_tc:
                        cell = stack.st[t_idx].cells[c_idx]
                        # get from stack
                        # sgm_guid, t, x, y, z, w, phi, eccentr = n_rd
                        t = t_idx
                        n = Node(cell.x, cell.y, cell.z, cell.w, cell.phi, cell.ecc, *cell.aux_ch)
                        tseg.add_node(n, t)

                    if tr is None:
                        tr = Track(tseg)
                    else:
                        tr.add_segment(tseg)

                tr_idx = len(tracks)

                l_vtx_tc = sgm.l_links[0][0]
                r_vtx_tc = sgm.r_links[0][1]

                if l_vtx_tc != (-1, -1):
                    if l_vtx_tc not in vtx_tc_to_r_tr_idx:  # in `vtx_tc_to_r_tr_idx` r is right from vtx
                        vtx_tc_to_r_tr_idx[l_vtx_tc] = []
                    vtx_tc_to_r_tr_idx[l_vtx_tc].append(tr_idx)

                if r_vtx_tc != (-1, -1):
                    if r_vtx_tc not in vtx_tc_to_l_tr_idx:
                        vtx_tc_to_l_tr_idx[r_vtx_tc] = []
                    vtx_tc_to_l_tr_idx[r_vtx_tc].append(tr_idx)

                tracks.append(tr)

    if get_tracks:
        return tracks

    # create all NC tracks
    for gr_idx, vtx_x_grp in enumerate(vtx_xg):
        for vtx_x in vtx_x_grp:
            for vtx in vtx_x:
                if vtx.l_nc_f_slv > 0 or vtx.r_nc_f_slv > 0:
                    t_idx, c_idx = vtx.t_idx, vtx.c_idx
                    vtx_tc = t_idx, c_idx

                    nc_types = [TSegment._NC_TP_END, TSegment._NC_TP_START]
                    nc_idxs = [range(vtx.l_nc_f_slv), range(vtx.r_nc_f_slv)]

                    for nc_type, nc_idx in zip(nc_types, nc_idxs):
                        for idx in nc_idx:
                            tseg = TSegment(nc_type=nc_type)
                            tseg.info = (gr_idx, -1)  # maintains original indexes for inspection and backtracing

                            cell = stack.st[t_idx].cells[c_idx]
                            # get from stack
                            # sgm_guid, t, x, y, z, w, phi, eccentr = n_rd
                            t = t_idx
                            n = Node(cell.x, cell.y, cell.z, cell.w, cell.phi, cell.ecc, *cell.aux_ch)
                            tseg.add_node(n, t)

                            tr = Track(tseg)

                            tr_idx = len(tracks)
                            # print(tr_idx, nc_type)

                            # this \|/  remains to be clarified, so that all tracks, including these virtual NCc are assigned correctly

                            # in `vtx_tc_to_r_tr_idx` r is right from vtx
                            vtx_tc_to_tr_idx = vtx_tc_to_r_tr_idx if nc_type == TSegment._NC_TP_START else vtx_tc_to_l_tr_idx
                            if vtx_tc not in vtx_tc_to_tr_idx:
                                vtx_tc_to_tr_idx[vtx_tc] = []
                            vtx_tc_to_tr_idx[vtx_tc].append(tr_idx)

                            tracks.append(tr)

    # create all endpoint tracks for segments ending inside crossing
    for gr_idx, sgm_g in enumerate(sgm_xg):
        for sgm in sgm_g:
            l_idx = set([lnk[0] for lnk in sgm.l_links])
            r_idx = set([lnk[1] for lnk in sgm.r_links])

            endpoint = (-1, -1)
            l_is_end, r_is_end = [endpoint in end_idxs for end_idxs in [l_idx, r_idx]]
            l_idx = list(l_idx)
            r_idx = list(r_idx)

            if l_is_end != r_is_end:  # only one is end, enother points to a vertex of the group
                vtx_tc = r_idx[0] if l_is_end else l_idx[0]
                t_idx, c_idx = vtx_tc

                nc_type = TSegment._NC_TP_END if l_is_end else TSegment._NC_TP_START
                nc_idx = range(sgm.flow_slv)

                for nc_tr_idx in nc_idx:
                    tseg = TSegment(nc_type=nc_type)
                    tseg.info = (gr_idx, -1)  # maintains original indexes for inspection and backtracing

                    cell = stack.st[t_idx].cells[c_idx]
                    # get from stack
                    # sgm_guid, t, x, y, z, w, phi, eccentr = n_rd
                    t = t_idx
                    n = Node(cell.x, cell.y, cell.z, cell.w, cell.phi, cell.ecc, *cell.aux_ch)
                    tseg.add_node(n, t)

                    tr = Track(tseg)

                    tr_idx = len(tracks)

                    vtx_tc_to_tr_idx = vtx_tc_to_r_tr_idx if nc_type == TSegment._NC_TP_START else vtx_tc_to_l_tr_idx
                    if vtx_tc not in vtx_tc_to_tr_idx:
                        vtx_tc_to_tr_idx[vtx_tc] = []
                    vtx_tc_to_tr_idx[vtx_tc].append(tr_idx)

                    tracks.append(tr)

    # create connection object and fill connections accordingy
    xing = Xings(tracks)

    for gr_idx, vtx_x_grp in enumerate(vtx_xg):
        for vtx_x in vtx_x_grp:
            crs_i, crs_o = set(), set()

            for vtx in vtx_x:
                t_idx, c_idx = vtx.t_idx, vtx.c_idx
                tc_idx = (t_idx, c_idx)
                # left, incoming, entering
                for tr_idx in vtx_tc_to_l_tr_idx.get(tc_idx, []):
                    crs_i.add((tr_idx, t_idx))
                # right, outgoing, leaving
                for tr_idx in vtx_tc_to_r_tr_idx.get(tc_idx, []):
                    crs_o.add((tr_idx, t_idx))

            crs = [crs_i, crs_o]
            # print(crs)

            xing.add_full_xing(crs)

    return xing


def check_solution_ok(slv):
    tracks = slv.tracks
    for xingi in slv.crossings.xings:
        if len(xingi.in_tracks_idxtime) or len(xingi.out_tracks_idxtime):
            i_trs = [idx for idx, tin in xingi.in_tracks_idxtime]
            o_trs = [idx for idx, tin in xingi.out_tracks_idxtime]
            i_nseg = [len(tracks[idx].segments) for idx in i_trs]
            o_nseg = [len(tracks[idx].segments) for idx in o_trs]
            if np.min(i_nseg + o_nseg) == 0:
                return False

    return True


# ## Merging broken tracks at diapedesis point   -  not used as doesn't really work
# ### Find number of tracks around a point

def get_tracks_near_point(r0, t0, tracks, dr0):
    dr02 = dr0 ** 2
    res = {}
    for tr_idx, tr in enumerate(tracks):
        if tr.merged_into is not None:
            continue
        if not tr.time_on_track(t0):
            continue
        is_near = False
        for seg in tr.segments:
            if seg.is_nc:
                continue
            #             for ri_min, ri, ri_max in zip(seg.r_min, r0, seg.r_max)
            #                 inside = ((ri_min-dr0) <= ri <= (ri_max+dr0))

            inside = [((ri_min - dr0) <= ri <= (ri_max + dr0)) for ri_min, ri, ri_max in zip(seg.r_min, r0, seg.r_max)]
            if not all(inside):
                continue

            for t, n in zip(seg.times, seg.nodes):
                if t == t0:
                    dr2 = Track.r2(r0, n.r)
                    if dr2 <= dr02:
                        is_near = True
                        break
            if is_near:
                break

        if is_near:
            assert len(n.pars) == 8
            res[tr_idx] = {
                'dr': sqrt(dr2),
                'doc': n.pars[-2] / 255.,
            }
    return res


def get_tracks_near_tc(tc_idx, stack, tracks, dr0, dt=0):
    t_idx, c_idx = tc_idx

    st = stack.st[t_idx]
    cell = st.cells[c_idx]
    r0 = cell.get_r()
    t0 = t_idx + dt
    return get_tracks_near_point(r0, t0, tracks, dr0)


def get_tracks_near_tc_rt(tc_idx, stack, tracks, dr0, dt0):
    """

    Args:
        tc_idx: time-cell indexes tuple
        stack: st_full
        tracks: list of tracks
        dr0: search radius
        dt0: search time radius

    Returns:

    """
    # ignores info, just collect set of tracks that are around
    set_tr = set()
    dt_to_n_tr = {}

    for dt in range(-dt0, dt0 + 1):
        tr_inf_dict = get_tracks_near_tc(tc_idx, stack, tracks, dr0, dt)
        tr_ids = set(tr_inf_dict.keys())
        dt_to_n_tr[dt] = len(tr_ids)
        set_tr.update(tr_ids)
    return {
        'set_tr': set_tr,
        'dt_to_n_tr': dt_to_n_tr
    }


def get_tracks_near_point_rt(r0, t0, tracks, dr0, dt0):
    # ignores info, just collect set of tracks that are around
    set_tr = set()
    dt_to_n_tr = {}

    for dt in range(-dt0, dt0 + 1):
        tr_inf_dict = get_tracks_near_point(r0, t0 + dt, tracks, dr0)
        tr_ids = set(tr_inf_dict.keys())
        dt_to_n_tr[dt] = len(tr_ids)
        set_tr.update(tr_ids)
    return {
        'set_tr': set_tr,
        'dt_to_n_tr': dt_to_n_tr
    }


def collect_doc(n, t, arg):
    arg[1].append(t - arg[0])
    arg[2].append(n.pars[-2] / 255)


def get_tr_doc_around(t, dt, track):
    tsns = [t, [], []]

    track.aggregate_nodes_in_time_interval(collect_doc, tsns, t - dt, t + dt)

    return tsns[1:]


# ### merging overlapping tracks
# The currently used track-to-track NLL metric doesn't allow for differentiation (not even poor :/)
# of broken track parts from 2 different tracks
def get_track_doc(tr):
    if tr.merged_into is not None:
        return None

    ts = []
    docs = []
    for seg in tr.segments:
        if seg.is_nc:
            continue

        for t, n in zip(seg.times, seg.nodes):
            ts.append(t)
            docs.append(n.pars[-2] / 255.)

    return ts, docs


def plot_track_doc(tr=None, tsdocs=None):
    tsdocs = tsdocs or get_track_doc(tr)
    if tsdocs is None:
        return

    ts, docs = tsdocs
    if len(ts):
        plt.plot(ts, docs)
        plt.ylim(0, 1)


##################################################################
#                            main                                #
##################################################################


def track_dataset(cells_filename='tr_cells_tmp.dat'):
    # prepare & wait to start


    make_image_dirs()
    wait_to_start()
    proc_start_t = timer()

    # ## Cell linking

    # load file
    cells_merged, cells_full = load_cells(cells_filename)
    save_cell_for_display(cells_ds=cells_merged)

    st_merged = Stack(cells_merged)
    st_full = Stack(cells_full)

    # ### Merging neighbouring cells

    # get and plot distribution of dr<50um:
    # plot_neighbour_cell_count_distrinution(st_full, dr_max=50)

    # get_cell_merge_groups(st_full, 20);
    # merge_groups_sc, remove_mc = get_subcell_merge_groups(st_full, 20)
    # cells_merged_n, cells_full_n = merge_cell_groups(cells_full, merge_groups_sc, remove_mc)

    # process: merge cells within _CELL_FUSE_RADIUS um and make new stacks
    cells_merged_n, cells_full_n = merge_nearby_cells(st_full, cells_full, dr_max=cfgm.CELL_FUSE_RADIUS)

    del cells_merged
    del cells_full

    # update main containers for new-merged cells
    if cfgm.MERGE_CLOSE_CELLS:
        st_merged = Stack(cells_merged_n)
        st_full = Stack(cells_full_n)

    save_cell_for_display(cells_ds=cells_merged_n)

    # ### Do linking
    # prepare solve
    # 1 make links
    links = make_links(max_dr=cfgm.LINKS_MAX_DR, eps=cfgm.LINKS_EPS, stack=st_merged, max_dt=3)
    # 2 calc weights
    links_nll = get_links_w(stack=st_merged, links=links, w_nn=cfgm.W_NN)
    links, links_nll = discard_bad_links(links, links_nll)

    # 3 solve
    resolved_links = solve_disjoint_cached(stack=st_merged,
                                           links=links, links_w=links_nll,
                                           w_nc=cfgm.W_NC, w_dm=cfgm.W_DM, use_cache=cfgm.USE_CASHED)

    # proc cells
    lc_map = get_linked_cell_map(st_merged, resolved_links, links, links_nll)

    # split disjointed components
    lc_disjoint_groups = get_disjointed_groups_links(list(lc_map.keys()), resolved_links, min_grp_size=2)

    # plot_lc_quality_info(lc_map)

    # get all track starts - each is list[tuple[node_tc_idx, link]
    all_starts, all_starts_g, all_starts_single_g = get_track_starts(lc_map, lc_disjoint_groups)
    all_tracks_xyt, all_tracks_start_tcidx = get_track_start_info(st_merged, all_starts, lc_map)

    # ### Tracks export
    save_tracks_simple(all_tracks_xyt, 'tracks_merged.txt')
    # plot_tracks_simple(st_merged, all_tracks_xyt)

    # ### Study of connectivity
    m_per_grp, groups_with_m_nodes = get_n_m_nodes_per_group(lc_map, lc_disjoint_groups)

    if cfgm.SAVE_IMS:
        plot_linked_cells_in_m_groups(groups_with_m_nodes, all_starts_g, all_starts_single_g,
                                      all_tracks_start_tcidx, all_tracks_xyt)


    # ## Solving for consistent picture

    # ### Simplify graph
    # replace resolved sequences with ~constant flow with "Segments", linking "vertices"
    vtx_g, sgm_g = get_group_segments_vertices_from_links(st_merged, lc_map, all_starts_single_g)

    # plot simplified segments

    if cfgm.SAVE_IMS:
        plot_segments_in_groups(st_merged,
                                groups_with_m_nodes,
                                all_starts_g,
                                all_tracks_start_tcidx, all_tracks_xyt,
                                vtx_g, sgm_g, folder='tracks_in_group_simpl')

    # ### Find posible vtx: jumps on segments

    # 1 find jump scale
    dr_merge = get_all_merge_jump_dr(vtx_g, sgm_g, st_merged)
    dr_sqrt_merge_min_is_vtx, *dr_sqrt_merge_mean_std = get_min_dr_sqrt_merge(dr_merge, plot=False)

    # 2 find potentially vtx nodes on segments. for now only by jump value
    vtx_cand_tc_idx = find_vtx_on_segments(sgm_g, st_merged, lc_map, min_sqrt_dr_candidate=dr_sqrt_merge_min_is_vtx)


    # split segments starting from selected nodes (vtx_cand_tc_idx) - will be starting vtx
    # repeat supplying new vtx_cand_tc_idx
    vtx_g, sgm_g = get_group_segments_vertices_from_links(st_merged, lc_map, all_starts_single_g, vtx_cand_tc_idx)

    # plot simplified segments
    if cfgm.SAVE_IMS:
        plot_segments_in_groups(st_merged,
                                groups_with_m_nodes,
                                all_starts_g,
                                all_tracks_start_tcidx, all_tracks_xyt,
                                vtx_g, sgm_g, folder='tracks_in_group_simpl_sgm_vtx')

    # ### Solving for track continuity (flow consistency)
    # 1 solve
    solve_groups_flow(stack=st_merged,
                      vtx_g=vtx_g, sgm_g=sgm_g,
                      w_nc=cfgm.W_NC_LOC, w_f_mult_end=cfgm.W_F_MULT_END_LOC, w_f_above_est=cfgm.W_F_ABOVE_EST_LOC)

    # plot simplified segments, vtx on segments
    if cfgm.SAVE_IMS:
        plot_segments_in_groups(st_merged,
                                groups_with_m_nodes,
                                all_starts_g,
                                all_tracks_start_tcidx, all_tracks_xyt,
                                vtx_g, sgm_g, folder='tracks_in_group_simpl_flow',
                                only_m_nodes=True, multiplicity_colors=True)

    # ### Global flow resolving

    # make potential segments connecting the across jumps and merge displacements
    vtx_tc_idx = [(vtx.t_idx, vtx.c_idx) for grp_vtx in vtx_g for vtx in grp_vtx]
    dr_sqrt_flow_mean_std = 25, 45 / 3.5
    vtx_merge_conns = search_vtx_conn(vtx_tc_idx, st_merged,
                                      sgm_g,
                                      dr_sqrt_merge_mean_std, dr_sqrt_flow_mean_std,
                                      w_nc_0=9.)  # segment candidate search and estimation of NLL

    sgm_c = get_conn_segments(vtx_merge_conns, lc_map, st_merged)  # creates segment candidate objects

    vtx_g2, sgm_g2, sgm_c_g2, all_starts_g2 = get_disjointed_groups_segments(vtx_g, sgm_g, sgm_c,
                                                                             all_starts_g,
                                                                             only_connected_sgm=False)

    # solve
    # 1 solve
    res = solve_groups_global_flow(stack=st_merged,
                                   vtx_g=vtx_g2, sgm_g=sgm_g2, sgm_gc=sgm_c_g2,
                                   w_nc=cfgm.W_NC_GLOB, w_f_mult_end=cfgm.W_F_MULT_END_GLOB,
                                   w_f_above_est=cfgm.W_F_ABOVE_EST_GLOB)

    if not res:
        print('Global flow solving failed sor at least one group (G2)')

    # plot simplified segments, vtx on segments, v g2
    if cfgm.SAVE_IMS:
        plot_segments_in_groups(stack=st_merged,
                                groups_with_m_nodes=None,
                                all_starts_g=all_starts_g2,
                                all_tracks_start_tcidx=all_tracks_start_tcidx, all_tracks_xyt=all_tracks_xyt,
                                vtx_g=vtx_g2, sgm_g=sgm_g2, sgm_c_g=sgm_c_g2,
                                folder='tracks_in_group_simpl_flow_glob',
                                only_m_nodes=False, multiplicity_colors=True)

    # ### Refining global result
    # PROD: solve iteratively with shaving, remove groups with <6 nodes in total (useless for any analysis)


    vtx_g3, sgm_g3, sgm_c_g3, all_starts_g3 = remove_small_groups(vtx_g2, sgm_g2, sgm_c_g2, all_starts_g2)
    sgm_g3, sgm_c_g3 = shave_and_solve_groups_global_flow(st_merged, vtx_g3, sgm_g3, sgm_c_g3)


    # plot simplified segments, vtx on segments, v g3
    if cfgm.SAVE_IMS:
        plot_seg_flow(vtx_g3, sgm_g3, sgm_c_g3, all_starts_g3, st_merged, all_tracks_start_tcidx, all_tracks_xyt, saveto='tracks_in_group_simpl_flow_glob_shaved\\')
        plot_seg_flow_change_dist(sgm_g3, saveto='seg_flow_change_hist.png')

    # ## Resolving crossings

    # ### Extract joined segments

    # 0. regroup, and merge all segments in one collection (all are approved)
    vtx_g4, sgm_g4, sgm_c_g4, all_starts_g4 = get_disjointed_groups_segments(vtx_g3, sgm_g3, sgm_c_g3,
                                                                             all_starts_g3,
                                                                             only_connected_sgm=True)

    vtx_g4, sgm_g4, sgm_c_g4, all_starts_g4 = merge_selected_potential_segments(vtx_g4, sgm_g4, sgm_c_g4, all_starts_g4)
    del sgm_c_g4

    # sgm_xg - segments within crossing, sgm_1g - between
    vtx_xg, sgm_xg, sgm_1g = restructure_segments_grps(vtx_g=vtx_g4, sgm_g=sgm_g4, stack=st_merged)

    # plot simplified segments, vtx on segments, v xg
    if cfgm.SAVE_IMS:
        plot_xing_segs(vtx_xg, sgm_xg, sgm_1g, all_starts_g4, st_merged, all_tracks_start_tcidx, all_tracks_xyt,
                       saveto='tracks_in_group_simpl_flow_glob_xing_ready')

    # ### Extend free segments to merged nodes in vtx/crossing segments
    # This section operates on a different other_cell stack than before:
    # with both m-cells and subcells in the collection.
    # This operation invalidates the `lc_map` on the `sgm` objects: different combined m-cells are made,
    # and the lc-map is not updated, since other_cell links aren't used anymore

    pickle_svs(st_merged, st_full, vtx_xg, sgm_xg, sgm_1g, sfx='pre_ext')
    # st_merged, st_full, vtx_xg, sgm_xg, sgm_1g = load_svs(sfx='pre_ext')

    save_tracks_for_display_from_sgm_vtx(stack=st_merged,
                                         vtx_xg=vtx_xg, sgm_xg=sgm_xg, sgm_1g=sgm_1g,
                                         fname='disp_tracks_pre_xing_ext.pckl')

    se = SegmentExtender(st_full, vtx_xg, sgm_xg, sgm_1g)
    vtx_xg, sgm_xg, sgm_1g = se.extend_all_outer_segments_iter()

    pickle_svs(st_merged, st_full, vtx_xg, sgm_xg, sgm_1g)
    # st_merged, st_full, vtx_xg, sgm_xg, sgm_1g = load_svs()

    save_tracks_for_display_from_sgm_vtx(stack=st_full,
                                         vtx_xg=vtx_xg, sgm_xg=sgm_xg, sgm_1g=sgm_1g,
                                         fname='disp_tracks_post_xing_ext.pckl')

    # plot simplified segments, vtx on segments, v xg
    if cfgm.SAVE_IMS:
        plot_xing_segs(vtx_xg, sgm_xg, sgm_1g, all_starts_g4, st_full, all_tracks_start_tcidx, all_tracks_xyt,
                       saveto='tracks_in_group_simpl_flow_glob_xing_ext')

    # explore_unused_cells_draft(st_full, vtx_xg, sgm_xg, sgm_1g)  # didn't lead wnywhere so far


    # # ### Convert to format for merger and run merging
    # xing0 = convert_to_xings(stack=st_full, vtx_xg=vtx_xg, sgm_xg=sgm_xg, sgm_1g=sgm_1g)
    #
    # # fill pm matrix
    # slv0 = Solver(xing0)
    # slv0.solve(None,  # plot_state,
    #            lap_only=True, priors_set=False,
    #            pm_mtr=None,
    #            stop_after_init=True)
    # pm_mtr_bk = slv0.pm.mtr
    #
    # # resolve crossings
    # # st_merged, st_full, vtx_xg, sgm_xg, sgm_1g = load_svs()
    # xing2 = convert_to_xings(stack=st_full, vtx_xg=vtx_xg, sgm_xg=sgm_xg, sgm_1g=sgm_1g)
    #
    # slv2 = Solver(xing2)
    # slv2.solve(None,  # plot_state,
    #            lap_only=True, priors_set=False,
    #            pm_mtr=pm_mtr_bk,
    #            stop_after_init=False,
    #            remove_short=False
    #            )
    #
    # save_tracks_for_display_from_solver(slv2, 'disp_tracks_slv_lap.pckl')
    #
    # print('check_solution_ok', check_solution_ok(slv2))

    # ## closing part
    proc_end_t = timer()
    print(f'notebook run time: {(proc_end_t - proc_start_t):.2f} sec')
    set_proc_end()

if __name__ == '__main__':
    track_dataset()