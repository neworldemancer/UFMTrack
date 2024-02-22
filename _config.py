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
"""

import math
import numpy as np

from multiprocessing import cpu_count

# *Constants*

# dataset dependent constants
_T_ACCUMULATION_START = 0  # starts from 0
_T_ACCUMULATION_END = 4  # starts from 0
_T_END_TO_COMPLETE = 5
_T_ACCUMULATION_COMPLETE = _T_ACCUMULATION_END + _T_END_TO_COMPLETE

_DT_OFFSET = 0  # offset timeframes to process from. Changes the resulting tracks timing. Default: starts from 0

_DT = 10.  # sec, timestep between sequential timeframes

_EXPECTED_NUM_AUX_CHANNELS = 8
_DOC_AUX_CHANNEL = 6


# run configuration constants
_USE_CASHED = False
_USE_SYNCH_FILES = False  # if true - wait for the _SYNC_FILE_START to appear
_SAVE_IMS = False  # False

_SYNC_FILE_START = 'start.txt'
_SYNC_FILE_DONE = 'done.txt'

_MERGE_CLOSE_CELLS = True

_NUM_WORKERS = 8  # cpu_count() - 1 (works better than all cores)
_LONG_RUN_PRINT_TIMEOUT = 10  # sec

_VERBOSE_XING_SOLVER = False

_IMG_DIRS = [
    'tracks_in_group',
    'tracks_in_group_simpl',
    'tracks_in_group_simpl_flow',
    'tracks_in_group_simpl_flow_glob',
    'tracks_in_group_simpl_flow_glob_shaved',
    'tracks_in_group_simpl_flow_glob_xing_ready',
    'tracks_in_group_simpl_flow_glob_xing_ext',
    'tracks_in_group_simpl_sgm_vtx'
]


# processing constants
_CELL_FUSE_RADIUS = 10.
_W_NC_0 = 9.

# linking constants
_W_NC, _W_NN, _W_DM = _W_NC_0, 2.3, 3.0  # not connected node weight, not-adjaecent time weight, distance weight
_MEAN_V = 0.15
_STD_V2 = 0.02  # std of v^2
_STD_REL_DW2 = 0.158  # std of ((w1-w2)/(w1+w2))^2
_LINKS_MAX_DR = 15.
_LINKS_EPS = 10.

_LINKING_SOLVER_TIMEOUT = 900  # sec
_DEFAULT_SOLVER_TIMEOUT = 350  # sec
_SHAVING_SOLVER_TIMEOUT = 345  # sec

_SHAVING_SOLVER_MAX_ITER = 3

# fiducial area border attenuation constants for `_W_NC`
_BORDER_ATT_DIST = 24.  # um
_CELL_RADIUS = 10.  # um
_EXPFT_A = (_W_NC_0 - 1) / (_W_NC_0 * (1 - math.exp(_CELL_RADIUS - _BORDER_ATT_DIST)))
_EXPFT_Y = 1 - _EXPFT_A

# multiplicity consistency solving constants
_W_NC_LOC, _W_F_MULT_END_LOC, _W_F_ABOVE_EST_LOC = 1.1, 0.5, 0.1
_W_NC_GLOB, _W_F_MULT_END_GLOB, _W_F_ABOVE_EST_GLOB = _W_NC_0, 4., 1.  # 9, 2.3, 2
_W_F_MULT_END_GLOB_SHAVED = 6.

_DRJS_MEAN, _DRJS_STD = np.sqrt(4) * 3, np.sqrt(4)
_DRJF_MEAN, _DRJF_STD = 25., 45. / 3.5

_SEARCH_MERGE_VTX_TIDX_RANGE = (1, 10)
_SEARCH_FLOW_VTX_TIDX_RANGE = (1, 4)

_MAX_TRACK_CONN_RAD_OFS = 150  # search within dt * v_max + _MAX_TRACK_CONN_RAD_OFS
_MAX_TRACK_CONN_RAD_DT = 50  # search within dt <=  _MAX_TRACK_CONN_RAD_DT
_TRACK_PARS_DT_RANGE_SV = (0, 3)
_TRACK_PARS_DT_RANGE_LINV = (5, 10)

_NO_JUMP_DR = 2.  # um
_MAX_JUMP_DX = 200.  # um
_TRACK_DIST_CHI2_MAX_DT = 60

# for each variable prior estimated of mu_pop, s2_pop, and s2_instance are given
_TRACK_PRIORS = {
    's_linv': [6, 5 ** 2, 1 ** 2],
    's_move_v': [6, 5 ** 2, 1 ** 2],
    'ecc': [5, 5 ** 2, 2 ** 2],
    'kink': [6, 5 ** 2, 1 ** 2],
    'directionality': [0, (np.pi / 2) ** 2, (np.pi / 2) ** 2],
    'w': [400, 300 ** 2, 130 ** 2],
    'vector': [(0, 0, 0), (np.pi / 2) ** 2, (np.pi / 2) ** 2]
}

# Track analysis constants
_TRACK_MIN_LEN_STAT = 6
_TRACK_MIN_LEN_ANALYSIS = 30
_NEIGHBOR_TRACK_DIST = 23.  # um
_PROBING_MAX_DISPLACEMENT = 20.  # um
_CLASSIFIER_NAC = 'not_a_cell_classifier_024.lrp'
_CLASSIFIER_DET = 'detached_classifier_024.lrp'