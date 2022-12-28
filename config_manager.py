import _config
from math import *


# class for managing constants from the _config module used as global variables
class ConfigManager:
    def __init__(self):
        """
        The ConfigManager class is used to manage the global variables from the _config module.
        It created default setters and getters for all the variables in the _config module
        except the specified in the custom_properties list.
        """
        self.all_cfg_vars = [v for v in _config.__dict__.keys() if len(v) > 1 and v[0] == '_' and v[1] != '_' and v.lower() != v]
        self.custom_properties = [
            '_T_END_TO_COMPLETE', '_T_ACCUMULATION_END', '_T_ACCUMULATION_COMPLETE',
            '_W_NC_0', '_BORDER_ATT_DIST', '_CELL_RADIUS',
            '_EXPFT_A', '_EXPFT_Y', '_TRACK_PRIORS'
        ]

        self.default_properties = [v for v in self.all_cfg_vars if v not in self.custom_properties]

        # print(self.all_cfg_vars, self.custom_properties, self.default_properties)

        self.create_default_properties()

    def create_default_property(self, cfg_var_name):
        name = cfg_var_name[1:]
        def getter(self):
            return _config.__dict__[cfg_var_name]

        def setter(self, value):
            _config.__dict__[cfg_var_name] = value

        setattr(self.__class__, name, property(getter, setter))

    # create default properties for all the variables in the _config
    # module except the specified in the custom_properties list
    def create_default_properties(self):
        for cfg_var_name in self.default_properties:
            self.create_default_property(cfg_var_name)

    # custom properties follow
    pass

    # # make getter for _config._T_ACCUMULATION_START  in _config as properties
    # @property
    # def T_ACCUMULATION_START(self):
    #     return _config._T_ACCUMULATION_START
    #
    # # make setter for _config._T_ACCUMULATION_START  in _config as properties
    # @T_ACCUMULATION_START.setter
    # def T_ACCUMULATION_START(self, value):
    #     _config._T_ACCUMULATION_START = value

    # make getter for _config._T_ACCUMULATION_START  in _config as properties
    @property
    def T_END_TO_COMPLETE(self):
        return _config._T_END_TO_COMPLETE

    # make setter for _config._T_ACCUMULATION_START  in _config as properties
    @T_END_TO_COMPLETE.setter
    def T_END_TO_COMPLETE(self, value):
        _config._T_END_TO_COMPLETE = value
        _config._T_ACCUMULATION_COMPLETE = _config._T_ACCUMULATION_END + _config._T_END_TO_COMPLETE

    # make getter for _config._T_ACCUMULATION_END  in _config as properties
    @property
    def T_ACCUMULATION_END(self):
        return _config._T_ACCUMULATION_END

    # make setter for _config._T_ACCUMULATION_END  in _config as properties
    @T_ACCUMULATION_END.setter
    def T_ACCUMULATION_END(self, value):
        _config._T_ACCUMULATION_END = value
        _config._T_ACCUMULATION_COMPLETE = _config._T_ACCUMULATION_END + _config._T_END_TO_COMPLETE

    # make getter for _config._T_ACCUMULATION_COMPLETE  in _config as properties
    @property
    def T_ACCUMULATION_COMPLETE(self):
        return _config._T_ACCUMULATION_COMPLETE

    # make setter for _config._T_ACCUMULATION_COMPLETE  in _config as properties
    @T_ACCUMULATION_COMPLETE.setter
    def T_ACCUMULATION_COMPLETE(self, value):
        raise AttributeError('Cannot set _T_ACCUMULATION_COMPLETE. \
        Use _T_ACCUMULATION_END and _T_END_TO_COMPLETE instead')

    # # make getter for _config._DT  in _config as properties
    # @property
    # def DT(self):
    #     return _config._DT
    #
    # # make setter for _config._DT  in _config as properties
    # @DT.setter
    # def DT(self, value):
    #     _config._DT = value
    #
    # # make getter for _config._EXPECTED_NUM_AUX_CHANNELS
    # # (number of feature idx in the segmented cells list) in _config as properties
    # @property
    # def EXPECTED_NUM_AUX_CHANNELS(self):
    #     return _config._EXPECTED_NUM_AUX_CHANNELS
    #
    # # make setter for _config._EXPECTED_NUM_AUX_CHANNELS in _config as properties
    # @EXPECTED_NUM_AUX_CHANNELS.setter
    # def EXPECTED_NUM_AUX_CHANNELS(self, value):
    #     _config._EXPECTED_NUM_AUX_CHANNELS = value
    #
    # # make getter for _config._DOC_AUX_CHANNEL (feature idx in the segmented cells list) in _config as properties
    # @property
    # def DOC_AUX_CHANNEL(self):
    #     return _config._DOC_AUX_CHANNEL
    #
    # # make setter for _config._DOC_AUX_CHANNEL  in _config as properties
    # @DOC_AUX_CHANNEL.setter
    # def DOC_AUX_CHANNEL(self, value):
    #     _config._DOC_AUX_CHANNEL = value
    #
    # # make getter for _config._USE_CASHED  in _config as properties
    # @property
    # def USE_CASHED(self):
    #     return _config._USE_CASHED
    #
    # # make setter for _config._USE_CASHED  in _config as properties
    # @USE_CASHED.setter
    # def USE_CASHED(self, value):
    #     _config._USE_CASHED = value
    #
    # # make getter for _config._USE_SYNCH_FILES  in _config as properties
    # @property
    # def USE_SYNCH_FILES(self):
    #     return _config._USE_SYNCH_FILES
    #
    # # make setter for _config._USE_SYNCH_FILES  in _config as properties
    # @USE_SYNCH_FILES.setter
    # def USE_SYNCH_FILES(self, value):
    #     _config._USE_SYNCH_FILES = value
    #
    # # make getter for _config._SAVE_IMS  in _config as properties
    # @property
    # def SAVE_IMS(self):
    #     return _config._SAVE_IMS
    #
    # # make setter for _config._SAVE_IMS  in _config as properties
    # @SAVE_IMS.setter
    # def SAVE_IMS(self, value):
    #     _config._SAVE_IMS = value
    #
    # # make getter for _config._SYNC_FILE_START  in _config as properties
    # @property
    # def SYNC_FILE_START(self):
    #     return _config._SYNC_FILE_START
    #
    # # make setter for _config._SYNC_FILE_START  in _config as properties
    # @SYNC_FILE_START.setter
    # def SYNC_FILE_START(self, value):
    #     _config._SYNC_FILE_START = value
    #
    # # make getter for _config._SYNC_FILE_DONE  in _config as properties
    # @property
    # def SYNC_FILE_DONE(self):
    #     return _config._SYNC_FILE_DONE
    #
    # # make setter for _config._SYNC_FILE_DONE  in _config as properties
    # @SYNC_FILE_DONE.setter
    # def SYNC_FILE_DONE(self, value):
    #     _config._SYNC_FILE_DONE = value
    #
    #
    # # make getter for _config._MERGE_CLOSE_CELLS  in _config as properties
    # @property
    # def MERGE_CLOSE_CELLS(self):
    #     return _config._MERGE_CLOSE_CELLS
    #
    # # make setter for _config._MERGE_CLOSE_CELLS  in _config as properties
    # @MERGE_CLOSE_CELLS.setter
    # def MERGE_CLOSE_CELLS(self, value):
    #     _config._MERGE_CLOSE_CELLS = value
    #
    # # make getter for _config._NUM_WORKERS  in _config as properties
    # @property
    # def NUM_WORKERS(self):
    #     return _config._NUM_WORKERS
    #
    # # make setter for _config._NUM_WORKERS  in _config as properties
    # @NUM_WORKERS.setter
    # def NUM_WORKERS(self, value):
    #     _config._NUM_WORKERS = value
    #
    # # make getter for _config._LONG_RUN_PRINT_TIMEOUT  in _config as properties
    # @property
    # def LONG_RUN_PRINT_TIMEOUT(self):
    #     return _config._LONG_RUN_PRINT_TIMEOUT
    #
    # # make setter for _config._LONG_RUN_PRINT_TIMEOUT  in _config as properties
    # @LONG_RUN_PRINT_TIMEOUT.setter
    # def LONG_RUN_PRINT_TIMEOUT(self, value):
    #     _config._LONG_RUN_PRINT_TIMEOUT = value
    #
    # # make getter for _config._VERBOSE_XING_SOLVER  in _config as properties
    # @property
    # def VERBOSE_XING_SOLVER(self):
    #     return _config._VERBOSE_XING_SOLVER
    #
    # # make setter for _config._VERBOSE_XING_SOLVER  in _config as properties
    # @VERBOSE_XING_SOLVER.setter
    # def VERBOSE_XING_SOLVER(self, value):
    #     _config._VERBOSE_XING_SOLVER = value
    #
    # # make getter for _config._IMG_DIRS  in _config as properties
    # @property
    # def IMG_DIRS(self):
    #     return _config._IMG_DIRS
    #
    # # make setter for _config._IMG_DIRS  in _config as properties
    # @IMG_DIRS.setter
    # def IMG_DIRS(self, value):
    #     _config._IMG_DIRS = value
    #
    #
    # # tracking properties
    #
    # # processing constants
    # # make getter for _config._CELL_FUSE_RADIUS  in _config as properties
    # @property
    # def CELL_FUSE_RADIUS(self):
    #     return _config._CELL_FUSE_RADIUS
    #
    # # make setter for _config._CELL_FUSE_RADIUS  in _config as properties
    # @CELL_FUSE_RADIUS.setter
    # def CELL_FUSE_RADIUS(self, value):
    #     _config._CELL_FUSE_RADIUS = value

    # make getter for _config._W_NC_0  in _config as properties
    @property
    def W_NC_0(self):
        return _config._W_NC_0

    # make setter for _config._W_NC_0  in _config as properties
    @W_NC_0.setter
    def W_NC_0(self, value):
        _config._W_NC_0 = value
        _config._W_NC = value
        _config._W_NC_GLOB = value

        self.update_EXPFT_A_Y()

    # # make getter for _config._W_NN  in _config as properties
    # @property
    # def W_NN(self):
    #     return _config._W_NN
    #
    # # make setter for _config._W_NN  in _config as properties
    # @W_NN.setter
    # def W_NN(self, value):
    #     _config._W_NN = value
    #
    # # make getter for _config._W_DM  in _config as properties
    # @property
    # def W_DM(self):
    #     return _config._W_DM
    #
    # # make setter for _config._W_DM  in _config as properties
    # @W_DM.setter
    # def W_DM(self, value):
    #     _config._W_DM = value
    #
    # # make getter for _config._MEAN_V  in _config as properties
    # @property
    # def MEAN_V(self):
    #     return _config._MEAN_V
    #
    # # make setter for _config._MEAN_V  in _config as properties
    # @MEAN_V.setter
    # def MEAN_V(self, value):
    #     _config._MEAN_V = value
    #
    #
    # # make getter for _config._STD_V2  in _config as properties
    # @property
    # def STD_V2(self):
    #     return _config._STD_V2
    #
    # # make setter for _config._STD_V2  in _config as properties
    # @STD_V2.setter
    # def STD_V2(self, value):
    #     _config._STD_V2 = value
    #
    # # make getter for _config._STD_REL_DW2  in _config as properties
    # @property
    # def STD_REL_DW2(self):
    #     return _config._STD_REL_DW2
    #
    # # make setter for _config._STD_REL_DW2  in _config as properties
    # @STD_REL_DW2.setter
    # def STD_REL_DW2(self, value):
    #     _config._STD_REL_DW2 = value
    #
    # # make getter for _config._LINKS_MAX_DR  in _config as properties
    # @property
    # def LINKS_MAX_DR(self):
    #     return _config._LINKS_MAX_DR
    #
    # # make setter for _config._LINKS_MAX_DR  in _config as properties
    # @LINKS_MAX_DR.setter
    # def LINKS_MAX_DR(self, value):
    #     _config._LINKS_MAX_DR = value
    #
    # # make getter for _config._LINKS_EPS  in _config as properties
    # @property
    # def LINKS_EPS(self):
    #     return _config._LINKS_EPS
    #
    # # make setter for _config._LINKS_EPS  in _config as properties
    # @LINKS_EPS.setter
    # def LINKS_EPS(self, value):
    #     _config._LINKS_EPS = value

    # make getter for _config._BORDER_ATT_DIST  in _config as properties
    @property
    def BORDER_ATT_DIST(self):
        return _config._BORDER_ATT_DIST

    # make setter for _config._BORDER_ATT_DIST  in _config as properties
    @BORDER_ATT_DIST.setter
    def BORDER_ATT_DIST(self, value):
        _config._BORDER_ATT_DIST = value
        self.update_EXPFT_A_Y()

    # make getter for _config._CELL_RADIUS  in _config as properties
    @property
    def CELL_RADIUS(self):
        return _config._CELL_RADIUS

    # make setter for _config._CELL_RADIUS  in _config as properties
    @CELL_RADIUS.setter
    def CELL_RADIUS(self, value):
        _config._CELL_RADIUS = value
        self.update_EXPFT_A_Y()

    # make getter for _config._EXPFT_A  in _config as properties
    @property
    def EXPFT_A(self):
        return _config._EXPFT_A

    # make setter for _config._EXPFT_A  in _config as properties
    @EXPFT_A.setter
    def EXPFT_A(self, value):
        raise ValueError('EXPFT_A is a constant dependent on _CELL_RADIUS and _BORDER_ATT_DIST. Set those to update EXPFT_A/Y')

    # make getter for _config._EXPFT_Y  in _config as properties
    @property
    def EXPFT_Y(self):
        return _config._EXPFT_Y

    # make setter for _config._EXPFT_Y  in _config as properties
    @EXPFT_Y.setter
    def EXPFT_Y(self, value):
        raise ValueError('EXPFT_Y is a constant dependent on _CELL_RADIUS \
        and _BORDER_ATT_DIST. Set those to update EXPFT_A/Y')

    # set EXPFT_A and EXPFT_Y
    def update_EXPFT_A_Y(self):
        _config._EXPFT_A = (self.W_NC_0 - 1) / (self.W_NC_0 * (1 - exp(self.CELL_RADIUS - self.BORDER_ATT_DIST)))
        _config._EXPFT_Y = 1 - self.EXPFT_A

    # # make getter for _config._W_NC_LOC  in _config as properties
    # @property
    # def W_NC_LOC(self):
    #     return _config._W_NC_LOC
    #
    # # make setter for _config._W_NC_LOC  in _config as properties
    # @W_NC_LOC.setter
    # def W_NC_LOC(self, value):
    #     _config._W_NC_LOC = value
    #
    #
    # # make getter for _config._W_F_MULT_END_LOC  in _config as properties
    # @property
    # def W_F_MULT_END_LOC(self):
    #     return _config._W_F_MULT_END_LOC
    #
    # # make setter for _config._W_F_MULT_END_LOC  in _config as properties
    # @W_F_MULT_END_LOC.setter
    # def W_F_MULT_END_LOC(self, value):
    #     _config._W_F_MULT_END_LOC = value
    #
    # # make getter for _config._W_F_ABOVE_EST_LOC  in _config as properties
    # @property
    # def W_F_ABOVE_EST_LOC(self):
    #     return _config._W_F_ABOVE_EST_LOC
    #
    # # make setter for _config._W_F_ABOVE_EST_LOC  in _config as properties
    # @W_F_ABOVE_EST_LOC.setter
    # def W_F_ABOVE_EST_LOC(self, value):
    #     _config._W_F_ABOVE_EST_LOC = value
    #
    # # make getter for _config._W_NC_GLOB  in _config as properties
    # @property
    # def W_NC_GLOB(self):
    #     return _config._W_NC_GLOB
    #
    # # make setter for _config._W_NC_GLOB  in _config as properties
    # @W_NC_GLOB.setter
    # def W_NC_GLOB(self, value):
    #     _config._W_NC_GLOB = value
    #
    # # make getter for _config._W_F_MULT_END_GLOB  in _config as properties
    # @property
    # def W_F_MULT_END_GLOB(self):
    #     return _config._W_F_MULT_END_GLOB
    #
    # # make setter for _config._W_F_MULT_END_GLOB  in _config as properties
    # @W_F_MULT_END_GLOB.setter
    # def W_F_MULT_END_GLOB(self, value):
    #     _config._W_F_MULT_END_GLOB = value
    #
    # # make getter for _config._W_F_ABOVE_EST_GLOB  in _config as properties
    # @property
    # def W_F_ABOVE_EST_GLOB(self):
    #     return _config._W_F_ABOVE_EST_GLOB
    #
    # # make setter for _config._W_F_ABOVE_EST_GLOB  in _config as properties
    # @W_F_ABOVE_EST_GLOB.setter
    # def W_F_ABOVE_EST_GLOB(self, value):
    #     _config._W_F_ABOVE_EST_GLOB = value
    #
    # # make getter for _config._W_F_MULT_END_GLOB_SHAVED  in _config as properties
    # @property
    # def W_F_MULT_END_GLOB_SHAVED(self):
    #     return _config._W_F_MULT_END_GLOB_SHAVED
    #
    # # make setter for _config._W_F_MULT_END_GLOB_SHAVED  in _config as properties
    # @W_F_MULT_END_GLOB_SHAVED.setter
    # def W_F_MULT_END_GLOB_SHAVED(self, value):
    #     _config._W_F_MULT_END_GLOB_SHAVED = value
    #
    # # make getter for _config._DRJS_MEAN  in _config as properties
    # @property
    # def DRJS_MEAN(self):
    #     return _config._DRJS_MEAN
    #
    # # make setter for _config._DRJS_MEAN  in _config as properties
    # @DRJS_MEAN.setter
    # def DRJS_MEAN(self, value):
    #     _config._DRJS_MEAN = value
    #
    # # make getter for _config._DRJS_STD  in _config as properties
    # @property
    # def DRJS_STD(self):
    #     return _config._DRJS_STD
    #
    # # make setter for _config._DRJS_STD  in _config as properties
    # @DRJS_STD.setter
    # def DRJS_STD(self, value):
    #     _config._DRJS_STD = value
    #
    # # make getter for _config._DRJF_MEAN  in _config as properties
    # @property
    # def DRJF_MEAN(self):
    #     return _config._DRJF_MEAN
    #
    # # make setter for _config._DRJF_MEAN  in _config as properties
    # @DRJF_MEAN.setter
    # def DRJF_MEAN(self, value):
    #     _config._DRJF_MEAN = value
    #
    # # make getter for _config._DRJF_STD  in _config as properties
    # @property
    # def DRJF_STD(self):
    #     return _config._DRJF_STD
    #
    # # make setter for _config._DRJF_STD  in _config as properties
    # @DRJF_STD.setter
    # def DRJF_STD(self, value):
    #     _config._DRJF_STD = value
    #
    # # make getter for _config._SEARCH_MERGE_VTX_TIDX_RANGE  in _config as properties
    # @property
    # def SEARCH_MERGE_VTX_TIDX_RANGE(self):
    #     return _config._SEARCH_MERGE_VTX_TIDX_RANGE
    #
    # # make setter for _config._SEARCH_MERGE_VTX_TIDX_RANGE  in _config as properties
    # @SEARCH_MERGE_VTX_TIDX_RANGE.setter
    # def SEARCH_MERGE_VTX_TIDX_RANGE(self, value):
    #     _config._SEARCH_MERGE_VTX_TIDX_RANGE = value
    #
    # # make getter for _config._SEARCH_FLOW_VTX_TIDX_RANGE  in _config as properties
    # @property
    # def SEARCH_FLOW_VTX_TIDX_RANGE(self):
    #     return _config._SEARCH_FLOW_VTX_TIDX_RANGE
    #
    # # make setter for _config._SEARCH_FLOW_VTX_TIDX_RANGE  in _config as properties
    # @SEARCH_FLOW_VTX_TIDX_RANGE.setter
    # def SEARCH_FLOW_VTX_TIDX_RANGE(self, value):
    #     _config._SEARCH_FLOW_VTX_TIDX_RANGE = value
    #
    # # make getter for _config._MAX_TRACK_CONN_RAD_OFS  in _config as properties
    # @property
    # def MAX_TRACK_CONN_RAD_OFS(self):
    #     return _config._MAX_TRACK_CONN_RAD_OFS
    #
    # # make setter for _config._MAX_TRACK_CONN_RAD_OFS  in _config as properties
    # @MAX_TRACK_CONN_RAD_OFS.setter
    # def MAX_TRACK_CONN_RAD_OFS(self, value):
    #     _config._MAX_TRACK_CONN_RAD_OFS = value
    #
    # # make getter for _config._MAX_TRACK_CONN_RAD_DT  in _config as properties
    # @property
    # def MAX_TRACK_CONN_RAD_DT(self):
    #     return _config._MAX_TRACK_CONN_RAD_DT
    #
    # # make setter for _config._MAX_TRACK_CONN_RAD_DT  in _config as properties
    # @MAX_TRACK_CONN_RAD_DT.setter
    # def MAX_TRACK_CONN_RAD_DT(self, value):
    #     _config._MAX_TRACK_CONN_RAD_DT = value
    #
    # # make getter for _config._TRACK_PARS_DT_RANGE_SV  in _config as properties
    # @property
    # def TRACK_PARS_DT_RANGE_SV(self):
    #     return _config._TRACK_PARS_DT_RANGE_SV
    #
    # # make setter for _config._TRACK_PARS_DT_RANGE_SV  in _config as properties
    # @TRACK_PARS_DT_RANGE_SV.setter
    # def TRACK_PARS_DT_RANGE_SV(self, value):
    #     _config._TRACK_PARS_DT_RANGE_SV = value
    #
    # # make getter for _config._TRACK_PARS_DT_RANGE_LINV  in _config as properties
    # @property
    # def TRACK_PARS_DT_RANGE_LINV(self):
    #     return _config._TRACK_PARS_DT_RANGE_LINV
    #
    # # make setter for _config._TRACK_PARS_DT_RANGE_LINV  in _config as properties
    # @TRACK_PARS_DT_RANGE_LINV.setter
    # def TRACK_PARS_DT_RANGE_LINV(self, value):
    #     _config._TRACK_PARS_DT_RANGE_LINV = value
    #
    # # make getter for _config._NO_JUMP_DR  in _config as properties
    # @property
    # def NO_JUMP_DR(self):
    #     return _config._NO_JUMP_DR
    #
    # # make setter for _config._NO_JUMP_DR  in _config as properties
    # @NO_JUMP_DR.setter
    # def NO_JUMP_DR(self, value):
    #     _config._NO_JUMP_DR = value
    #
    # # make getter for _config._MAX_JUMP_DX  in _config as properties
    # @property
    # def MAX_JUMP_DX(self):
    #     return _config._MAX_JUMP_DX
    #
    # # make setter for _config._MAX_JUMP_DX  in _config as properties
    # @MAX_JUMP_DX.setter
    # def MAX_JUMP_DX(self, value):
    #     _config._MAX_JUMP_DX = value
    #
    # # make getter for _config._TRACK_DIST_CHI2_MAX_DT  in _config as properties
    # @property
    # def TRACK_DIST_CHI2_MAX_DT(self):
    #     return _config._TRACK_DIST_CHI2_MAX_DT
    #
    # # make setter for _config._TRACK_DIST_CHI2_MAX_DT  in _config as properties
    # @TRACK_DIST_CHI2_MAX_DT.setter
    # def TRACK_DIST_CHI2_MAX_DT(self, value):
    #     _config._TRACK_DIST_CHI2_MAX_DT = value

    # make getter for _config._TRACK_PRIORS  in _config as properties
    @property
    def TRACK_PRIORS(self):
        return _config._TRACK_PRIORS

    # make setter for _config._TRACK_PRIORS  in _config as properties
    @TRACK_PRIORS.setter
    def TRACK_PRIORS(self, value):
        _config._TRACK_PRIORS = value

    def set_TRACK_PRIORS(self, variable, mu_s2population_s2instance):
        _config._TRACK_PRIORS[variable] = mu_s2population_s2instance

    def set_TRACK_PRIORS(self, variable, mu_s2population_s2instance):
        _config._TRACK_PRIORS[variable] = mu_s2population_s2instance

    def get_TRACK_PRIORS(self, variable):
        return _config._TRACK_PRIORS[variable]
