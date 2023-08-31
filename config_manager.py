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
