"""
   Copyright 2015-2025, University of Bern,
    Data Science Lab and Theodor Kocher Institute,
    M. Vladymyrov

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

import os, sys, traceback
from datetime import datetime

from UFMAna import *
from config_manager import ConfigManager as cfg

from time import time as timer


import shutil

from IPython.display import display
from ipywidgets import Layout, HBox, VBox, Text, IntText, Button, Output, HTML, SelectMultiple, Label, Checkbox

from dataclasses import dataclass, asdict
import json
import traceback
import datetime

import pathlib as pl
import pandas as pd


# Configs, params, dbs etc classes
@dataclass
class DataAnalysisConfig:
    seg_ds_path: str = os.path.abspath('../../datasets_seg')
    ds_inf_path: str = os.path.abspath('../../datasets_seg/info.txt')
    ana_ds_path: str = os.path.abspath('../../datasets_ana')
    ds_db_path: str = os.path.abspath('../../datasets_ana/dbs/ds_db.csv')
    ana_db_path: str = os.path.abspath('../../datasets_ana/dbs/ana_db.csv')


# methods for loading and storing the config to file
_cfg_filename = 'ana_cfg.json'


def load_analysis_cfg(cfg_path=None):
    cfg_path = cfg_path or os.path.join(os.path.abspath(os.path.curdir), _cfg_filename)
    if os.path.exists(cfg_path):
        with open(cfg_path, 'rt') as f:
            cfg = DataAnalysisConfig(**json.load(f))
    else:
        cfg = DataAnalysisConfig()
    return cfg


def save_analysis_cfg(cfg, cfg_path=None):
    cfg_path = cfg_path or os.path.join(os.path.abspath(os.path.curdir), _cfg_filename)
    with open(cfg_path, 'wt') as f:
        # human-readable, 4 spaces indentation
        json.dump(asdict(cfg), f, indent=4)


def read_info_file(ds_inf_file_path):
    try:
        with open(ds_inf_file_path, 'rt') as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = []
    ds_inf = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        subs = line.split('-')
        k = int(subs[0].strip())
        v = '-'.join(subs[1:])

        ds_inf[k] = v.strip()
    return ds_inf


def get_ana_db(da_cfg: DataAnalysisConfig):
    ana_db_path = pl.Path(da_cfg.ana_db_path)
    if ana_db_path.exists():
        ana_db_df = pd.read_csv(ana_db_path)
    else:
        ana_db_df = pd.DataFrame(columns=['study_id', 'name', 'experimentor', 'date', 'details', 'info_filename', 'dir'])
        # the study dir  will ba path relative from ana_ds_path (rood of all ana) to the dir
    return ana_db_df


def get_ds_db(da_cfg: DataAnalysisConfig):
    ds_db_path = pl.Path(da_cfg.ds_db_path)
    if ds_db_path.exists():
        ds_db_df = pd.read_csv(ds_db_path)
    else:
        ds_db_df = pd.DataFrame(columns=['ds_id', 'ds_name', 'experimentor', 'date', 'details', 'accumulation_end_frame', 'has_endothelium'])
    return ds_db_df


def sync_analysis_datasets(da_cfg: DataAnalysisConfig):
    # 1. load datasets info
    ds_inf = read_info_file(da_cfg.ds_inf_path)

    # 2. read csv db:
    ds_db_df = get_ds_db(da_cfg)

    # prevent splitting of printed dataframe-s in many lines by coluns:
    pd.set_option('display.max_colwidth', None)


    print(ds_db_df)


    seg_dir = pl.Path(da_cfg.seg_ds_path)
    ana_dir = pl.Path(da_cfg.ana_ds_path)

    # 3. create directory structure (with a star)
    # *`ana_dir`/
    #
    #   *datasets/
    #     001/
    #       tr_cells_tmp.dat
    #     002/
    #       tr_cells_tmp.dat
    #     012/
    #       tr_cells_tmp.dat
    #     xxx/
    #       tr_cells_tmp.dat
    #
    #   *dbs/
    #     ana_db.csv
    #     studies_db.csv
    #     study_info_000.json
    #     study_info_yyy.json
    #
    #   *studies/
    #     000_date_study_name/ # (former "groups_g1_analytics"; actual name with spaces replaced by underscores and all unallowed characters removed ("/\:*?<>|&")
    #      study_info_000.json  # copy of the study info from the db dir
    #     yyy_date_study_name/
    #      study_info_yyy.json

    ana_dir.mkdir(parents=True, exist_ok=True)
    (ana_dir / 'dbs').mkdir(parents=True, exist_ok=True)
    (ana_dir / 'studies').mkdir(parents=True, exist_ok=True)
    ana_datasets_dir = ana_dir / 'datasets'
    ana_datasets_dir.mkdir(parents=True, exist_ok=True)

    print(f'ds_inf:')
    db_updated = False
    for ds_id, ds_name in ds_inf.items():
        print(f'\t {ds_id}: "{ds_name}"', end='')

        # 2. create datasets in the analysis folder if files don't exist and prefill the csv accordingly
        ds_dir_name = f'{ds_id:03d}'
        seg_ds_dir = seg_dir / ds_dir_name
        ana_ds_dir = ana_datasets_dir / ds_dir_name

        cells_file = seg_ds_dir / 'segmentation' / 'cells' / 'tr_cells_tmp.dat'
        tgt_cells_file = ana_ds_dir / 'tr_cells_tmp.dat'

        if not cells_file.exists():
            # print(f' <- ! skipping - no cells file ({str(cells_file)})')
            print()
            continue

        if not tgt_cells_file.exists():
            tgt_cells_file.parent.mkdir(parents=True, exist_ok=True)

            # 3. copy the cells file to the analysis folder
            shutil.copy(cells_file, tgt_cells_file)
            print(f' <- new file copied ', end='')

        # 4. update the db
        if ds_id not in ds_db_df['ds_id'].values:
            ds_db_df = ds_db_df.append({'ds_id': ds_id, 'ds_name': ds_name, 'accumulation_end_frame': 30, 'has_endothelium': 'yes'}, ignore_index=True)

            print(f' <- db record added', end='')
            db_updated = True

        print()

    # 5. save the db
    ds_db_path = pl.Path(da_cfg.ds_db_path)
    ds_db_df.to_csv(ds_db_path, index=False)

    # 6. return if the db was updated
    if db_updated:
        print(f'\n\n_________________________________'
              f'\nUpdated db: {ds_db_path}'
              f'\nPlease fill the missing information in this db file manually before continuing')
    return db_updated


# UI preparation and initialization
_da_cfg: DataAnalysisConfig or None = None


def save_with_widgets():
    """
    Saving widgets state for history hack, reused from
    https://stackoverflow.com/questions/59123005/how-to-save-state-of-ipython-widgets-in-jupyter-notebook-using-python-code
    """
    code = '<script>Jupyter.menubar.actions._actions["widgets:save-with-widgets"].handler()</script>'
    display(HTML(code))


def get_style():
    return HTML('''
    <style>
        .widget-label { min-width: 200px !important; }
    </style>''')


def configure_analysis_gui():
    # 0. Load config
    da_cfg = load_analysis_cfg()

    # 1. Directory path input string "Segmentation datasets path"
    seg_ds_path = Text(value=da_cfg.seg_ds_path, description='Segmentation datasets path:', disabled=False,
                       layout=Layout(width='600px'))

    # 2. File path with datasets IDs list - text human readable file with dataset id - names pairs like " 85 - Untreated_2024.06.18"
    ds_inf_path = Text(value=da_cfg.ds_inf_path, description='Datasets info path:', disabled=False,
                       layout=Layout(width='600px'))

    # 3. Directory path input string "Analysis datasets path"
    ana_ds_path = Text(value=da_cfg.ana_ds_path, description='Analysis datasets path:', disabled=False,
                       layout=Layout(width='600px'))

    # 4. File path "Analysis DB path"
    ds_db_path = Text(value=da_cfg.ds_db_path, description='Datasets DB path:', disabled=False,
                       layout=Layout(width='600px'))

    ana_db_path = Text(value=da_cfg.ana_db_path, description='Analysis DB path:', disabled=False,
                       layout=Layout(width='600px'))

    # 5. Button "Process"
    process_btn = Button(description='Configure and sync', layout=Layout(width='600px'))

    display_styles = get_style()

    # 6. Output - text box with the progress
    out = Output()

    def on_process_click(b):
        with out:
            try:
                print('Configuring...')
                # save config
                da_cfg.seg_ds_path = seg_ds_path.value
                da_cfg.ds_inf_path = ds_inf_path.value
                da_cfg.ana_ds_path = ana_ds_path.value
                da_cfg.ana_db_path = ana_db_path.value
                da_cfg.ds_db_path = ds_db_path.value

                save_analysis_cfg(da_cfg)
                # print('Saved cfg')

                ds_inf = read_info_file(da_cfg.ds_inf_path)

                sync_analysis_datasets(da_cfg)

                save_with_widgets()

                global _da_cfg
                _da_cfg = da_cfg

            except Exception as e:
                print('Error:', e)
                trace_str = traceback.format_exc()
                print(trace_str, flush=True)

    process_btn.on_click(on_process_click)

    controls_b = VBox([
        display_styles,
        seg_ds_path,
        ds_inf_path,
        ana_ds_path,
        ds_db_path,
        ana_db_path,
        process_btn,
        out
    ])

    display(controls_b)


# Study execution interface
class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, msg):
        for s in self.streams:
            s.write(msg)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()


def prep_cfg(accumulation_end=30, start_proc_to_accumulation_end_tf=7):
    __ACCUMULATION_END = accumulation_end
    #start_proc_to_accumulation_end_tf = 7

    __DT_OFFSET = __ACCUMULATION_END - start_proc_to_accumulation_end_tf
    __ACCUMULATION_END = __ACCUMULATION_END - __DT_OFFSET

    print(f'__ACCUMULATION_END={__ACCUMULATION_END}')

    cfgm = cfg()
    cfgm.T_ACCUMULATION_END = __ACCUMULATION_END

    lib_root = os.path.dirname(os.path.abspath(__file__))

    #

    # 1. count the number of occurrences of lib_root in the path cfgm.CLASSIFIER_DET, and first position:
    n_occ = cfgm.CLASSIFIER_DET.count(lib_root)
    pos = None if n_occ == 0 else cfgm.CLASSIFIER_DET.index(lib_root)

    # 2. if >1 - remove all - replace ''
    if pos == 0 and n_occ > 1:
        cfgm.CLASSIFIER_DET = cfgm.CLASSIFIER_DET.replace(lib_root+ '\\', '')
        cfgm.CLASSIFIER_NAC = cfgm.CLASSIFIER_NAC.replace(lib_root+ '\\', '')


    if n_occ != 1:
        cfgm.CLASSIFIER_DET = lib_root + '\\' + cfgm.CLASSIFIER_DET
        cfgm.CLASSIFIER_NAC = lib_root + '\\' + cfgm.CLASSIFIER_NAC

    print(f'T_END_TO_COMPLETE = {cfgm.T_END_TO_COMPLETE}, T_ACCUMULATION_COMPLETE = {cfgm.T_ACCUMULATION_COMPLETE}')

    return cfgm


def perform_study(study_info_fn:str, da_cfg: DataAnalysisConfig):
    """
    Runs tracking and analysis for the study defined in the study_info_fn file.
    Args:
        study_info_fn (str): namAe of the study info file in the dbs folder
        da_cfg (DataAnalysisConfig): config object

    Returns:

    """
    print(f'Performing study {study_info_fn}...')
    study_info_fn_path = pl.Path(da_cfg.ana_ds_path) / 'dbs' / study_info_fn

    study_info: StudyInfo = None
    with open(study_info_fn_path, 'rt') as f:
        study_info = StudyInfo(**json.load(f))

    print(study_info)
    study_name = study_info.study_name
    study_date = study_info.date
    study_id = study_info.study_id

    clean_study_name = get_clean_name(study_name)
    # for i, p in enumerate([study_id, study_date, clean_study_name]):
    #     print(i, p, type(p))
    study_dir_stem = f'{study_id:03d}_{study_date}_{clean_study_name}'

    study_dir = pl.Path(da_cfg.ana_ds_path) / 'studies' / study_dir_stem

    # get conditions, make dic and lsist of all ds_ids

    conditions = study_info.conditions
    conditions = {cond_name: [int(ds_id) for ds_id in ds_ids] for cond_name, ds_ids in conditions}
    all_ds_ids = [ds_id for ds_ids in conditions.values() for ds_id in ds_ids]

    # get the datasets from the db

    ds_db_df = get_ds_db(da_cfg)
    study_datasets = ds_db_df[ds_db_df['ds_id'].isin(all_ds_ids)]

    # get the datasets id and correcponding accumulation_end_frame from the db:
    datasets_ids = study_datasets['ds_id'].values
    datasets_acc_complete_t = study_datasets['accumulation_end_frame'].values
    datasets_acc_complete_t = [int(t) for t in datasets_acc_complete_t]
    datasets_no_transm_mode = ['yes' != v for v in study_datasets['has_endothelium'].values]
    n_proc_advance_tf = study_info.n_proc_advance_tf
    reprocess = study_info.reproc
    skip_processed = not reprocess

    # process the datasets

    condition_id_to_condition_name = {0: 'all_conditions_'+study_name}
    condition_id_to_ds_id = {0:datasets_ids}
    ds_id_to_condition_id = {ds_id:cond_id for cond_id, ds_ids in condition_id_to_ds_id.items() for ds_id in ds_ids}

    plot_dir = study_dir / 'ds_plot'
    datasets_dir = pl.Path(da_cfg.ana_ds_path) / 'datasets'

    std_out_0 = sys.stdout
    std_err_0 = sys.stderr

    study_log_fn = study_dir / f'study.log'
    if study_log_fn.exists():
        # rename old into study_bk_yyyy.mm.dd_hh.mm.ss.log
        study_log_fn_bk = study_dir / f'study_bk_{datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")}.log'
        os.rename(study_log_fn, study_log_fn_bk)

    with open(study_log_fn, 'w') as f_study_log:
        # tee the output to the log file
        sys.stdout = Tee(sys.stdout, f_study_log)
        sys.stderr = Tee(sys.stderr, f_study_log)

        cfgm = prep_cfg(accumulation_end=datasets_acc_complete_t[0], start_proc_to_accumulation_end_tf=n_proc_advance_tf)

        failed_ds = []
        n_att = 1

        for acc_complete_t, no_trans_mode, ds_id in zip(datasets_acc_complete_t, datasets_no_transm_mode, datasets_ids):

            cfgm.T_ACCUMULATION_END = acc_complete_t - cfgm.T_END_TO_COMPLETE

            cfgm.SHAVING_SOLVER_TIMEOUT = 30
            # cfgm.SHAVING_SOLVER_MAX_ITER = 3


            dir_ds_i = pl.Path(da_cfg.ana_ds_path) / 'datasets' / f'{ds_id:03d}'
            track_file =  dir_ds_i / 'st_merged.pckl'

            print(f'\nProcessing dataset {ds_id} ({dir_ds_i}), track file: {track_file} (exists={track_file.exists()}), reprocess={reprocess})')
            if not track_file.exists() or reprocess:
                # we need to run tracking:
                trak_log_fn = dir_ds_i / 'tracking.log'
                if os.path.exists(trak_log_fn):
                    # rename old into tracking_bk_yyyy.mm.dd_hh.mm.ss.log
                    tral_log_fn_bk = dir_ds_i / f'tracking_bk_{datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")}.log'

                    os.rename(trak_log_fn, tral_log_fn_bk)

                with open(trak_log_fn, 'w') as f:
                    # tee the output to the log file
                    std_err_tra_0 = sys.stderr
                    std_out_tra_0 = sys.stdout

                    sys.stdout = Tee(sys.stdout, f)
                    sys.stderr = Tee(sys.stderr, f)

                    track_dataset(ds_root=dir_ds_i)

                    # reset the outputs
                    sys.stdout = std_out_tra_0
                    sys.stderr = std_err_tra_0


            condition_id_to_ds_id_l = {0: [ds_id]}
            condition_id_to_condition_name_l = {0: 'T'}

            ds_id_to_condition_id_l = {ds_id:0}

            for att in range(n_att):
                try:
                    tas_single = process_datasets([ds_id],
                                             condition_id_to_condition_name_l, condition_id_to_ds_id_l, ds_id_to_condition_id_l,
                                             datasets_dir, plot_dir,
                                             use_ds_priors=False,
                                             skip_processed_xing=skip_processed, skip_processed_tr_ana=skip_processed,
                                             no_transm_mode=no_trans_mode # option for Binding assys: True - no transmigration in this case possible.
                                            )
                    break
                except Exception as e:
                    print(f'\nRaised exception in ds_id={ds_id}\n', e)
                    print(traceback.format_exception(None, # <- type(e) by docs, but ignored
                                                     e, e.__traceback__),
                          file=sys.stderr, flush=True)
                print('\n')
            else:
                failed_ds.append(ds_id)

        # read processed and fill the tas_all structure
        filtered_condition_id_to_ds_id = {k: [vi for vi in v if vi not in failed_ds] for k, v in condition_id_to_ds_id.items()}
        tas_all = process_datasets([ids for ids in datasets_ids if ids not in failed_ds],
                                     condition_id_to_condition_name, filtered_condition_id_to_ds_id, ds_id_to_condition_id,
                                     datasets_dir, plot_dir,
                                     use_ds_priors=False,
                                     skip_processed_xing=True, skip_processed_tr_ana=True,
                                     no_transm_mode=no_trans_mode # option for Binding assys: True - no transmigration in this case possible.
                                    )



        # go through conditions and creat   e the condition_id_to_ds_id and comparisson_groups
        condition_id_to_ds_id = {i: ds_ids for i, ds_ids in enumerate(conditions.values())}
        comparisson_groups = {i: {'condition_id': i, 'name': name} for i, name in enumerate(conditions.keys())}
        analyze_group(tas_all, comparisson_groups, condition_id_to_ds_id, study_dir, show=False)

        print(f'Study {study_info_fn} completed')

        # recover back:
        sys.stdout = std_out_0
        sys.stderr = std_err_0


# UI studies
# Analysis GUI:
# 1. make a function creating pannel with all dataset given the ana_db_df
def get_ds_panel(ds_db_df: pd.DataFrame):
    # needs to have a filter textbox and a list of datasets that can be selected with ctrl+click
    formated_ids_dict = {ds_id: f'[{ds_id}] {ds_name}: {ds_exp}, "{ds_date}" ({ds_det})' for ds_id, ds_name, ds_exp, ds_date, ds_det in zip(
        ds_db_df['ds_id'].values,
        ds_db_df['ds_name'].values,
        ds_db_df['experimentor'].values,
        ds_db_df['date'].values,
        ds_db_df['details'].values)
                         }
    filter_text = Text(value='', description='Filter:', disabled=False, layout=Layout(width='600px'))
    datasets_list = SelectMultiple(options=formated_ids_dict.values(), layout=Layout(width='600px', height='430px'))
    style = get_style()

    # handle the filter - on change of the filter text, filter the datasets_list
    def filter_datasets_list(change):
        filter_text_val = filter_text.value
        if not filter_text_val:
            datasets_list.options = formated_ids_dict.values()
        else:
            datasets_list.options = [v for v in formated_ids_dict.values() if filter_text_val.lower() in v.lower()]

    filter_text.observe(filter_datasets_list, names='value')


    ds_panel = VBox([style, filter_text, datasets_list])
    return ds_panel

def get_ds_panel_selection(panel):
    selected_options = panel.children[2].value

    selected_ids = [opt.split(']')[0][1:] for opt in selected_options]

    #make dict of pairs form the selected_ids->selected_options
    seleced_dict = {id:opt for id, opt in zip(selected_ids, selected_options)}

    return seleced_dict


def get_cond_panel_list(panel):
    panel_options = panel.children[2].options

    selected_ids = [opt.split(']')[0][1:] for opt in panel_options]

    #make dict of pairs form the selected_ids->selected_options
    seleced_dict = {id:opt for id, opt in zip(selected_ids, panel_options)}

    return seleced_dict


def get_condition_panel(condition_id, ds_panel, study_datasets, status_bar):
    # given the condition_id and the ds_panel, create a panel with the condition name text field, SelectMultiple (empty) and two buttons below on one row add and remove
    condition_name = Text(value=f'Condition {condition_id}', description='Condition name:', disabled=False, layout=Layout(width='600px'))
    cond_ds_list = SelectMultiple(options=[], layout=Layout(width='600px', height='400px'))
    add_btn = Button(description='Add', layout=Layout(width='300px'))
    rem_btn = Button(description='Remove', layout=Layout(width='300px'))
    style = get_style()

    button_panel = HBox([add_btn, rem_btn])

    cond_panel = VBox([style, condition_name, cond_ds_list, button_panel])

    # on add button click, add the selected datasets from the ds_panel to the cond_ds_list

    def find_overlaps():
        # 1. fill dict condition name->list of ds_ids for all condition panels in study_datasets
        cond_ds_dict = {}
        for cond_panel_i in study_datasets.children[1:]:
            cond_name = cond_panel_i.children[1].value
            cond_ds_dict[cond_name] = list(get_cond_panel_list(cond_panel_i).keys())

        # print to status bar the cond_ds_dict
        status_bar.value = f'{cond_ds_dict}'

        # 2. check for overlaps in the datasets between the conditions
        overlaps = {}
        for i, (cond_name, ds_ids) in enumerate(cond_ds_dict.items()):
            for j, (cond_name_j, ds_ids_j) in enumerate(cond_ds_dict.items()):
                if i >= j:
                    continue
                overlap = set(ds_ids) & set(ds_ids_j)
                if overlap:
                    overlaps[(cond_name, cond_name_j)] = overlap

        # print to status bar the overlaps
        if overlaps:
            # set value in red color
            overlap_info = ''
            for (c1, c2), ds_ids in overlaps.items():
                overlap_info += f'{c1} - {c2}: datasets ids {" ,".join(ds_ids)}; '
            status_bar.value = f'<font color="red">Overlaps between conditions detected:</font>{overlap_info}'

        else:
            status_bar.value = 'OK'

    def on_add_btn_click(b):
        selected_ds = get_ds_panel_selection(ds_panel)
        options = list(cond_ds_list.options) + list(selected_ds.values())
        options = list(set(options))
        cond_ds_list.options = options

        find_overlaps()


    add_btn.on_click(on_add_btn_click)


    # on remove button click, remove the selected datasets from the cond_ds_list
    def on_rem_btn_click(b):
        selected_options = cond_ds_list.value
        cond_ds_list.options = [opt for opt in cond_ds_list.options if opt not in selected_options]
        find_overlaps()

    rem_btn.on_click(on_rem_btn_click)

    return cond_panel


def get_study_ui(da_cfg: DataAnalysisConfig):
    ds_db_df = get_ds_db(da_cfg)

    # create a panel with the study name, user, date, details (text fields) and number of conditions (int selector) all in vbox `study_info`
    # below it a h_box `study_datasets` with ds_panel

    study_name = Text(value='', description='Study name:', disabled=False, layout=Layout(width='600px'))
    user = Text(value='', description='User:', disabled=False, layout=Layout(width='600px'))
    date = Text(value=datetime.datetime.now().strftime('%Y.%m.%d'),
                description='Date:', disabled=False, layout=Layout(width='600px'))
    details = Text(value='', description='Details:', disabled=False, layout=Layout(width='600px'))
    n_conditions = IntText(value=1, description='Number of conditions:', disabled=False, layout=Layout(width='600px'))

    n_processing_advance = IntText(value=7, description='Processing advance (frames):', disabled=False, layout=Layout(width='600px'))
    n_accumulation_end = Label(value='Number of frames before accumulation end (frames): <read from the ds_db table>', description='Number of frames before accumulation end (frames):', disabled=True, layout=Layout(width='600px'))
    reproc_chbx = Checkbox(value=False, description='Reprocess previously completed', disabled=False, layout=Layout(width='600px'))

    proc_config = VBox([n_processing_advance, n_accumulation_end, reproc_chbx])

    style = get_style()

    study_info = VBox([style, study_name, user, date, details, n_conditions])

    # spacer with horizontal line
    spacer_vert = HTML(value='<hr>', description='', layout=Layout(width='100vp', height='20px'))

    ds_panel = get_ds_panel(ds_db_df)

    study_datasets = HBox([ds_panel])

    # add vbox with status bar label for warnigns etc and the Process button
    status_bar = HTML(value='OK', description='Status:', layout=Layout(width='100vp'))
    proc_btn = Button(description='Process', layout=Layout(width='100px'))
    out = Output()
    footer = VBox([style, status_bar, out, proc_btn])

    study_ui = VBox([study_info, proc_config, spacer_vert, study_datasets, spacer_vert, footer])


    # upon change of the n_conditions, add or remove the condition panels to the `study_datasets`

    def on_n_conditions_change(change):
        n_conds = n_conditions.value
        n_children = len(study_datasets.children)
        if n_conds > n_children-1:  # -1 for the ds_panel
            for i in range(n_children-1, n_conds):
                cond_panel = get_condition_panel(i, ds_panel, study_datasets, status_bar)
                study_datasets.children = study_datasets.children + (cond_panel,)
        elif n_conds <= n_children-1:
            study_datasets.children = study_datasets.children[:n_conds+1]

    n_conditions.observe(on_n_conditions_change, names='value')

    # init the condition panels - according to initial n_conditions value
    on_n_conditions_change(None)

    # handle the process button click
    def on_proc_btn_click(b):
        # set button text to "Processing..." and diasable it
        proc_btn.description = 'Processing...'
        proc_btn.disabled = True
        with out:
            try:
                study_info_dict = get_study_info(study_ui)
                # print(study_info)
                # set same to status bar
                status_bar.value = f'Study info: {study_info_dict}'

                save_study_info(study_info_dict, da_cfg)

                # run the analysis printing in output
                print('\n Processing...')
                study_info_fn = f'study_info_{study_info_dict.study_id:03d}.json'
                perform_study(study_info_fn, da_cfg)

            except Exception as e:
                print('Error:', e)
                trace_str = traceback.format_exc()
                print(trace_str, flush=True)
                #status_bar.value = f'<font color="red">Error: {trace_str}</font>'

        # set button text to "Process" and enable it
        proc_btn.description = 'Process'
        proc_btn.disabled = False

    proc_btn.on_click(on_proc_btn_click)

    return study_ui


@dataclass
class StudyInfo:
    study_name: str
    study_id: int
    user: str
    date: str
    details: str
    n_conditions: int
    conditions: list
    n_proc_advance_tf: int
    reproc: bool


def get_study_info(study_ui):
    study_info = study_ui.children[0]
    study_proc = study_ui.children[1]
    study_datasets = study_ui.children[3]

    study_name = study_info.children[1].value
    user = study_info.children[2].value
    date = study_info.children[3].value
    details = study_info.children[4].value
    n_conditions = study_info.children[5].value

    n_proc_advance_tf = study_proc.children[0].value
    reproc = study_proc.children[2].value

    conditions = []
    for cond_panel in study_datasets.children[1:]:
        cond_name = cond_panel.children[1].value
        cond_ds = list(get_cond_panel_list(cond_panel).keys())
        cond_ds = [int(ds_id) for ds_id in cond_ds]
        conditions.append((cond_name, cond_ds))

    study_info_dict = StudyInfo(study_name=study_name, study_id=None,
                                user=user, date=date, details=details,
                                n_conditions=n_conditions, conditions=conditions,
                                n_proc_advance_tf=n_proc_advance_tf,
                                reproc=reproc)

    return study_info_dict


def get_clean_name(name:str):
    replace_chars = "/\:*?<>|&"
    for c in replace_chars:
        name = name.replace(c, '_')
    return name


def save_study_info(study_info:StudyInfo, da_cfg: DataAnalysisConfig):
    """
    Saves the study info to the study db and to the study folder and updates the study_id in the study_info_dict
    Args:
        study_info (StudyInfo): study info dataclass
        da_cfg (DataAnalysisConfig): config object

    Returns:

    """
    # load analysis db
    ana_db_df = get_ana_db(da_cfg)
    # print(ana_db_df)

    # add new study to the db and get id:
    study_id = (int(ana_db_df['study_id'].max()) + 1) if not ana_db_df.empty else 0
    study_info.study_id = study_id

    study_info_fn = f'study_info_{study_id:03d}.json'
    clean_study_name = get_clean_name(study_info.study_name)
    study_dir_stem = f'{study_id:03d}_{study_info.date}_{clean_study_name}'
    study_dir = pl.Path(da_cfg.ana_ds_path) / 'studies' / study_dir_stem
    study_dir.mkdir(parents=True, exist_ok=True)

    study_dir_rel = study_dir.relative_to(pl.Path(da_cfg.ana_ds_path))

    study_info_fn_local = study_dir / study_info_fn

    new_row = {'study_id': study_id,
               'name': study_info.study_name,
               'experimentor': study_info.user,
               'date': study_info.date,
               'details': study_info.details,
               'info_filename': study_info_fn,
               'dir': str(study_dir_rel)
              }

    # print(new_row)

    ana_db_df = ana_db_df.append(new_row, ignore_index=True)

    ana_db_df.to_csv(da_cfg.ana_db_path, index=False)

    study_info_fn_path = pl.Path(da_cfg.ana_db_path).parent / study_info_fn
    study_info_dict = asdict(study_info)
    with open(study_info_fn_path, 'wt') as f:
        json.dump(study_info_dict, f, indent=4)

    # copy to the study dir
    shutil.copy(study_info_fn_path, study_info_fn_local)

    return study_id


def study_ui():
    da_cfg = _da_cfg or load_analysis_cfg()
    study_ui = get_study_ui(da_cfg)
    display(study_ui)