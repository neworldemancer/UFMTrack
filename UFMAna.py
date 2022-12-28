from config_manager import ConfigManager as Cfg

# load libs
import re
import os
from enum import Enum

from itertools import groupby
import pandas as pd

from scipy.ndimage import gaussian_filter1d as gauss_flt1d
from scipy import optimize as opt

from statannotations.Annotator import Annotator

from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages

import seaborn as sns

import roifile
from zipfile import ZipFile

from classificator import Classificator
from GUFMTrack import *

sns.set_theme(font_scale=1.3, palette=sns.color_palette('colorblind'))
cfgm = Cfg()


# # Color palettes

cp_cb = sns.color_palette('colorblind')
cp_p = sns.color_palette('Paired')
cp_s2 = sns.color_palette('Set2')

# 3 types, auto
cols_3t_a = [cp_p[1], cp_p[7], cp_p[5]]
cp_3t_a = sns.color_palette(cols_3t_a)

# 2 types, auto
cols_2t_a = [cp_p[1], cp_p[5]]
cp_2t_a = sns.color_palette(cols_2t_a)

# 2 types, man
cols_2t_m = [cp_p[0], cp_p[4]]
cp_2t_m = sns.color_palette(cols_2t_m)

# 2 types, man and auto
cols_2t_ma = [cp_p[0], cp_p[1], cp_p[4], cp_p[5]]
cp_2t_ma = sns.color_palette(cols_2t_ma)

# 1 Advanced experimentor, 3 experimenters, auto
cols_4exp_1a = [cp_s2[1], cp_s2[7], cp_s2[2], cp_s2[6], cp_s2[4]]
cp_4exp_1a = sns.color_palette(cols_4exp_1a)

# 1 Advanced experimentor, 2 experimenters, auto
cols_3exp_1a = [cp_s2[1], cp_s2[2], cp_s2[6], cp_s2[4]]
cp_3exp_1a = sns.color_palette(cols_3exp_1a)
sns.set_palette(cp_3t_a)


# # General utility methods
def path_safe_str(s):
    return s.replace('$', '').replace('\\', '').replace('/', ' per ')


def sc(show=True):
    if show:
        plt.show()
    plt.close()


def ssc(name, root=None, show=True):
    name = path_safe_str(name)
    p = name if root is None else os.path.join(root, name)

    extensions = ['.png', '.pdf']
    for ext in extensions:
        plt.savefig(p + ext)
    sc(show=show)


plt.sc = sc
plt.ssc = ssc


def max_cont_len(bool_arr):
    """
    Get longest streak of True vals in the boolean array
    """
    sustain_len = [0] + [len([1 for _ in grp]) for val, grp in groupby(bool_arr) if val]
    max_sustain_len = max(sustain_len)
    return max_sustain_len


def mask_short_seq(bool_arr, min_len, min_len_b=1, min_len_e=1):
    """
    Remove short streaks of True values in the boolean array.
    'short':
    < min_len in the middle
    < min_len_b if starts from the beginning of the array
    < min_len_e if reaches the end of the array
    """
    ofs = 0
    n = len(bool_arr)
    masked = np.ones_like(bool_arr).astype(bool)

    for val, grp in groupby(bool_arr):
        grp = [g_i for g_i in grp]
        grp_len = len(grp)

        if val:
            if ofs == 0:
                # in the beginning
                # print('beg l=', l)
                if grp_len < min_len_b:
                    masked[:grp_len] = False
            elif ofs + grp_len == n:
                # print('end l=', l)
                if grp_len < min_len_e:
                    masked[-grp_len:] = False
            else:
                if grp_len < min_len:
                    masked[ofs:ofs + grp_len] = False

        # print(l)
        ofs += grp_len

    bool_arr = bool_arr * masked
    return bool_arr


def close_short_gaps(bool_arr, min_len, min_len_b=1, min_len_e=1):
    """
    Remove short streaks of False values in the boolean array.
    'short':
    < min_len in the middle
    < min_len_b if starts from the beginning of the array
    < min_len_e if reaches the end of the array
    """
    inv = ~bool_arr

    inv_only_long = mask_short_seq(inv, min_len, min_len_b, min_len_e)
    closed = ~inv_only_long

    return closed


def collected_inner_to_dict(d, idxs, tgt_dict, lvl, key_level):
    """
    recursive collector function.
    Rearranges nested dictionary into 2D table where the original keys at level `i`
    are stored in columns `'idx_{i}'`.
    """
    if lvl == key_level:
        for lvl_i, idx in zip(range(key_level), idxs):
            tgt_dict[f'idx_{lvl_i}'].append(idx)

        for k, v in d.items():
            tgt_dict[k].append(v)
    else:
        for d_k, d_v in d.items():
            idxs_k = idxs + [d_k]
            lvl_k = lvl + 1

            collected_inner_to_dict(d_v, idxs_k, tgt_dict, lvl_k, key_level)


def nested_dict_to_dict(d, key_level):
    """
    Rearranges nested dictionary into 2D table where the original keys at level `i`
    are stored in columns `'idx_{i}'`.
    """

    # get inner keys
    keys = []

    d_i = d
    for lvl in range(key_level):
        for k, d_in in d_i.items():
            d_i = d_in
            break
        keys.append(f'idx_{lvl}')
    keys.extend(list(d_i.keys()))
    # print(keys)

    d_flat = {k: [] for k in keys}

    idxs = []
    collected_inner_to_dict(d, idxs, d_flat, 0, key_level)

    return d_flat


def nested_dict_to_df(d, key_level):
    """
    converts nested dictionary data structure into a DataFrame
    key level - at which dataframe keys will be. 0 is top
    """

    d_flat = nested_dict_to_dict(d, key_level)
    return pd.DataFrame(d_flat)


def save_roi_array(x, y, t, fname='_RoiSet.zip', px_x=0.629, px_y=-0.629, names=None):
    """
    saves FIJI-compatible RoI set file from x/y arrays, and the point names if available
    """
    zip_obj = ZipFile(fname, 'w')

    try:
        for i, (x_i, y_i, t_i) in enumerate(zip(x, y, t)):
            p_name = f'p{i:04d}' if names is None else names[i]
            roi_f_name = f'{p_name}.roi'

            x_p = int(x_i / px_x)
            y_p = int(y_i / px_y)
            roi = roifile.ImagejRoi.frompoints(np.array([[x_p, y_p]]), position=t_i)
            roi.roitype = roifile.ROI_TYPE.POINT
            roi.tofile(roi_f_name)
            zip_obj.write(roi_f_name)
            os.remove(roi_f_name)
    except Exception as ex:
        print(ex)
    finally:
        zip_obj.close()


def sort_data_dict(d, key, order):
    a = d[key]
    order_dict = {v: i for i, v in enumerate(order)}
    a_key_idx = [order_dict[v] for v in a]
    sorting_idx = np.argsort(a_key_idx)

    # sorted_a = [a[i] for i in sorting_idx]
    for k in d:
        d_k = d[k]
        d[k] = [d_k[i] for i in sorting_idx]


def rename_data_dict(d, key, rename_map):
    d_k = d[key]
    d[key] = [rename_map[v] if v in rename_map else v for v in d_k]  # if not in the dict - keep as it is


def gen_ds_info(txt_filename):
    d = {}
    with open(txt_filename, 'rt') as f:
        line: str or None = None
        for line in f.readlines():
            line = line.strip()
            if len(line) == 0:
                continue
            s_id, s_desc = re.findall('(.*) - (.*)', line)[0]
            ds_id = int(s_id)
            d[ds_id] = s_desc
    return d


# # Analysis methods
# ## Xing resolving interface

def get_priors_from_datasets(datasets_ids, path='.'):
    """
    Assumes datasets in folders `path/ds_id`
    """

    ds_path_tmpl = os.path.join(path, '%d')

    all_ds_tracks = []
    for ds_id in datasets_ids:
        l_st_merged, l_st_full, l_vtx_xg, l_sgm_xg, l_sgm_1g = load_svs(path=ds_path_tmpl % ds_id)

        tracks = convert_to_xings(stack=l_st_full, vtx_xg=l_vtx_xg, sgm_xg=l_sgm_xg, sgm_1g=l_sgm_1g, get_tracks=True)
        all_ds_tracks.extend(tracks)

    xings = Xings(all_ds_tracks)
    slv = Solver(xings)
    # slv.test()
    slv.fill_priors(min_len=20)
    # print(Track.priors)

    priors = copy.deepcopy(Track.priors)

    return priors


def resolve_track_xings_datasets(datasets_ids, priors, path='.', skip_processed=True):
    ds_path_tmpl = os.path.join(path, '%d')

    resolved_tracks = {}

    use_priors = priors is not None
    if use_priors:
        Track.priors = copy.deepcopy(priors)

    skipped_ids = []
    for ds_id in datasets_ids:
        t_s = timer()
        tracks_filename = os.path.join(ds_path_tmpl % ds_id, 'merged_tracks.pckl')

        if skip_processed and os.path.exists(tracks_filename):
            tracks = load_pckl(tracks_filename)
            skipped_ids.append(ds_id)
        else:
            l_st_merged, l_st_full, l_vtx_xg, l_sgm_xg, l_sgm_1g = load_svs(path=ds_path_tmpl % ds_id)
            xing2 = convert_to_xings(stack=l_st_full, vtx_xg=l_vtx_xg, sgm_xg=l_sgm_xg, sgm_1g=l_sgm_1g)
            slv = Solver(xing2)
            slv.solve(plot_state,
                      lap_only=True, priors_set=use_priors,
                      pm_mtr=None,
                      stop_after_init=False)  # , orig_tr_idx_map=tr_idx_map, orig_mu_sgm=tr_mu_sgm

            tracks = slv.tracks

            for track in tracks:
                track.set_ds_id(ds_id)

                # fill in_fid_vol attribute according to the set boundary for current xing2
                _ = track.contained_in_fiducial_volume()

            save_pckl(tracks, tracks_filename)

        resolved_tracks[ds_id] = tracks

        t_e = timer()
        print(f'DS {ds_id} merging, {(t_e - t_s):.1f} sec')

    return resolved_tracks, skipped_ids


# ## Track selection

def filter_tracks_by_length(tracks, min_len):
    return {ds_id: [t for t in trx if t.get_num_nodes() >= min_len] for ds_id, trx in tracks.items()}


def filter_tracks_by_start_time_before(tracks, start_time):
    return {ds_id: [t for t in trx if t.get_first_time() < start_time] for ds_id, trx in tracks.items()}


def filter_tracks_in_fid_area(tracks):
    return {ds_id: [t for t in trx if t.in_fid_vol] for ds_id, trx in tracks.items()}


def filter_ok_long_tracks(resolved_tracks, n_nodes_ok=6, n_nodes_long=30):
    """
    collect reasonable tracks: not NC, >6 (`existing_tracks`) & >30 (`long_tracks`), plot info
    all tr # / in FidVol  ; long tr # / in fid Vol
    """
    ok_tracks = filter_tracks_by_length(resolved_tracks, min_len=n_nodes_ok)
    long_tracks = filter_tracks_by_length(resolved_tracks, min_len=n_nodes_long)

    print(f'ds_id:\t #tr{n_nodes_ok} ; #tr{n_nodes_long}')
    for ds_id in resolved_tracks:
        print(ds_id, f':\t {len(ok_tracks[ds_id])}; {len(long_tracks[ds_id])}')

    return ok_tracks, long_tracks


def filter_fv_physflow_tracks(ok_tracks, long_tracks):
    # track selection for analysis
    long_tracks_fv = filter_tracks_in_fid_area(long_tracks)
    all_tracks_fv = filter_tracks_in_fid_area(ok_tracks)

    long_tracks_phf = filter_tracks_by_start_time_before(long_tracks, cfgm.T_ACCUMULATION_COMPLETE)
    long_tracks_fv_phf = filter_tracks_in_fid_area(long_tracks_phf)

    all_tracks_phf = filter_tracks_by_start_time_before(ok_tracks, cfgm.T_ACCUMULATION_COMPLETE)
    all_tracks_fv_phf = filter_tracks_in_fid_area(all_tracks_phf)

    return long_tracks_fv, all_tracks_fv, long_tracks_phf, long_tracks_fv_phf, all_tracks_phf, all_tracks_fv_phf


def get_total_tracks_num(resolved_tracks):
    return np.sum([len(trx) for trx in resolved_tracks.values()])


# ## Crude distributions plots fn

def plot_dataset_speed_distributions(tracks, save_dir='.\\ds_plot', show=False):
    # speed for all ds
    all_v_long = []
    for ds_id, trx in tracks.items():
        av = []
        if len(trx) == 0:
            continue
        for t in trx:
            a = t.get_mean_std(bayesian_estimators=False)
            sqrt_v = a['s_move_v'][0]
            v = sqrt_v ** 2
            av.append(v)

        av = np.array(av)
        av *= 60 / cfgm.DT
        _ = plt.hist(av, 50)
        plt.title('ds_%d' % ds_id)
        plt.xlabel(r'v, $\mu m/min$')
        plt.ssc(f'crude_speed_dist{ds_id}', root=save_dir, show=show)
        all_v_long.extend(av)

    _ = plt.hist(all_v_long, 50)
    plt.title('all datasets')
    plt.xlabel(r'v, $\mu m/min$')
    plt.ssc(f'crude_speed_dist_all', root=save_dir, show=show)


def plot_tracks_arr(tracks_arr, title, show_line=False, t_max=180,
                    from_cnt=False, col='time',
                    save_dir='.\\ds_plot', show=False):
    f = 5
    if from_cnt:
        fig = plt.figure(figsize=(10 * f, 10 * f))
    else:
        fig = plt.figure(figsize=(14.1 * f, 10.7 * f))

    x_all = []
    y_all = []
    t_all = []
    d_all = []
    e_all = []
    f_all = []
    for tr in tracks_arr:
        x = []
        y = []
        t = []
        d = []
        e = []
        f = []

        t_0 = None
        x_0 = 0
        y_0 = 0
        f_0 = tr.in_fid_vol

        for seg in tr.segments:
            for time, node in zip(seg.times, seg.nodes):
                xi, yi = node.r[:2]
                x.append(xi)
                y.append(yi)
                t.append(time)
                d.append(node.pars[cfgm.DOC_AUX_CHANNEL])
                e.append(node.eccentr)
                if t_0 is None or t_0 > time:
                    t_0, x_0, y_0 = time, xi, yi
                f.append(f_0)
        x = np.array(x)
        y = np.array(y)
        if from_cnt:
            x -= x_0
            y -= y_0

        if show_line:
            plt.plot(x, y, color='k', alpha=0.12)

        x_all.extend(x)
        y_all.extend(y)
        t_all.extend(t)
        d_all.extend(d)
        e_all.extend(e)
        f_all.extend(f)

    t_all = np.array(t_all)
    d_all = np.array(d_all)
    e_all = np.array(e_all)
    f_all = np.array(f_all)

    color = 0

    if col == 'time':
        color = t_all / t_max
    elif col == 'ecc':
        color = e_all / 5.
    elif col == 'fid':
        color = f_all / 1.
    else:
        color = d_all / 255.

    plt.scatter(x_all, y_all, color=cm.rainbow(color), marker='.')

    plt.grid(False)
    if title is not None:
        plt.title(title, fontdict={'fontsize': 44, 'weight': 'bold'})

    fig.axes[0].set_aspect(1.0)

    if from_cnt:
        plt.xlabel(r'x displacement, $\mu m$', fontdict={'fontsize': 40})
        plt.ylabel(r'y displacement, $\mu m$', fontdict={'fontsize': 40})
        plt.xlim(-500, 500)
        plt.ylim(-500, 500)
    else:
        plt.xlabel(r'x, $\mu m$', fontdict={'fontsize': 40})
        plt.ylabel(r'y, $\mu m$', fontdict={'fontsize': 40})

    plt.xticks(fontsize=36)
    plt.yticks(fontsize=36)

    plt.ssc(title, root=save_dir, show=show)


def plot_tracks_datasets_all_vis(tracks, save_dir='.\\ds_plot', show=False):
    for ds_id, trx in tracks.items():
        if len(trx) == 0:
            continue
        plot_tracks_arr(trx, title='tracks_fov_time_ds_%d' % ds_id, show_line=True, save_dir=save_dir,
                        show=show)  # 2D FoV time color
        plot_tracks_arr(trx, title='tracks_fov_diap_ds_%d' % ds_id, col='diap', save_dir=save_dir,
                        show=show)  # 2D FoV diap color
        plot_tracks_arr(trx, title='tracks_fov_ecc_%d' % ds_id, col='ecc', save_dir=save_dir,
                        show=show)  # 2D FoV ecc color
        plot_tracks_arr(trx, title='tracks_fov_fid_ds_%d' % ds_id, col='fid', save_dir=save_dir,
                        show=show)  # 2D FoV in fiducial area color

        plot_tracks_arr(trx, title='tracks_ds_cent_time_%d' % ds_id, from_cnt=True, show_line=True, save_dir=save_dir,
                        show=show)  # 2D center time color
        plot_tracks_arr(trx, title='tracks_ds_cent_diap_%d' % ds_id, from_cnt=True, show_line=True, col='diap',
                        save_dir=save_dir, show=show)  # 2D center diap color


# type to id mapping
def get_cond_id(ds_id, ds_id_to_condition_id):
    return ds_id_to_condition_id.get(ds_id, -1)


def get_cond_name(ds_id, ds_id_to_condition_id, condition_id_to_condition_name):
    return condition_id_to_condition_name.get(get_cond_id(ds_id, ds_id_to_condition_id), 'NotFound')


def plot_tracks_att_det_distribution(tracks_arr, title, t_max=200, save_dir='.\\ds_plot', show=False):
    tr_starts = [t.get_first_time() for t in tracks_arr]
    tr_ends = [t.get_last_time() for t in tracks_arr]

    n_bins = (t_max // 2 if t_max > 100 else t_max) + 1
    bins = np.linspace(0, t_max, n_bins)
    plt.figure(figsize=(20, 4))
    plt.grid(True)

    plt.hist(tr_starts, bins, label='attachment/start')
    plt.hist(tr_ends, bins, label='detachment/end')

    plt.xlabel('timeframe')
    plt.ylabel('count')

    plt.title(title)
    plt.legend()
    plt.ssc(title, root=save_dir, show=show)


def plot_datasets_att_det_distribution(ds_id_to_condition_id, condition_id_to_condition_name,
                                       tracks, t_max=200, save_dir='.\\ds_plot', show=False):
    for ds_id, trx in tracks.items():
        if len(trx) == 0:
            continue
        cond_name = get_cond_name(ds_id, ds_id_to_condition_id, condition_id_to_condition_name)
        plot_tracks_att_det_distribution(tracks_arr=trx,
                                         title=f'att_det_dist_ds_{ds_id}_{cond_name}_t(0-{t_max})',
                                         save_dir=save_dir, show=show, t_max=t_max)


def plot_groups_att_det_distribution(condition_id_to_condition_name, condition_id_to_ds_id,
                                     tracks, t_max=200, save_dir='.\\ds_plot', show=False):
    # attachment and detachment, all tracks, per type
    # existing_tracks - all>6 nodes

    n_bins = (t_max // 2 if t_max > 100 else t_max) + 1

    plt.rcParams.update({'axes.labelsize': 30})
    plt.rcParams.update({'xtick.labelsize': 24})
    plt.rcParams.update({'ytick.labelsize': 24})
    plt.rcParams.update({'legend.fontsize': 24})
    plt.rcParams.update({'axes.titlesize': 30})

    for cond_id, cond_name in condition_id_to_condition_name.items():
        ds_ids = condition_id_to_ds_id[cond_id]

        tr_starts = [t.get_first_time() * cfgm.DT / 60 for ds_idx in ds_ids for t in tracks[ds_idx]]
        tr_ends = [t.get_last_time() * cfgm.DT / 60 for ds_idx in ds_ids for t in tracks[ds_idx]]

        bins = np.linspace(0, t_max * cfgm.DT / 60, n_bins)
        plt.figure(figsize=(16, 7), facecolor='white', dpi=100)

        plt.hist(tr_starts, bins, label='Track start time')
        plt.hist(tr_ends, bins, label='Track end time')

        plt.xlabel('time, min')
        plt.ylabel('count')

        plt.legend()
        plt.title(cond_name)

        plt.ssc(f'detachment_{cond_name}_t(0-{t_max})', root=save_dir, show=show)

    plt.rcParams.update({'axes.labelsize': 'large'})
    plt.rcParams.update({'legend.fontsize': 14.3})
    plt.rcParams.update({'xtick.labelsize': 'medium'})
    plt.rcParams.update({'ytick.labelsize': 'medium'})
    plt.rcParams.update({'axes.titlesize': 'large'})


def plot_phf_detachment_rate(condition_id_to_condition_name, condition_id_to_ds_id,
                             tracks, save_dir='.\\ds_plot', show=False):
    # detachment rate per condition: fraction detached till tf 35 of all attached tracks (in fv)

    # attachment and detachment, all tracks, per type, zoom
    # existing_tracks - all>6 nodes

    detached_fraction = {}
    detached_fraction_arr = {}
    t_acc_end = cfgm.T_ACCUMULATION_END
    t_acc_complete = cfgm.T_ACCUMULATION_COMPLETE

    for cond_id, cond_name in condition_id_to_condition_name.items():
        ds_ids = condition_id_to_ds_id[cond_id]

        tr_starts = [np.array([t.get_first_time() for t in tracks[ds_idx]]) for ds_idx in ds_ids]
        tr_ends = [np.array([t.get_last_time() for t in tracks[ds_idx]]) for ds_idx in ds_ids]

        mask_att_before_acc_end = [tr_starts_i <= t_acc_end for tr_starts_i in tr_starts]
        mask_det_before_acc_com = [tr_ends_i <= t_acc_complete for tr_ends_i in tr_ends]

        n_attached_before_acc_end = [len(tr_starts_i[m_e]) for tr_starts_i, m_e in
                                     zip(tr_starts, mask_att_before_acc_end)]
        n_detached_before_acc_com = [len(tr_ends_i[m_e * m_c])
                                     for (tr_ends_i, m_e, m_c) in zip(tr_ends,
                                                                      mask_att_before_acc_end,
                                                                      mask_det_before_acc_com)
                                     ]

        fractions = [det / att for att, det in zip(n_attached_before_acc_end, n_detached_before_acc_com) if att > 0]
        fraction = np.sum(n_detached_before_acc_com) / np.sum(n_attached_before_acc_end)

        detached_fraction[cond_name] = fraction * 100
        detached_fraction_arr[cond_name] = [fi * 100 for fi in fractions]

    plot_dict = {'type': [], 'det_frac': []}

    for t, arr in detached_fraction_arr.items():
        for ai in arr:
            plot_dict['type'].append(t)
            plot_dict['det_frac'].append(ai)

    plot_dict = pd.DataFrame(plot_dict)

    plt.figure(figsize=(10, 7))
    sns.set(style="whitegrid")

    ax = sns.barplot(x='type',
                     y='det_frac',
                     data=plot_dict, ci='sd')

    types = list(set(plot_dict['type']))
    pairs = [(p1, p2) for i, p1 in enumerate(types[:-1]) for j, p2 in enumerate(types[i + 1:])]

    annotator = Annotator(plt.gca(), pairs, data=plot_dict, x='type', y='det_frac', )
    annotator.configure(test='t-test_ind', text_format='star',
                        loc='inside')
    annotator.apply_and_annotate()

    ax.set_title(
        f'Fraction of cells detached after flow increase (<=tf{t_acc_complete}),\
         of all attached (<= tf {t_acc_end}), %')

    plt.ssc(name=f'PHF_detachment_rate_ds_types_({"_".join(types[:3])})', root=save_dir, show=show)


def plot_neighbour_tracks_info(tas, save_dir='.\\ds_plot', filename='track_neighbour_info', show=False):
    n_plots_per_page = 5
    n_plots = len(tas)
    keys = list(tas.keys())

    fn = os.path.join(save_dir, filename + f'_{keys}.pdf')
    with PdfPages(fn) as pdf:
        for page in range((n_plots + n_plots_per_page - 1) // n_plots_per_page):
            page_keys = keys[page * n_plots_per_page: (page + 1) * n_plots_per_page]

            n_col = 2
            n_row = len(page_keys)

            fig, ax = plt.subplots(nrows=n_row, ncols=n_col, figsize=(16, 24 / 5 * n_row), dpi=100)
            if n_row == 1:
                ax = [ax]

            for axi, k in zip(ax, page_keys):
                n_neigh_max = [np.max(ta.neigh_count) for ta in tas[k]]
                n_neigh_mean = [np.mean(ta.neigh_count) for ta in tas[k]]

                for max_n, lbl, n_neigh, axij in zip([10, 5], ['max neigh', 'mean neigh'], [n_neigh_max, n_neigh_mean],
                                                     axi):
                    axij.hist(n_neigh, np.linspace(0, max_n, 11))
                    axij.text(max_n / 2, axij.get_ylim()[1] * 0.8, f'mean = {np.mean(n_neigh):.2f}')
                    axij.set_title(f'ds_id: {k} {lbl}')

            pdf.savefig(fig)
            if show:
                plt.show()
            plt.close()


# ## Analysis classes

class NeighborhoodChecker:
    """
    Helper class for cheching number of track neighbour tracks.
    If track doesnt have noced at some time-points - those are interpolated between the closes present points
    """

    class TrackNode:
        def __init__(self, x, y, t_idx, tr_idx, interpolated):
            self.x = x
            self.y = y
            self.t_idx = t_idx
            self.tr_idx = tr_idx
            self.interpolated = interpolated

        def r2(self, x, y):
            return (self.x - x) ** 2 + (self.y - y) ** 2

    def __init__(self, neighbour_dist_max):
        self.neighbour_dist_max = neighbour_dist_max
        self.neighbour_dist_max2 = neighbour_dist_max ** 2
        self.containers = {}

    @staticmethod
    def get_nodes_between(n1, n2):
        n_nodes = n2.t_idx - n1.t_idx + 1
        assert (n1.tr_idx == n2.tr_idx)

        tr_idx = n1.tr_idx
        x_int = np.linspace(n1.x, n2.x, n_nodes)
        y_int = np.linspace(n1.y, n2.y, n_nodes)
        t_int = np.linspace(n1.t_idx, n2.t_idx, n_nodes).astype(int)

        t_nodes = {t_int[i]: NeighborhoodChecker.TrackNode(x_int[i], y_int[i], t_int[i], tr_idx, interpolated=True)
                   for i in range(1, n_nodes - 1)}

        return t_nodes

    def get_track_nodes(self, track, tr_idx):
        """
        Return dictionary t->node for all nodes between first and last
        """
        # all_x = []
        # all_y = []
        # all_times = []

        last_t = None
        t_nodes = {}
        for seg in track.segments:
            if seg.is_nc:
                continue
            for n, t in zip(seg.nodes, seg.times):
                x, y, _ = n.r

                curr_node = NeighborhoodChecker.TrackNode(x, y, t, tr_idx, interpolated=False)

                if last_t is not None:
                    if t > last_t + 1:
                        last_node = t_nodes[last_t]

                        int_nodes = self.get_nodes_between(last_node, curr_node)
                        for t_i, n_i in int_nodes.items():
                            t_nodes[t_i] = n_i

                t_nodes[t] = curr_node
                last_t = t

        return t_nodes

    def fill_container_points(self, tracks):
        if len(tracks) == 0:
            return
        t_min = np.min([tr.get_first_time() for tr in tracks])
        t_max = np.max([tr.get_last_time() for tr in tracks])

        node_lists = {t: [] for t in range(t_min, t_max + 1)}
        for tr_idx, tr in enumerate(tracks):
            t_nodes_dict = self.get_track_nodes(tr, tr_idx)
            for t, node in t_nodes_dict.items():
                node_lists[t].append(node)

        # print(node_lists)
        self.containers = {t: ObjContainer2D(nodes, self.neighbour_dist_max) for t, nodes in node_lists.items()}

    def get_n_neighbours(self, x, y, t):
        container: ObjContainer2D = self.containers[t]
        nodes = container.obj_around((x, y), self.neighbour_dist_max)
        n_tracks = np.sum([n.r2(x, y) < self.neighbour_dist_max2 for n in nodes])

        return n_tracks - 1  # w/o self


def do_residuals(pars, dx, dt):
    s, tau = pars

    exp_term = np.exp(-dt / tau) if tau > 0 else 0.
    dx2 = np.abs(2 * s ** 2 * tau * (dt - tau * (1 - exp_term)))

    res = dx - np.sqrt(dx2)
    return res.flatten()


def do_jacobian_res(pars, dx, dt):
    s, tau = pars
    exp_term = np.exp(-dt / tau) if tau > 0 else 0.

    dx2 = np.abs(2 * s ** 2 * tau * (dt - tau * (1 - exp_term)))
    dx2 = dx2.clip(1.e-12)
    sdx2 = np.sqrt(dx2)

    # for this loss df/dp = (1-dx/sdx2) * d (dx2) / dp = f * d (dx2) / dp:
    f = -1. / (2. * sdx2)

    dl_ds = f * 4 * s * tau * (dt - tau * (1 - exp_term)) - (1.e-3 if s == 0 else 0.)
    dl_dtau = f * 2 * s ** 2 * (dt * (1 + exp_term) - 2 * tau * (1 - exp_term)) - (1.e2 if tau <= 0 else 0.)

    dl_ds = dl_ds.flatten()
    dl_dtau = dl_dtau.flatten()

    jacobian = np.stack((dl_ds, dl_dtau), axis=1)

    return jacobian


def do_residuals2(pars, dx2, dt):
    s, tau = pars

    exp_term = np.exp(-dt / tau) if tau > 0 else 0.
    dx2_est = np.abs(2 * s ** 2 * tau * (dt - tau * (1 - exp_term)))

    res = dx2 - dx2_est
    return res.flatten()


def do_jacobian_res2(pars, dx2, dt):
    s, tau = pars
    exp_term = np.exp(-dt / tau) if tau > 0 else 0.

    # dx2_est = np.abs(2 * s ** 2 * tau * (dt - tau * (1 - exp_term)))
    # dx2_est = dx2_est.clip(1.e-12)

    f = -1
    dl_ds = f * 4 * s * tau * (dt - tau * (1 - exp_term)) - (1.e-3 if s == 0 else 0.)
    dl_dtau = f * 2 * s ** 2 * (dt * (1 + exp_term) - 2 * tau * (1 - exp_term)) - (1.e2 if tau <= 0 else 0.)

    dl_ds = dl_ds.flatten()
    dl_dtau = dl_dtau.flatten()

    jacobian = np.stack((dl_ds, dl_dtau), axis=1)

    return jacobian


def get_diff_persistance(dr_arr, dt_arr, s0=1., tau0=1.):
    """
    Performs Dunn-Othmer fit of the cells' displacement:
    dx2 = 2 * s**2 * tau * (dt - tau * (1-exp_term))

    Returns:
        (s, tau): fit parameters
    """

    p0 = [s0, tau0]

    res = opt.least_squares(do_residuals2, p0,
                            bounds=([1.e-6, 1.e-6], [1e2, 1e2]),
                            jac=do_jacobian_res2,
                            args=(dr_arr ** 2, dt_arr))
    s, tau = res.x
    assert s > 0 and tau > 0

    return s, tau


class TrackAnalyzer:
    _diapedesis_profile = {}  # list of t->[d/c]
    _diapedesis_error_t = set()  # set of times when abnormal diapedesis has occured

    _thres_part_diap = 0.3
    _min_part_diap_len = 3
    _thres_comp_diap = 0.75
    _min_comp_diap_len = 4
    _min_comp_diap_e_len = 2

    # thres_min_mean_comp_diap = 0.4
    # if mean DoC from diapedesis start till track end above this value - consider diapedesis

    # thres_min_sustain_comp_diap = 0.95  # if DoC reaches this value sustaining for at least ...
    # thres_min_sustain_len_comp_diap = 10  # this many tf-s - consider full diapedesis

    @staticmethod
    def reset():
        TrackAnalyzer._diapedesis_profile = {}
        TrackAnalyzer._diapedesis_error_t = set()

    def __init__(self, track: Track, tf_min=cfgm.T_ACCUMULATION_COMPLETE,
                 tf_max=189, n_min=6,
                 max_displ_dt=20,
                 crawling_speed_min=0.15,  # per tf, i.e. 0.15um/10s * 6(10s/min)=0.9um
                 crawling_speed_max=6,  # per tf, i.e.    6um/10s * 6(10s/min)=36 um
                 min_coverage_fraction=0.75,  # at least this frac of time should have nodes
                 classificator_det=None,
                 classificator_not_a_cell=None,
                 neighbourhood_checker=None,
                 probing_max_displ=20,  # um
                 jump_displ_thres=8,  # um
                 forse_ta=False
                 ):
        if (track.__class__ is not TrackAnalyzer and not forse_ta) and track.is_nc():
            raise ValueError('Pure NC tracks not accepted here')

        self.ds_id = -1
        # cell sequence
        self.all_nodes = []  # in valid time range
        self.all_times = []  # in valid time range
        self.steps = []  # element - ((n1, n2), dt, (dx,dy,dz), dr, v)
        self.neigh_count = None
        self.mask_run = None

        # set to false if track has no run, or has to be discarded due
        # to observed potential inefficiency, inconsistency, etc
        self.is_ok = True
        self.not_a_cell = False

        # proc params
        self.neighbourhood_checker = neighbourhood_checker
        self.classificator_det = classificator_det
        self.classificator_not_a_cell = classificator_not_a_cell
        self.probing_max_displ = probing_max_displ
        self.jump_displ_thres = jump_displ_thres

        self.tf_min = tf_min
        self.tf_max = tf_max
        self.min_coverage_fraction = min_coverage_fraction

        self.n_min = n_min
        self.max_displ_dt = max_displ_dt  # see dr*_dt
        self.crawling_speed_min = crawling_speed_min  # um / 1 timeframes
        self.crawling_speed_max = crawling_speed_max  # um / 1 timeframes

        # intermediate values
        self.c = []  # normalozed (0-1) mean cell prob
        self.d = []  # normalozed (0-1) mean diap prob
        self.dc = []  # normalozed (0-1) mean (diap prob)/(cell prob)
        self.f_dc = []  # LP filtered normalozed (0-1) mean (diap prob)/(cell prob)

        self.ending_info = None

        # masks
        self.mask_above = None
        self.mask_full_diap = None
        self.mask_any_diap = None
        self.mask_udiap = None  # unknown, but not full

        self.mask_dir_tdiap = None  # transient direct
        self.mask_rev_tdiap = None  # transient reversed
        self.mask_pdiap = None  # partial diap - retracts to above
        self.mask_pre_diap = None

        self.mask_jumps = None
        self.mask_ineff = None

        self.mask_steps_jumps = None
        self.mask_steps_ineff = None

        self.mask_steps_segment_crawling_above = None
        self.mask_steps_segment_crawling_below = None

        self.mask_steps_segment_probing_above = None
        self.mask_steps_segment_probing_below = None

        self.field_max_segment_displacement_above = None
        self.field_max_segment_displacement_below = None

        self.horizon_path = None
        self.horizon_dr = None
        self.horizon_f_path = None  # LP filtered
        self.horizon_f_dr = None  # LP filtered

        # time values
        self.fdiap_start_idxs = []  # start in
        self.fdiap_done_idxs = []  # fully in
        self.pdiap_start_idxs = []  # start out
        self.pdiap_done_idxs = []  # fully out
        self.rdiap_start_idxs = []  # start out
        self.rdiap_done_idxs = []  # fully out
        self.first_diap_start_idxs = []

        self.first_full_diap_tf = None  # after tf_min
        self.detached_tf = None  # after tf_min

        # flags
        self.performed_run = False  # any(mask_run)
        self.performed_fdiap = False  # full
        self.performed_pdiap = False  # partial (above->non-full-diap->above)
        self.performed_rdiap = False  # reversed
        self.performed_any_diap = False  # any
        self.performed_no_diap = False  # did not diapedese at all

        self.performed_detachment = False

        self.detected_diap_ineff = False
        self.detected_tracking_ineff = False

        self.performed_jumps = False
        self.performed_crawling = False  # only before first diap conted here.

        self.performed_probing = False
        self.performed_probing_below = False

        # full diap stats
        self.fdiap_fraction = 0
        self.fdiap_max = 0
        self.fdiap_integral = 0
        self.fdiap_num_tf = 0
        self.fdiap_frac_tf = 0

        # partial diap stats
        self.pdiap_fraction = 0
        self.pdiap_max = 0
        self.pdiap_integral = 0
        self.pdiap_num_tf = 0
        self.pdiap_frac_tf = 0

        # any diap stats
        self.adiap_fraction = 0
        self.adiap_max = 0
        self.adiap_integral = 0
        self.adiap_num_tf = 0
        self.adiap_frac_tf = 0

        self.v_nj_mean = None  # no jump
        self.sigma_v_nj = None
        self.n_v_nj = None

        self.v_nd_mean = None  # no diapedesis and no jump
        self.sigma_v_nd = None
        self.n_v_nd = None

        self.jumps_dr3_dt = []  # jumps info

        # displacement metrics
        self.mtr_crawling_before_diap = None  # main metric
        self.mtr_before_diap = None
        self.mtr_all_crawling = None
        self.mtr_all_probing = None
        self.mtr_all_fdiap = None
        self.mtr_all_valid = None
        self.mtr_all_run = None

        self.fill_nodes(track=track, forse_ta=forse_ta)
        self.fill_steps()

    def fill_nodes(self, track: Track, forse_ta):
        self.mask_run = []

        if track.__class__ == TrackAnalyzer or forse_ta:
            try:
                self.ds_id = track.ds_id
            except Exception as ex:
                self.ds_id = -1  # not implemented in older version

            for n, t in zip(track.all_nodes, track.all_times):
                if t > self.tf_max:
                    continue
                self.mask_run.append(t >= self.tf_min)
                self.all_nodes.append(n)
                self.all_times.append(t)
        else:
            self.ds_id = track.ds_id
            for seg in track.segments:
                if seg.is_nc:
                    continue
                for n, t in zip(seg.nodes, seg.times):
                    if t > self.tf_max:
                        continue
                    self.mask_run.append(t >= self.tf_min)
                    self.all_nodes.append(n)
                    self.all_times.append(t)

        # check is ok
        self.mask_run = np.array(self.mask_run)
        self.performed_run = np.any(self.mask_run)
        if not self.performed_run:
            self.is_ok = False
        else:
            run_times = np.array(self.all_times)[self.mask_run]
            assert len(run_times)
            dt = run_times[-1] - run_times[0] + 1
            coverage_fraction = len(run_times) / dt

            if coverage_fraction < self.min_coverage_fraction:
                # print(f'coverage_fraction={coverage_fraction:.3f}')
                self.is_ok = False

        c_d_dc = [[n.pars[2], n.pars[4], n.pars[6]] for n in self.all_nodes]
        c_d_dc = np.asarray(c_d_dc).transpose() / 255

        self.c, self.d, self.dc = c_d_dc
        self.f_dc = gauss_flt1d(self.dc, sigma=2.5, mode='reflect')

    def fill_steps(self):
        if len(self.all_times) < self.n_min:
            raise ValueError('Too little nodes')
        assert (len(self.all_times) > 1)
        for t1, t2, n1, n2 in zip(self.all_times[:-1], self.all_times[1:],
                                  self.all_nodes[:-1], self.all_nodes[1:]):
            dt = (t2 - t1)
            r1 = n1.r
            r2 = n2.r
            dr = [x2 - x1 for x1, x2 in zip(r1, r2)]

            d = np.linalg.norm(dr)
            v = (d / dt / cfgm.DT) if dt != 0 else 1.e12  # um/s
            self.steps.append(((n1, n2), dt, dr, d, v))

        all_d = np.array([s[3] for s in self.steps])
        all_dt = np.array([s[1] for s in self.steps])
        all_dr = np.array([s[2] for s in self.steps])

        cum_d = np.cumsum(all_d, axis=0)
        cum_dt = np.cumsum(all_dt, axis=0)
        cum_dr = np.cumsum(all_dr, axis=0)

        # print(len(self.steps), len(cum_d))
        horizon_path = []
        for idx0, d0 in enumerate(cum_d):
            for d_idx1, d1 in enumerate(cum_d[idx0 + 1:]):
                idx1 = idx0 + d_idx1 + 1
                # print(idx0, idx1)
                d = max(d1 - d0, 1e-3)
                dt = cum_dt[idx1] - cum_dt[idx0]
                if d >= self.probing_max_displ:
                    horizon_path.append(dt)
                    break
            else:
                dt_est = int(np.ceil(dt / d * self.probing_max_displ))
                horizon_path.append(dt_est)
        horizon_path.append(horizon_path[-1])
        self.horizon_path = np.array(horizon_path)

        horizon_dr = []
        for idx0, dr0 in enumerate(cum_dr):
            for d_idx1, dr1 in enumerate(cum_dr[idx0 + 1:]):
                idx1 = idx0 + d_idx1 + 1

                dr = dr1 - dr0
                d = max(np.linalg.norm(dr), 1e-3)
                dt = cum_dt[idx1] - cum_dt[idx0]

                if d >= self.probing_max_displ:
                    horizon_dr.append(dt)
                    break
            else:
                dt_est = int(np.ceil(dt / d * self.probing_max_displ))
                horizon_dr.append(dt_est)
        horizon_dr.append(horizon_dr[-1])
        self.horizon_dr = np.array(horizon_dr)

        self.horizon_f_path = gauss_flt1d(self.horizon_path, 2)
        self.horizon_f_dr = gauss_flt1d(self.horizon_dr, 2)

    def fill_ending_info(self):
        w_arr = np.array([n.w for n in self.all_nodes])
        f_doc_arr = self.f_dc
        c_arr = self.c

        # mask before run start
        mask_run = self.mask_run

        if not self.performed_run:
            self.ending_info = None
            return

        w_arr = w_arr[mask_run]
        f_doc_arr = f_doc_arr[mask_run]
        c_arr = c_arr[mask_run]

        w_mean = np.mean(w_arr)
        w_med = np.median(w_arr)

        c_mean = np.mean(c_arr)
        c_med = np.median(c_arr)

        w_l1 = w_arr[-1]
        w_mean_l5 = np.mean(w_arr[-5:])

        fdoc_mean_l5 = np.mean(f_doc_arr[-5:])
        c_l1 = c_arr[-1]
        c_mean_l5 = np.mean(c_arr[-5:])

        # track_info = {
        #    'w_mean': w_mean,
        #    'w_med' : w_med,
        #    'wl1_o_wmean': w_l1/w_mean,
        #    'wl1_o_wmed': w_l1/w_med,
        #    'wml5_o_wmean': w_mean_l5/w_mean,
        #    'wml5_o_wmed': w_mean_l5/w_med,
        #    'c_mean': c_mean,
        #    'c_med' : c_med,
        #    'fdoc_mean_l5': fdoc_mean_l5,
        #    'c_l1' : c_l1,
        #    'c_mean_l5': c_mean_l5
        # }

        self.ending_info = np.array([
            w_mean,
            w_med,
            w_l1 / w_mean,
            w_l1 / w_med,
            w_mean_l5 / w_mean,
            w_mean_l5 / w_med,
            c_mean,
            c_med,
            fdoc_mean_l5,
            c_l1,
            c_mean_l5
        ])

    def check_is_cell(self):
        if self.classificator_not_a_cell and self.ending_info is not None:
            w_mean_w_med = [self.ending_info[:2]]
            self.not_a_cell = self.classificator_not_a_cell.predict(w_mean_w_med)[0] == 1
            if self.not_a_cell:
                self.is_ok = False

    def fill_neighbouring_counts(self):
        # if can will fill # tracks within dr to this one in vicinity
        if self.neighbourhood_checker is None:
            return
        self.neigh_count = []
        for t, n in zip(self.all_times, self.all_nodes):
            x, y, _ = n.r
            n_neigh = self.neighbourhood_checker.get_n_neighbours(x, y, t)  # if self.all_times[-1]==t else 0

            assert n_neigh >= 0, f'n_neigh = {n_neigh}'
            self.neigh_count.append(n_neigh)

    def fill_ending_flags(self):
        if self.all_times[-1] < self.tf_max:
            end_has_neighb = (self.neigh_count[-1] > 0) if self.neigh_count else None

            if self.mask_any_diap[-1]:
                # assume diapedesis &/or inefficiency,
                # but track usable for e,g, crawling speeds etc before diap

                if end_has_neighb:
                    # set diap & tracking ineff (cell colisions)
                    self.detected_diap_ineff = True
                    self.detected_tracking_ineff = True
                elif end_has_neighb is None:
                    # set diap & crowd ineff, even we don't know abt crowd. but it's always there soo
                    self.detected_diap_ineff = True
                    self.detected_tracking_ineff = True
                else:
                    # set diap ineff
                    self.detected_diap_ineff = True

            else:
                # no diap activity in the track end
                # either detachment or inefficiency
                # if not crowded - assume diap, otherwise give up on the tracks

                if self.classificator_det and self.ending_info is not None:
                    detached = self.classificator_det.predict(self.ending_info)[0] == 1
                    self.performed_detachment = detached
                else:
                    self.performed_detachment = True

                if self.performed_detachment:
                    self.detached_tf = self.all_times[-1] - self.tf_min

                if not self.performed_detachment:  # otherwise all good
                    if end_has_neighb is False:
                        # assume transmigrated
                        # set diap inefficiency
                        self.detected_diap_ineff = True

                        self.mask_any_diap[-1] = True
                        self.mask_full_diap[-1] = True
                        self.mask_pre_diap[-1] = False
                        self.mask_above[-1] = False
                        self.performed_any_diap = True
                        self.performed_fdiap = True
                        self.performed_no_diap = False

                        last_idx = len(self.all_times) - 1
                        self.fdiap_start_idxs.append(last_idx)
                        self.fdiap_done_idxs.append(last_idx)
                        if len(self.first_diap_start_idxs) == 0:
                            self.first_diap_start_idxs.append(last_idx)

                        if self.first_full_diap_tf is None:
                            self.first_full_diap_tf = self.all_times[last_idx] - self.tf_min

                    else:  # has neighbours or we don't know - for sure some inefficiency, and we cant guess
                        self.detected_tracking_ineff = True
                        self.is_ok = False

    def get_full_diap(self):
        c, d, dc, fdc = self.c, self.d, self.dc, self.f_dc
        mask_full_diap = fdc >= TrackAnalyzer._thres_comp_diap

        mask_full_diap = close_short_gaps(mask_full_diap,
                                          min_len=TrackAnalyzer._min_comp_diap_e_len,
                                          min_len_b=1,
                                          min_len_e=TrackAnalyzer._min_comp_diap_e_len
                                          )
        mask_full_diap = mask_short_seq(mask_full_diap,
                                        min_len=TrackAnalyzer._min_comp_diap_len,
                                        min_len_b=TrackAnalyzer._min_comp_diap_len,
                                        min_len_e=TrackAnalyzer._min_comp_diap_e_len
                                        )
        self.mask_full_diap = mask_full_diap

    def get_any_diap(self):
        c, d, dc, fdc = self.c, self.d, self.dc, self.f_dc
        mask_any_diap = fdc >= TrackAnalyzer._thres_part_diap
        mask_any_diap = close_short_gaps(mask_any_diap,
                                         min_len=TrackAnalyzer._min_part_diap_len,
                                         min_len_b=1,
                                         min_len_e=TrackAnalyzer._min_part_diap_len
                                         )
        mask_any_diap = mask_short_seq(mask_any_diap,
                                       min_len=TrackAnalyzer._min_part_diap_len,
                                       min_len_b=TrackAnalyzer._min_part_diap_len,
                                       min_len_e=TrackAnalyzer._min_part_diap_len
                                       )
        self.mask_any_diap = mask_any_diap

    def get_full_diap_boundaries(self):
        self.mask_dir_tdiap = np.zeros_like(self.mask_run).astype(bool)
        self.mask_rev_tdiap = np.zeros_like(self.mask_run).astype(bool)
        self.mask_pdiap = np.zeros_like(self.mask_run).astype(bool)

        fdiap_start_idxs = []  # start in
        fdiap_done_idxs = []  # fully in
        rdiap_start_idxs = []  # start out
        rdiap_done_idxs = []  # fully out
        pdiap_start_idxs = []  # start out
        pdiap_done_idxs = []  # fully out

        if np.any(self.mask_any_diap):
            k_above = 1
            k_pdiap = 2
            k_fdiap = 4
            int_mask = k_above * self.mask_above + k_pdiap * self.mask_udiap + k_fdiap * self.mask_full_diap
            n = len(int_mask)

            block_lengths = [0]
            for val, grp in groupby(int_mask):
                grp_len = len([0 for _ in grp])
                block_lengths.append(grp_len)

            block_offsets = np.cumsum(block_lengths)
            # block_lengths = block_lengths[1:]

            curr_above = True  # curr state
            prev_val = None
            prev_ofs = None

            for curr_ofs in block_offsets:
                if curr_ofs == n:  # meaning reached the end:
                    # print(curr_ofs)
                    if prev_val == k_pdiap:  # cunfinished transition
                        # actually not: we can't conclusively clain any of these
                        # if curr_above:
                        #    self.mask_dir_tdiap[prev_ofs:curr_ofs] = True
                        # else:
                        #    self.mask_rev_tdiap[prev_ofs:curr_ofs] = True
                        pass
                else:
                    val = int_mask[curr_ofs]
                    # print(curr_ofs, val)
                    if val == k_above:
                        if not curr_above:
                            # switch
                            curr_above = True
                            rdiap_done_idxs.append(curr_ofs)

                            if prev_val == k_pdiap:
                                # fill dir P-diap mask
                                self.mask_rev_tdiap[prev_ofs:curr_ofs] = True
                                rdiap_start_idxs.append(prev_ofs)
                            else:
                                rdiap_start_idxs.append(curr_ofs)
                        else:
                            # partial diapedesis
                            if prev_val == k_pdiap:
                                # fill dir P-diap mask
                                self.mask_pdiap[prev_ofs:curr_ofs] = True
                                pdiap_start_idxs.append(prev_ofs)
                                pdiap_done_idxs.append(curr_ofs)

                    elif val == k_pdiap:
                        pass  # this can lead anywhere
                    elif val == k_fdiap:
                        if curr_above:
                            # switch
                            curr_above = False
                            fdiap_done_idxs.append(curr_ofs)

                            if prev_val == k_pdiap:
                                # fill dir P-diap mask
                                self.mask_dir_tdiap[prev_ofs:curr_ofs] = True
                                fdiap_start_idxs.append(prev_ofs)
                            else:
                                fdiap_start_idxs.append(curr_ofs)

                    else:
                        raise ValueError(f'unexpected val = {val}')

                    prev_val = val
                    prev_ofs = curr_ofs

        self.fdiap_start_idxs = fdiap_start_idxs
        self.fdiap_done_idxs = fdiap_done_idxs
        self.rdiap_start_idxs = rdiap_start_idxs
        self.rdiap_done_idxs = rdiap_done_idxs
        self.pdiap_start_idxs = pdiap_start_idxs
        self.pdiap_done_idxs = pdiap_done_idxs

    def get_detailed_diap(self):
        # fill all maps here and extend existing, eg if part diap to short inside full

        # 1.refine diap mask
        self.mask_any_diap += self.mask_full_diap

        # 2.aux masks
        self.mask_above = ~self.mask_any_diap
        self.mask_udiap = self.mask_any_diap * (~self.mask_full_diap)

        # 3. fill direct and reversed p diap masks and timepoints & idx of change
        self.get_full_diap_boundaries()

        # 4. get prediap
        self.mask_pre_diap = np.ones_like(self.mask_run).astype(bool)
        if np.any(self.mask_any_diap):
            any_diap_start_idx = np.argmax(self.mask_any_diap)  # if everywhere - will return 0
            self.mask_pre_diap[any_diap_start_idx:] = False
            self.first_diap_start_idxs = [any_diap_start_idx]

    def fill_diap_times(self):
        self.first_full_diap_tf = (None
                                   if len(self.fdiap_done_idxs) == 0
                                   else self.all_times[self.fdiap_done_idxs[0]] - self.tf_min
                                   )

    def fill_diap_flags(self):
        self.performed_fdiap = np.any(self.mask_run * self.mask_full_diap)
        self.performed_any_diap = np.any(self.mask_run * self.mask_any_diap)
        self.performed_pdiap = np.any(self.mask_run * self.mask_pdiap)
        self.performed_rdiap = len([idx for idx in self.rdiap_done_idxs if self.all_times[idx] >= self.tf_min]) > 0
        self.performed_no_diap = ~self.performed_any_diap

        # evaluate params
        n_run = sum(self.mask_run)
        if self.performed_fdiap:
            fdiap_dcs = self.dc[self.mask_full_diap * self.mask_run]

            self.fdiap_fraction = fdiap_dcs.mean()
            self.fdiap_max = fdiap_dcs.max()
            self.fdiap_integral = fdiap_dcs.sum()
            self.fdiap_num_tf = len(fdiap_dcs)
            self.fdiap_frac_tf = self.fdiap_num_tf / n_run

        if self.performed_any_diap:
            adiap_dcs = self.dc[self.mask_any_diap * self.mask_run]

            self.adiap_fraction = adiap_dcs.mean()
            self.adiap_max = adiap_dcs.max()
            self.adiap_integral = adiap_dcs.sum()
            self.adiap_num_tf = len(adiap_dcs)
            self.adiap_frac_tf = self.adiap_num_tf / n_run

        if self.performed_pdiap:
            pdiap_dcs = self.dc[self.mask_pdiap * self.mask_run]

            self.pdiap_fraction = pdiap_dcs.mean()
            self.pdiap_max = pdiap_dcs.max()
            self.pdiap_integral = pdiap_dcs.sum()
            self.pdiap_num_tf = len(pdiap_dcs)
            self.pdiap_frac_tf = self.pdiap_num_tf / n_run

    def collect_aggregate(self):
        c, d, dc = self.c, self.d, self.dc

        for dci, ti in zip(dc, self.all_times):
            if ti not in TrackAnalyzer._diapedesis_profile:
                TrackAnalyzer._diapedesis_profile[ti] = []

            TrackAnalyzer._diapedesis_profile[ti].append(dci)

    def get_diap_info(self):
        n = len(self.all_nodes)
        if n == 0:
            return

        self.get_full_diap()
        self.get_any_diap()
        self.get_detailed_diap()

        self.fill_diap_times()
        self.fill_diap_flags()

        self.collect_aggregate()

    class DisplacementMetric:
        def __init__(self):
            self.dr3 = np.array([0., 0., 0.])
            self.dr = 0

            self.path = 0

            self.dt = 0

            self.v = 0
            self.v_p = 0  # path

            self.v_mean = 0
            self.v_std = 0

            self.s = None  # Dunn-Othmer
            self.tau = None

    @staticmethod
    def get_diff_persistance_from_steps(steps, min_fit_pairs=6, max_dt=30):
        last_node = None
        t = 0

        r_groups = []
        t_groups = []

        r_group_curr = []
        t_group_curr = []

        for (n1, n2), dt, dr3, dr, v in steps:
            if last_node is not None and n1 != last_node:
                # new group starts
                if len(r_group_curr) > 3:  # min 3 sequential steps
                    r_groups.append(r_group_curr)
                    t_groups.append(t_group_curr)
                r_group_curr = [n1.r]
                t_group_curr = [0.]
                t = 0.

            last_node = n2
            t += dt
            r_group_curr.append(n2.r)
            t_group_curr.append(t)

        if len(r_group_curr) > 3:  # min 3 sequential steps
            r_groups.append(r_group_curr)
            t_groups.append(t_group_curr)

            # print(r_groups)

        dr_arr = []
        dt_arr = []

        for grp_idx in range(len(r_groups)):
            r_group = r_groups[grp_idx]
            t_group = t_groups[grp_idx]

            n_smpl = len(r_group)

            for first_idx in range(n_smpl - 1):
                r1 = r_group[first_idx]
                t1 = t_group[first_idx]

                for last_idx in range(first_idx + 1, min(first_idx + max_dt, n_smpl)):
                    r2 = r_group[last_idx]
                    t2 = t_group[last_idx]

                    dr = [x2 - x1 for x1, x2 in zip(r1, r2)]
                    dr = np.linalg.norm(dr)
                    dt = t2 - t1

                    dr_arr.append(dr)
                    dt_arr.append(dt)

        if len(dr_arr) < min_fit_pairs:
            return None

        dr_arr, dt_arr = np.array(dr_arr), np.array(dt_arr)
        return get_diff_persistance(dr_arr, dt_arr)

    def get_dispacement_metric_from_steps(self, steps):
        dm = TrackAnalyzer.DisplacementMetric()
        # n1 = steps[0][0][0]
        # n2 = steps[-1][0][1]

        # r1 = n1.r
        # r2 = n2.r

        all_v = []
        for n1n2, dt, dr3, dr, v in steps:
            dm.dt += dt
            dm.dr3 += dr3
            dm.path += dr
            all_v.append(v)

        dm.dr = np.linalg.norm(dm.dr3)

        dm.v = dm.dr / dm.dt
        dm.v_p = dm.path / dm.dt
        dm.v_mean, dm.v_std = mean_std(all_v) if len(all_v) > 1 else (all_v[0], 0)

        s_tau = self.get_diff_persistance_from_steps(steps)
        if s_tau is not None:
            dm.s, dm.tau = s_tau

        return dm

    def fill_displacement_metrics(self):
        mask_invalid = self.mask_jumps + self.mask_ineff
        mask_valid = (~mask_invalid) * self.mask_run
        mask_valid = mask_valid[:-1]

        mask_before_diap = mask_valid * self.mask_pre_diap[:-1]
        if np.sum(mask_before_diap):
            steps_before_diap = [self.steps[i] for i, v in enumerate(mask_before_diap) if v]
            self.mtr_before_diap = self.get_dispacement_metric_from_steps(steps_before_diap)
        else:
            self.mtr_before_diap = None

        mask_crawling_before_diap = mask_valid * self.mask_pre_diap[:-1]
        if np.sum(mask_crawling_before_diap):
            steps_crawling_before_diap = [self.steps[i] for i, v in enumerate(mask_crawling_before_diap) if v]
            self.mtr_crawling_before_diap = self.get_dispacement_metric_from_steps(steps_crawling_before_diap)
        else:
            self.mtr_crawling_before_diap = None

        mask_all_crawling = mask_valid * self.mask_above[:-1] * self.mask_steps_segment_crawling_above
        if np.sum(mask_all_crawling):
            steps_all_crawling = [self.steps[i] for i, v in enumerate(mask_all_crawling) if v]
            self.mtr_all_crawling = self.get_dispacement_metric_from_steps(steps_all_crawling)
        else:
            self.mtr_all_crawling = None

        mask_all_fdiap = mask_valid * self.mask_full_diap[:-1]
        if np.sum(mask_all_fdiap):
            steps_all_fdiap = [self.steps[i] for i, v in enumerate(mask_all_fdiap) if v]
            self.mtr_all_fdiap = self.get_dispacement_metric_from_steps(steps_all_fdiap)
        else:
            self.mtr_all_fdiap = None

        if np.sum(mask_valid):
            steps_all_valid = [self.steps[i] for i, v in enumerate(mask_valid) if v]
            self.mtr_all_valid = self.get_dispacement_metric_from_steps(steps_all_valid)
        else:
            self.mtr_all_valid = None

        mask_all_run = self.mask_run[:-1]
        if np.sum(mask_all_run):
            steps_all_run = [self.steps[i] for i, v in enumerate(mask_all_run) if v]
            self.mtr_all_run = self.get_dispacement_metric_from_steps(steps_all_run)
        else:
            self.mtr_all_run = None

    def fill_jumps_metric(self):
        mask_run_jumps = self.mask_jumps * self.mask_run * self.mask_above

        self.performed_jumps = np.any(mask_run_jumps)

        mask_run_jumps = mask_run_jumps[:-1]

        if np.sum(mask_run_jumps):
            steps_run_jumps = [self.steps[i] for i, v in enumerate(mask_run_jumps) if v]
            self.jumps_dr3_dt = [(dr3d, dt) for n1n2, dt, dr3d, dr, v in steps_run_jumps]
        else:
            self.jumps_dr3_dt = []

    def fill_probing(self):
        mask_invalid = self.mask_jumps + self.mask_ineff
        mask_valid = (~mask_invalid) * self.mask_run
        mask_valid = mask_valid[:-1]

        self.mask_steps_segment_probing_above = mask_valid * (~self.mask_any_diap[:-1]) * (
            ~ self.mask_steps_segment_crawling_above)
        self.mask_steps_segment_probing_below = mask_valid * (self.mask_full_diap[:-1]) * (
            ~ self.mask_steps_segment_crawling_below)

        self.performed_probing = not self.performed_crawling  # np.any(self.mask_steps_segment_probing_above)  #
        self.performed_probing_below = np.any(self.mask_steps_segment_probing_below)

        if np.any(self.mask_steps_segment_probing_above):
            steps_probing = [self.steps[i] for i, v in enumerate(self.mask_steps_segment_probing_above) if v]
            self.mtr_all_probing = self.get_dispacement_metric_from_steps(steps_probing)
        else:
            self.mtr_all_probing = None

    def fill_metrics(self):
        self.fill_displacement_metrics()
        self.fill_jumps_metric()
        self.fill_probing()

    def get_jump_info(self, dt_max=3):
        # node indeces correspond to start node of step with same idx
        min_jump_dr = self.jump_displ_thres
        max_crawl_v = min(self.v_nj_mean + 3 * self.sigma_v_nj, self.crawling_speed_max)

        mask_jumps = []  # idx -> idx(step[idx].n1n2[n1] == idx)
        mask_ineff = []  # idx -> idx(step[idx].n1n2[n1] == idx)

        for idx, (n1n2, dt, dr3d, dr, v) in enumerate(self.steps):
            if dt > dt_max:
                mask_ineff.append(True)
                mask_jumps.append(False)
            else:
                is_jump = (dr / dt > max_crawl_v) and (dr > min_jump_dr)
                mask_ineff.append(False)
                mask_jumps.append(is_jump)

        mask_jumps.append(False)  # for nodes-compatibility
        mask_ineff.append(False)

        mask_jumps = np.array(mask_jumps)
        mask_ineff = np.array(mask_ineff)

        self.mask_jumps = mask_jumps
        self.mask_ineff = mask_ineff
        self.mask_steps_jumps = mask_jumps[:-1]
        self.mask_steps_ineff = mask_ineff[:-1]

    def get_speed_no_jumps(self, dt_max=3):
        self.v_nj_mean = self.crawling_speed_max / 2  # default 36/2 = 18 um/min
        self.sigma_v_nj = self.crawling_speed_max / 6  # default 36/6 = 6  um/min

        # only non-diap, run part
        mask_run_pre_d = (self.mask_run * self.mask_pre_diap)[:-1]
        steps = [self.steps[i] for i, v in enumerate(mask_run_pre_d) if v]

        if len(steps) < self.n_min * 3:  # 18 tf
            # raise ValueError('Too little elements in steps')
            return

        v_arr = [v for n1n2, dt, dr3d, dr, v in steps if dt <= dt_max]

        if len(v_arr) < self.n_min * 3:
            # raise ValueError('Too little elements in v_arr')
            return

        med, mad = median_MAD(v_arr)
        sigma_est = mad * 1.4826

        v_no_outl = [v for v in v_arr if v < med + 3 * sigma_est]

        if len(v_no_outl) < self.n_min:
            # raise ValueError('Too little elements in v_no_outl')
            return

        self.v_nj_mean = np.mean(v_no_outl)
        self.sigma_v_nj = np.std(v_no_outl, ddof=1)

    def get_no_jump_dr(self, from_node_idx, to_node_idx):
        # get t1 t2
        # t1, t2 = self.all_times[from_node_idx], self.all_times[to_node_idx]

        # get all jumps between
        sel_mask = self.mask_jumps[from_node_idx:to_node_idx]
        sel_steps = self.steps[from_node_idx:to_node_idx]
        sel_jump_steps = [sel_steps[i] for i, v in enumerate(sel_mask) if v]

        if len(sel_jump_steps):
            # get jump correction (sum)
            all_dr = [dr3d for n1n2, dt, dr3d, dr, v in sel_jump_steps]
            dr3d_jmps = np.sum(all_dr, axis=0)
        else:
            dr3d_jmps = [0., 0., 0.]

        # get to-from node distance
        n1 = self.all_nodes[from_node_idx]
        n2 = self.all_nodes[to_node_idx]
        r1 = n1.r
        r2 = n2.r
        dr3d = np.array([x2 - x1 for x1, x2 in zip(r1, r2)])

        # correct
        dr3d_corr = dr3d - dr3d_jmps

        dr_corr = np.linalg.norm(dr3d_corr)

        return dr_corr

    def get_crawling(self):
        adhesion_offset = 4
        # 1. for each non-diap region, each diap - max distance to els, before next diap start
        # masked = np.ones_like(bool_arr).astype(bool)

        offsets_above = []
        lenghth_above = []
        offsets_below = []
        lenghth_below = []
        ofs = 0
        for val, grp in groupby(self.mask_any_diap[:-1]):
            grp = [g_i for g_i in grp]
            grp_len = len(grp)

            if not val:  # above
                offsets_above.append(ofs)
                lenghth_above.append(grp_len)
            ofs += grp_len

        ofs = 0
        for val, grp in groupby(self.mask_full_diap[:-1]):
            grp = [g_i for g_i in grp]
            grp_len = len(grp)

            if val and grp_len > 0:  # below
                offsets_below.append(ofs)
                lenghth_below.append(grp_len)
            ofs += grp_len

        n_steps = len(self.steps)
        field_max_segment_displacement_above = np.ones(n_steps) * (-1)  # -1 means unknown/meaningless
        field_max_segment_displacement_below = np.ones(n_steps) * (-1)  # -1 means unknown/meaningless

        # 2. fn to get dr_ij, excluding gaps

        # 3. iterate each el, fill distance maps
        for ofs, grp_len in zip(offsets_above, lenghth_above):
            for i in range(ofs, ofs + grp_len - 1):  # from_nodes
                drs_abs_node = []
                for j in range(i + 1, ofs + grp_len):  # to_nodes
                    dr_abs_segm = self.get_no_jump_dr(from_node_idx=i, to_node_idx=j)
                    drs_abs_node.append(dr_abs_segm)

                max_sgm_dr = np.max(drs_abs_node)
                field_max_segment_displacement_above[i] = max_sgm_dr
            field_max_segment_displacement_above[ofs + grp_len - 1] = 0

        for ofs, grp_len in zip(offsets_below, lenghth_below):
            for i in range(ofs, ofs + grp_len - 1):  # from_nodes
                drs_abs_node = []
                for j in range(i + 1, ofs + grp_len):  # to_nodes
                    dr_abs_segm = self.get_no_jump_dr(from_node_idx=i, to_node_idx=j)
                    drs_abs_node.append(dr_abs_segm)

                max_sgm_dr = np.max(drs_abs_node)
                field_max_segment_displacement_below[i] = max_sgm_dr
            field_max_segment_displacement_below[ofs + grp_len - 1] = 0

        # \ fill groups masks
        mask_steps_segment_crawling_above = np.zeros(n_steps).astype(bool)
        mask_steps_segment_crawling_below = np.zeros(n_steps).astype(bool)

        # mask as crawling if group crawling
        for idx, (ofs, grp_len) in enumerate(zip(offsets_above, lenghth_above)):

            if ofs < adhesion_offset:
                # 5.update for first group from min(run_idx, start + adhesion_offset tf)  # adhesion/rolling
                d = adhesion_offset - ofs
                ofs += d
                grp_len -= d

            if np.any(field_max_segment_displacement_above[ofs:ofs + grp_len] >= self.probing_max_displ):
                mask_steps_segment_crawling_above[ofs:ofs + grp_len] = True

        for idx, (ofs, grp_len) in enumerate(zip(offsets_below, lenghth_below)):
            ofs_l = ofs
            l_l = grp_len

            if ofs_l < adhesion_offset:
                # 5.update for first group from min(run_idx, start + adhesion_offset tf)  # adhesion/rolling
                d = adhesion_offset - ofs_l
                ofs_l += d
                l_l -= d

            if np.any(field_max_segment_displacement_below[ofs_l:ofs_l + l_l] >= self.probing_max_displ):
                mask_steps_segment_crawling_below[ofs:ofs + grp_len] = True

        # 6.if till first diap probing - set track as crawling=false
        mask_run_pre_d = (self.mask_run * self.mask_pre_diap)[:-1]
        if np.sum(mask_run_pre_d):
            crawling = np.any(field_max_segment_displacement_above[mask_run_pre_d] >= self.probing_max_displ)
            mask_steps_segment_crawling_above[mask_run_pre_d] = crawling
            self.performed_crawling = crawling
        else:
            # 7. if fisrt diap - fill crawling based on crawling map beforehand:
            crawling = mask_steps_segment_crawling_below[adhesion_offset]
            self.performed_crawling = crawling

        self.field_max_segment_displacement_above = field_max_segment_displacement_above
        self.field_max_segment_displacement_below = field_max_segment_displacement_below

        self.mask_steps_segment_crawling_above = mask_steps_segment_crawling_above
        self.mask_steps_segment_crawling_below = mask_steps_segment_crawling_below

    # all proc
    def analyze(self):
        """
        runs all analysius and prepares outputs
        """
        self.fill_neighbouring_counts()

        self.fill_ending_info()
        self.check_is_cell()

        self.get_diap_info()
        self.fill_ending_flags()  # requires diap results

        self.get_speed_no_jumps()
        self.get_jump_info()

        if self.is_ok:
            self.get_crawling()

            self.fill_metrics()

    def plot_diap_info(self, draw_v=False, draw_w=False, draw_all=False):
        c, d, dc, fdc = self.c, self.d, self.dc, self.f_dc
        n = len(dc)
        if (self.dc.max() > 0.5 and n > 30) or self.performed_fdiap or self.performed_pdiap or draw_all:
            v_norm = 10.
            w_norm = 1000.
            v_arr = np.array([v / v_norm for (n1n2, dt, dr3d, dr, v) in self.steps])
            w_arr = np.array([n.w / w_norm for n in self.all_nodes])

            if self.performed_fdiap:
                print('Diapedesis time:', self.first_full_diap_tf)
                print('Diapedesis start/done idx:', (self.fdiap_start_idxs[0], self.fdiap_done_idxs[0]))
                print('frac %.2f, max %.2f, int %.2f, ntf %d' % (self.fdiap_fraction,
                                                                 self.fdiap_max,
                                                                 self.fdiap_integral,
                                                                 self.fdiap_num_tf))
            if self.performed_pdiap:
                print('Partial diap: frac %.2f, max %.2f, int %.2f, ntf %d' % (self.pdiap_fraction,
                                                                               self.pdiap_max,
                                                                               self.pdiap_integral,
                                                                               self.pdiap_num_tf),

                      )

            xy = [n.r[:2] for n in self.all_nodes]
            xy.append(Track._boundary_x0x1y0y1[::2])
            xy.append(Track._boundary_x0x1y0y1[1::2])

            xy = np.asarray(xy).transpose()
            x, y = xy

            t = self.all_times.copy()
            t.extend([0] * 2)
            t = np.asarray(t)

            fig, ax = plt.subplots(1, 4, figsize=(16, 4))

            ax[0].plot(self.all_times, c, label='c')
            ax[0].plot(self.all_times, d, label='d')
            ax[0].plot(self.all_times, dc, label='d/c')

            if draw_v:
                ax[0].plot(self.all_times[:-1], v_arr + 1, '--', label=f'v/{v_norm}')
            if draw_w:
                ax[0].plot(self.all_times, w_arr + 1, '--', label=f'w/{w_norm}')

            ax[0].legend(fontsize='xx-small')

            ax[1].plot(self.all_times, fdc, label='F(d/c)')

            masks = [
                self.mask_any_diap,

                self.mask_full_diap,
                self.mask_udiap,
                self.mask_dir_tdiap,
                self.mask_rev_tdiap,
                self.mask_pdiap,
                self.mask_above,
                self.mask_pre_diap,
            ]
            labels = [
                'any_diap',
                'full_diap',
                'udiap',
                'dir_tdiap',
                'rev_tdiap',
                'pdiap',
                'above',
                'pre_diap',
            ]

            for i, (m, l) in enumerate(zip(masks[::-1], labels[::-1])):
                ax[1].plot(self.all_times, m * 0.5 - i - 1, label=l)
            ax[1].legend(fontsize='xx-small')

            t_col = t
            d_col = dc

            r = ax[2].scatter(x, y, c=t_col, marker='.', cmap=cm.rainbow, vmin=0, vmax=180)
            ax[2].set_aspect('equal')
            fig.colorbar(r, ax=ax[2])

            r = ax[3].scatter(x[:-2], y[:-2], c=d_col, marker='.', cmap=cm.rainbow, vmin=0, vmax=1)
            ax[3].set_aspect('equal')
            fig.colorbar(r, ax=ax[3])

            plt.show()
            plt.close()

    # ---
    def get_speed_no_diap(self, dt_max=3):
        v_arr = [v for (n1n2, dt, dr3d, dr, v), pdiap in zip(self.steps, self.mask_any_diap) if
                 (dt <= dt_max and not pdiap)]
        if len(v_arr) < self.n_min:
            raise ValueError('Too little elements in v_arr')

        med, mad = median_MAD(v_arr)
        sigma_est = mad * 1.4826

        v_no_outl = [v for v in v_arr if v < med + 3 * sigma_est]
        self.n_v_nd = len(v_no_outl)
        if self.n_v_nd < self.n_min:
            raise ValueError('Too little elements in v_no_outl')

        self.v_nd_mean = np.mean(v_no_outl)
        self.sigma_v_nd = np.std(v_no_outl, ddof=1)
        # print(self.v_nd_mean, self.sigma_v_nd, self.n_v_nd, len(v_arr), len(self.steps))

    @staticmethod
    def plot_aggregated_diap_prof():
        # fudge, no sence on prod
        t = []
        dc = []
        dcs = []
        for ti, dca in TrackAnalyzer._diapedesis_profile.items():
            t.append(ti)
            dci = np.mean(dca)
            dc.append(dci)
            dcsi = np.std(dca)
            dcs.append(dcsi)

        t = np.asarray(t)
        dc = np.asarray(dc)
        dcs = np.asarray(dcs)
        t_idx = np.argsort(t)
        t = t[t_idx]
        dc = dc[t_idx]
        dcs = dcs[t_idx]
        if len(dcs) == 0 or np.any(dcs > 0) is False:
            return
        dcs[dcs == 0] = dcs[dcs > 0].min()

        # fit dc = a*t + b
        a, b = np.polyfit(t, dc, deg=1, w=1 / dcs)
        dc_corr = dc - (a * t + b)
        std_corr = dc_corr.std()
        idx_abnormal = dc_corr > 2 * std_corr
        t_abnormal = t[idx_abnormal]
        print(std_corr, t_abnormal)
        TrackAnalyzer._diapedesis_error_t.update(t_abnormal)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot(t, dc)
        ax[0].set_title('mean DoC')
        ax[1].plot(t, dc_corr)
        ax[1].set_title('detranded mean DoC')
        plt.show()


def plot_analyzed_track(ta):
    x = [n.r[0] for n in ta.all_nodes]
    y = [n.r[1] for n in ta.all_nodes]

    col = ta.mask_run * 1 + (ta.mask_full_diap * ta.mask_run) * (-1) + ta.mask_full_diap * 2
    plt.scatter(x, y, s=4, c=col)
    plt.plot(x, y, linewidth=0.5, alpha=0.5)


def save_ds_tracks(tracks, fname, confined_only=True):
    xyzt_arrs = []
    idx_arr = []

    for tr_idx, tr in enumerate(tracks):
        if confined_only and not tr.in_fid_vol:
            continue

        xyzt_arr = []

        def list_xyzt(n, t, xyzt_arr):
            xyzt_arr.append(list(n.r) + [t])
            return xyzt_arr

        xyzt_arr = tr.aggregate_nodes(list_xyzt, xyzt_arr)

        if len(xyzt_arr):
            xyzt_arrs.append(xyzt_arr)
            idx_arr.append(tr_idx)

    if fname:
        with open(fname, 'wt') as f:
            n_tr = len(xyzt_arrs)
            f.write(str(n_tr) + '\n')

            for tr_idx, xyzt_arr in zip(idx_arr, xyzt_arrs):
                f.write('%d %d' % (len(xyzt_arr), tr_idx) + '\n')
                for xyzt in xyzt_arr:
                    sxyzt = ['%.3f' % v for v in xyzt[:3]] + [str(int(xyzt[3]))]
                    f.write(' '.join(sxyzt) + '\n')
    else:
        n_tr = len(xyzt_arrs)
        print(n_tr)

        for tr_idx, xyzt_arr in zip(idx_arr, xyzt_arrs):
            print(len(xyzt_arr), tr_idx)
            for xyzt in xyzt_arr:
                sxyzt = ['%.3f' % v for v in xyzt[:3]] + [str(int(xyzt[3]))]
                print(' '.join(sxyzt))


# ## Analytics methods

def mtx_to_pathlen(mtx):
    return mtx.path


def mtx_to_displacement(mtx):
    return mtx.dr


def mtx_to_crawling_speed(mtx):
    return mtx.v_p * 60. / cfgm.DT


def mtx_to_migration_speed(mtx):
    return mtx.v * 60. / cfgm.DT


def mtx_to_inst_speed(mtx):
    return mtx.v_mean * 60.


def mtx_to_inst_speed_std(mtx):
    return mtx.v_std * 60.


def mtx_to_meandering_index(mtx):
    return mtx.path / mtx.dr


def mtx_to_directionality(mtx):
    return mtx.dr / mtx.path


def mtx_to_migration_time(mtx):
    return mtx.dt * cfgm.DT / 60.


def mtx_to_x_displacement(mtx):
    return mtx.dr3[0]


def mtx_to_y_displacement(mtx):
    return mtx.dr3[1]


def mtx_to_x_rel_displacement(mtx):
    return (mtx.dr3[0] / mtx.dr) if mtx.dr > 0 else 0.


def mtx_to_y_rel_displacement(mtx):
    return (mtx.dr3[1] / mtx.dr) if mtx.dr > 0 else 0.


def mtx_to_s(mtx):
    s = mtx.s
    if s is None:
        return None
    return s * 60. / cfgm.DT


def mtx_to_tau(mtx):
    tau = mtx.tau
    if tau is None:
        return None
    return tau * cfgm.DT / 60.


motility_params = {
    0: {'id': 'path', 'name': 'Path length, $\\mu m$', 'fn': mtx_to_pathlen},
    1: {'id': 'disp', 'name': 'Displacement, $\\mu m$', 'fn': mtx_to_displacement},
    2: {'id': 'v_p', 'name': 'Crawling speed, $\\mu m/min$', 'fn': mtx_to_crawling_speed},
    3: {'id': 'v_m', 'name': 'Migration speed, $\\mu m/min$', 'fn': mtx_to_migration_speed},
    4: {'id': 'v_i', 'name': 'Instantaneous speed, $\\mu m/min$', 'fn': mtx_to_inst_speed},
    5: {'id': 'sv_i', 'name': 'Inst. speed std, $\\mu m/min$', 'fn': mtx_to_inst_speed_std},
    6: {'id': 'mi', 'name': 'Meandering Index', 'fn': mtx_to_meandering_index},
    7: {'id': 'dir', 'name': 'Directionality', 'fn': mtx_to_directionality},
    8: {'id': 'x_disp', 'name': 'X Displacement, $\\mu m$', 'fn': mtx_to_x_displacement},
    9: {'id': 'y_disp', 'name': 'Y Displacement, $\\mu m$', 'fn': mtx_to_y_displacement},
    10: {'id': 'x_rel_disp', 'name': 'Relative X Displacement', 'fn': mtx_to_x_rel_displacement},
    11: {'id': 'y_rel_disp', 'name': 'Relative Y Displacement', 'fn': mtx_to_y_rel_displacement},
    12: {'id': 's', 'name': 'DO Speed, $\\mu m/min$', 'fn': mtx_to_s},
    13: {'id': 'tau', 'name': 'DO Persistence, min', 'fn': mtx_to_tau},
    14: {'id': 't', 'name': 'Migration time, min', 'fn': mtx_to_migration_time},
}


def get_metric_crawling_no_det_no_diap(ta: TrackAnalyzer) -> TrackAnalyzer.DisplacementMetric:
    # non-diapedesing, non-detaching, crawling till the end
    return ta.mtr_crawling_before_diap if (ta.performed_crawling and
                                           not ta.performed_detachment and
                                           not ta.performed_any_diap
                                           ) else None


def get_metric_crawling_before_diap(ta: TrackAnalyzer) -> TrackAnalyzer.DisplacementMetric:
    # before first diapedesis, non-detaching, crawling
    return ta.mtr_crawling_before_diap if (ta.performed_crawling and
                                           not ta.performed_detachment
                                           ) else None


def get_metric_all_crawling(ta: TrackAnalyzer) -> TrackAnalyzer.DisplacementMetric:
    # all crawling above segmnets
    return ta.mtr_all_crawling


def get_metric_probing_before_diap(ta: TrackAnalyzer) -> TrackAnalyzer.DisplacementMetric:
    """
    Excluding jumps and inefficiency
    """
    return ta.mtr_before_diap if ta.performed_probing else None


def get_metric_probing(ta: TrackAnalyzer) -> TrackAnalyzer.DisplacementMetric:
    """
    Excluding jumps and inefficiency
    """
    return ta.mtr_all_probing


def get_metric_all_transmigrated(ta: TrackAnalyzer) -> TrackAnalyzer.DisplacementMetric:
    # all below segmnets
    return ta.mtr_all_fdiap


def get_metric_valid(ta: TrackAnalyzer) -> TrackAnalyzer.DisplacementMetric:
    """
    Excluding jumps and inefficiency
    """
    return ta.mtr_all_valid


motility_regimes = {
    0: {'id': 'crawling_only', 'name': 'Crawling only', 'fn': get_metric_crawling_no_det_no_diap},
    1: {'id': 'crawling_bt', 'name': 'Crawling before transmigration', 'fn': get_metric_crawling_before_diap},
    2: {'id': 'crawling_all', 'name': 'All crawling segmnets', 'fn': get_metric_all_crawling},

    3: {'id': 'probing_bt', 'name': 'Probing before transmigration', 'fn': get_metric_probing_before_diap},
    4: {'id': 'probing_all', 'name': 'All probing segmnets', 'fn': get_metric_probing},

    5: {'id': 'ft_all', 'name': 'All transmigrated', 'fn': get_metric_all_transmigrated},
    6: {'id': 'valid_all', 'name': 'All valid', 'fn': get_metric_valid},
}


def ta_to_am_dx(ta: TrackAnalyzer):
    return [dx for (dx, dy, dz), dt in ta.jumps_dr3_dt if dx < 0]


def ta_to_am_v(ta: TrackAnalyzer):
    return [(dx / dt * cfgm.DT / 60.) for (dx, dy, dz), dt in ta.jumps_dr3_dt if dx < 0]


am_params = {
    0: {'id': 'disp_am', 'name': 'AM Displacement, $\\mu m$', 'fn': ta_to_am_dx},
    1: {'id': 'v_am', 'name': 'AM  speed, $\\mu m/min$', 'fn': ta_to_am_v},
}


# In[43]:


class Behavior(Enum):
    probing = 0
    probing_UT = 1
    probing_FT = 2
    crawling = 3
    crawling_UT = 4
    crawling_FT = 5
    detached = 6


behavior_names = {
    Behavior.probing.value: 'Probing',
    Behavior.probing_UT.value: 'Probing UT',
    Behavior.probing_FT.value: 'Probing FT',
    Behavior.crawling.value: 'Crawling',
    Behavior.crawling_UT.value: 'Crawling UT',
    Behavior.crawling_FT.value: 'Crawling FT',
    Behavior.detached.value: 'Detached',
}


class BehaviorPC(Enum):
    probing = 0
    crawling = 1


behavior_pc_names = {
    BehaviorPC.probing.value: 'Probing',
    BehaviorPC.crawling.value: 'Crawling',
}


class BehaviorNUF(Enum):
    NT = 0
    UT = 1
    FT = 2


behavior_nuf_names = {
    BehaviorNUF.NT.value: BehaviorNUF.NT.name,
    BehaviorNUF.UT.value: BehaviorNUF.UT.name,
    BehaviorNUF.FT.value: BehaviorNUF.FT.name,
}


def get_motility_analytics_regime_value_dict(tas_dict, comparisson_group, condition_id_to_ds_id, metric_fn, val_fn):
    analytics_dict = {
        'group': [],
        'ds_id': [],
        'val': []
    }
    for group_id, condition_info in comparisson_group.items():
        condition_id = condition_info['condition_id']
        group_name = condition_info['name']

        ds_ids = condition_id_to_ds_id[condition_id]
        for ds_id in ds_ids:
            for ta in tas_dict.get(ds_id, []):  # tas might be limited
                mtx = metric_fn(ta)
                if mtx is not None:  # checl has data
                    val = val_fn(mtx)
                    if val is not None:
                        analytics_dict['group'].append(group_name)
                        analytics_dict['val'].append(val)
                        analytics_dict['ds_id'].append(ds_id)
    return analytics_dict


def get_motility_analytics_dict(tas_dict, comparisson_group, condition_id_to_ds_id):
    motility_analytics_dict = {}
    for motility_regime_id, motility_regime_info in motility_regimes.items():
        regime_name = motility_regime_info['name']
        metric_fn = motility_regime_info['fn']
        regime_sid = motility_regime_info['id']

        regime_analytics_val = {}
        for motility_param_id, motility_param_info in motility_params.items():
            motility_param_name = motility_param_info['name']
            motility_param_fn = motility_param_info['fn']
            motility_param_sid = motility_param_info['id']

            analytics_dict = get_motility_analytics_regime_value_dict(tas_dict,
                                                                      comparisson_group,
                                                                      condition_id_to_ds_id,
                                                                      metric_fn=metric_fn,
                                                                      val_fn=motility_param_fn)

            regime_analytics_val[motility_param_sid] = {
                'name': motility_param_name,
                'val': analytics_dict,

            }
        regime_analytics_dict = {'name': regime_name, 'val': regime_analytics_val}

        motility_analytics_dict[regime_sid] = regime_analytics_dict
    return motility_analytics_dict


def get_acc_movement_analytics_value_dict(tas_dict, comparisson_group, condition_id_to_ds_id, val_fn):
    analytics_dict = {
        'group': [],
        'ds_id': [],
        'val': []
    }
    for group_id, condition_info in comparisson_group.items():
        condition_id = condition_info['condition_id']
        group_name = condition_info['name']

        ds_ids = condition_id_to_ds_id[condition_id]
        for ds_id in ds_ids:
            for ta in tas_dict.get(ds_id, []):  # tas might be limited
                if ta.is_ok:  # checl has data
                    vals = val_fn(ta)
                    for val in vals:
                        analytics_dict['group'].append(group_name)
                        analytics_dict['val'].append(val)
                        analytics_dict['ds_id'].append(ds_id)
    return analytics_dict


def get_acc_movement_analytics_dict(tas_dict, comparisson_group, condition_id_to_ds_id):
    acc_movement_analytics_dict = {}

    for am_param_id, am_param_info in am_params.items():
        am_param_name = am_param_info['name']
        am_param_fn = am_param_info['fn']
        am_param_sid = am_param_info['id']

        analytics_dict = get_acc_movement_analytics_value_dict(tas_dict,
                                                               comparisson_group,
                                                               condition_id_to_ds_id,
                                                               val_fn=am_param_fn)

        acc_movement_analytics_dict[am_param_sid] = {
            'name': am_param_name,
            'val': analytics_dict,

        }
    return acc_movement_analytics_dict


def get_behavior_analytics_dict(tas_dict, comparisson_group, condition_id_to_ds_id):
    behavior_count = {}
    for group_id, condition_info in comparisson_group.items():
        condition_id = condition_info['condition_id']
        group_name = condition_info['name']
        behavior_count[group_name] = {}

        ds_ids = condition_id_to_ds_id[condition_id]
        for ds_id in ds_ids:
            behavior_count[group_name][ds_id] = [0] * len(Behavior)

            for ta in tas_dict.get(ds_id, []):  # tas might be limited
                # probing 0, probing pdiap 1, probing diap 2, crawling 3,
                # crawling pdiap 4 , crawling diap 5 , detached 6
                if not ta.is_ok:
                    continue

                if ta.performed_detachment:
                    class_idx = Behavior.detached.value
                else:

                    if ta.performed_crawling:
                        if ta.performed_fdiap:
                            class_idx = Behavior.crawling_FT.value
                        elif ta.performed_any_diap:
                            class_idx = Behavior.crawling_UT.value
                        else:
                            class_idx = Behavior.crawling.value
                    else:
                        if ta.performed_fdiap:
                            class_idx = Behavior.probing_FT.value
                        elif ta.performed_any_diap:
                            class_idx = Behavior.probing_UT.value
                        else:
                            class_idx = Behavior.probing.value
                behavior_count[group_name][ds_id][class_idx] += 1

    behavior_pc_count = {}
    behavior_nuf_count = {}
    for tp_n, behavior_count_ds in behavior_count.items():
        behavior_pc_count[tp_n] = {}
        behavior_nuf_count[tp_n] = {}

        for ds_id, vals in behavior_count_ds.items():
            behavior_pc_count[tp_n][ds_id] = [
                vals[Behavior.probing.value] + vals[Behavior.probing_UT.value] + vals[Behavior.probing_FT.value],
                vals[Behavior.crawling.value] + vals[Behavior.crawling_UT.value] + vals[Behavior.crawling_FT.value],
                ]
            behavior_nuf_count[tp_n][ds_id] = [vals[Behavior.crawling.value] + vals[Behavior.probing.value],
                                               vals[Behavior.crawling_UT.value] + vals[Behavior.probing_UT.value],
                                               vals[Behavior.crawling_FT.value] + vals[Behavior.probing_FT.value]
                                               ]

    behavior_count_dict = {
        'group': [],
        'ds_id': [],
        'behavior': [],
        'val': [],
        'err': []
    }
    behavior_frac_dict = copy.deepcopy(behavior_count_dict)
    behavior_pc_count_dict = copy.deepcopy(behavior_count_dict)
    behavior_pc_frac_dict = copy.deepcopy(behavior_count_dict)
    behavior_nuf_count_dict = copy.deepcopy(behavior_count_dict)
    behavior_nuf_frac_dict = copy.deepcopy(behavior_count_dict)

    def _fill_item(d, grp, ds_id, beh, val, err):
        d['group'].append(grp)
        d['ds_id'].append(ds_id)
        d['behavior'].append(beh)
        d['val'].append(val)
        d['err'].append(err)

    for k, ds in behavior_count.items():
        for ds_id, a in ds.items():
            n_tr = np.sum(a)
            if n_tr == 0:
                continue
            for ki, ni in enumerate(a):
                sq_ni = np.sqrt(ni)
                _fill_item(behavior_count_dict, k, ds_id, behavior_names[ki], ni, sq_ni)
                _fill_item(behavior_frac_dict, k, ds_id, behavior_names[ki], ni / n_tr, sq_ni / n_tr)
    for k, ds in behavior_pc_count.items():
        for ds_id, a in ds.items():
            n_tr = np.sum(a)
            if n_tr == 0:
                continue
            for ki, ni in enumerate(a):
                sq_ni = np.sqrt(ni)
                _fill_item(behavior_pc_count_dict, k, ds_id, behavior_pc_names[ki], ni, sq_ni)
                _fill_item(behavior_pc_frac_dict, k, ds_id, behavior_pc_names[ki], ni / n_tr, sq_ni / n_tr)
    for k, ds in behavior_nuf_count.items():
        for ds_id, a in ds.items():
            n_tr = np.sum(a)
            if n_tr == 0:
                continue
            for ki, ni in enumerate(a):
                sq_ni = np.sqrt(ni)
                _fill_item(behavior_nuf_count_dict, k, ds_id, behavior_nuf_names[ki], ni, sq_ni)
                _fill_item(behavior_nuf_frac_dict, k, ds_id, behavior_nuf_names[ki], ni / n_tr, sq_ni / n_tr)

    behavior_analytics_dict = {
        'all_count': {'name': 'Number of cells', 'val': behavior_count_dict},
        'all_frac': {'name': 'Fraction of adhered cells', 'val': behavior_frac_dict},
        'pc_count': {'name': 'Number of cells', 'val': behavior_pc_count_dict},
        'pc_frac': {'name': 'Fraction of adhered cells', 'val': behavior_pc_frac_dict},
        'nuf_count': {'name': 'Number of cells', 'val': behavior_nuf_count_dict},
        'nuf_frac': {'name': 'Fraction of adhered cells', 'val': behavior_nuf_frac_dict}

    }
    return behavior_analytics_dict


def _fill_item(d, grp, val, ds_id):
    d['group'].append(grp)
    d['val'].append(val)
    d['ds_id'].append(ds_id)


# noinspection PyPep8Naming
def get_statistics_analytics_dict(tas_dict, comparisson_group, condition_id_to_ds_id):
    #  n_probe/crawling_track,
    # n_am/track n_diap_events per track, n_p_diap_events per track (adhered?),
    # n_rev, distribution of p-, and diap events time see paper,

    cell_count = {'group': [], 'val': [], 'ds_id': []}
    cell_det_count = copy.deepcopy(cell_count)

    cell_UT_attempts = copy.deepcopy(cell_count)
    cell_FT_attempts = copy.deepcopy(cell_count)
    cell_RT_attempts = copy.deepcopy(cell_count)
    cell_UT_attempts_pt = copy.deepcopy(cell_count)  # per UT track
    cell_FT_attempts_pt = copy.deepcopy(cell_count)  # per FT track
    cell_RT_attempts_pt = copy.deepcopy(cell_count)  # per FT track (RT are rare, normalizing to them is irrelevant)
    cell_UT_tf_frac = copy.deepcopy(cell_count)
    cell_FT_tf_frac = copy.deepcopy(cell_count)
    cell_UT_area_frac_mean = copy.deepcopy(cell_count)
    cell_FT_area_frac_mean = copy.deepcopy(cell_count)
    cell_UT_area_frac_max = copy.deepcopy(cell_count)
    cell_FT_area_frac_max = copy.deepcopy(cell_count)

    #     cell_probing_attempts = copy.deepcopy(cell_count)
    #     cell_probing_horizon_f_dr = copy.deepcopy(cell_count)
    #     cell_probing_horizon_f_path = copy.deepcopy(cell_count)

    cell_det_time = copy.deepcopy(cell_count)  # min
    cell_UT_time = copy.deepcopy(cell_count)  # min
    cell_FT_time = copy.deepcopy(cell_count)  # min

    for group_id, condition_info in comparisson_group.items():
        condition_id = condition_info['condition_id']
        group_name = condition_info['name']

        ds_ids = condition_id_to_ds_id[condition_id]
        for ds_id in ds_ids:
            n_cells = 0
            n_det = 0

            n_fdiap = 0
            n_pdiap = 0
            n_rdiap = 0

            n_fdiap_att = 0
            n_pdiap_att = 0
            n_rdiap_att = 0

            ta: TrackAnalyzer or None = None
            for ta in tas_dict.get(ds_id, []):  # tas might be limited
                if not ta.is_ok:
                    continue

                n_cells += 1
                if ta.performed_detachment:
                    n_det += 1
                    _fill_item(cell_det_time, group_name, ta.detached_tf * cfgm.DT / 60., ds_id)

                if ta.performed_fdiap:
                    n_fdiap += 1

                    start_t_idx = [ta.all_times[idx] - ta.tf_min for idx in ta.fdiap_start_idxs]
                    n_fdiap_att += len(start_t_idx)
                    for t_idx in start_t_idx:
                        _fill_item(cell_FT_time, group_name, t_idx * cfgm.DT / 60., ds_id)

                    _fill_item(cell_FT_tf_frac, group_name, ta.fdiap_frac_tf, ds_id)
                    _fill_item(cell_FT_area_frac_mean, group_name, ta.fdiap_fraction, ds_id)
                    _fill_item(cell_FT_area_frac_max, group_name, ta.fdiap_max, ds_id)

                if ta.performed_pdiap:
                    n_pdiap += 1

                    start_t_idx = [ta.all_times[idx] - ta.tf_min for idx in ta.pdiap_start_idxs]
                    n_pdiap_att += len(start_t_idx)
                    for t_idx in start_t_idx:
                        _fill_item(cell_UT_time, group_name, t_idx * cfgm.DT / 60., ds_id)

                    _fill_item(cell_UT_tf_frac, group_name, ta.pdiap_frac_tf, ds_id)
                    _fill_item(cell_UT_area_frac_mean, group_name, ta.pdiap_fraction, ds_id)
                    _fill_item(cell_UT_area_frac_max, group_name, ta.pdiap_max, ds_id)

                if ta.performed_rdiap:
                    n_rdiap += 1
                    start_t_idx = [ta.all_times[idx] - ta.tf_min for idx in ta.rdiap_start_idxs]
                    n_rdiap_att += len(start_t_idx)

                # if ta.performed_probing:
                #     _fill_item(cell_probing_horizon_f_dr, group_name, ta.horizon_f_dr*cfgm.DT/60., ds_id)
                #     _fill_item(cell_probing_horizon_f_path, group_name, ta.horizon_f_path*cfgm.DT/60., ds_id)

            _fill_item(cell_count, group_name, n_cells, ds_id)
            _fill_item(cell_det_count, group_name, n_det, ds_id)

            _fill_item(cell_FT_attempts, group_name, n_fdiap_att, ds_id)
            if n_fdiap:
                _fill_item(cell_FT_attempts_pt, group_name, n_fdiap_att / n_fdiap, ds_id)

            _fill_item(cell_UT_attempts, group_name, n_pdiap_att, ds_id)
            if n_pdiap:
                _fill_item(cell_UT_attempts_pt, group_name, n_pdiap_att / n_pdiap, ds_id)

            _fill_item(cell_RT_attempts, group_name, n_rdiap_att, ds_id)
            if n_fdiap:
                _fill_item(cell_RT_attempts_pt, group_name, n_rdiap_att / n_fdiap, ds_id)

    statistics_analytics_dict = {
        'cell_count': {'name': 'cell count / ds', 'val': cell_count},
        'cell_det_count': {'name': 'detachment count / ds', 'val': cell_det_count},
        'cell_UT_attempts': {'name': 'UT attempts / ds', 'val': cell_UT_attempts},
        'cell_FT_attempts': {'name': 'FT attempts', 'val': cell_FT_attempts},
        'cell_RT_attempts': {'name': 'RT attempts', 'val': cell_RT_attempts},
        'cell_UT_attempts_pt': {'name': 'UT attempts/UT tr', 'val': cell_UT_attempts_pt},
        'cell_FT_attempts_pt': {'name': 'FT attempts/FT tr', 'val': cell_FT_attempts_pt},
        'cell_RT_attempts_pt': {'name': 'RT attempts/FT tr', 'val': cell_RT_attempts_pt},
        'cell_UT_tf_frac': {'name': 'UT time fraction', 'val': cell_UT_tf_frac},
        'cell_FT_tf_frac': {'name': 'FT time fraction', 'val': cell_FT_tf_frac},
        'cell_UT_area_frac_mean': {'name': 'UT cell area mean fraction', 'val': cell_UT_area_frac_mean},
        'cell_FT_area_frac_mean': {'name': 'FT cell area mean fraction', 'val': cell_FT_area_frac_mean},
        'cell_UT_area_frac_max': {'name': 'UT cell area max fraction', 'val': cell_UT_area_frac_max},
        'cell_FT_area_frac_max': {'name': 'FT cell area max fraction', 'val': cell_FT_area_frac_max},
        # 'cell_probing_attempts':        {'name':'', 'val': cell_probing_attempts},
        # 'cell_probing_horizon_f_dr':    {'name':'displ. probing time horizon, min', 'val': cell_probing_horizon_f_dr},
        # 'cell_probing_horizon_f_path':  {'name':'path probing time horizon, min', 'val': cell_probing_horizon_f_path},
        'cell_det_time': {'name': 'Detachment time, min', 'val': cell_det_time},
        'cell_UT_time': {'name': 'UT time, min', 'val': cell_UT_time},
        'cell_FT_time': {'name': 'FT time, min', 'val': cell_FT_time},
    }

    return statistics_analytics_dict


def get_analytics_dict(tas_dict, comparisson_group, condition_id_to_ds_id):
    motility_analytics_dict = get_motility_analytics_dict(tas_dict, comparisson_group, condition_id_to_ds_id)
    acc_movement_analytics_dict = get_acc_movement_analytics_dict(tas_dict, comparisson_group, condition_id_to_ds_id)
    behavior_analytics_dict = get_behavior_analytics_dict(tas_dict, comparisson_group, condition_id_to_ds_id)
    stats_analytics_dict = get_statistics_analytics_dict(tas_dict, comparisson_group, condition_id_to_ds_id)

    analytics_dict = {
        'behavior': behavior_analytics_dict,
        'motility': motility_analytics_dict,
        'acc_movement': acc_movement_analytics_dict,
        'stats': stats_analytics_dict,
    }

    return analytics_dict


def save_detachment_rois(tas_dict, datasets_dir):
    for ds_id, tas in tas_dict.items():
        xs = []
        ys = []
        ts = []
        ns = []
        for tr_idx, ta in enumerate(tas):
            if ta.performed_detachment:
                t = ta.all_times[-1]
                n = ta.all_nodes[-1]
                x, y, z = n.r
                xs.append(x)
                ys.append(y)
                ts.append(t)
                ns.append(f'{tr_idx:04d}_{x:.1f}_{y:.1f}_{t:03d}')
        fname = os.path.join(datasets_dir, f'{ds_id}', 'detachment_RoiSet.zip')
        save_roi_array(xs, ys, ts, fname=fname, names=ns)


def save_data_dict(d, filename):
    df = pd.DataFrame(d)
    df.to_csv(filename)


def save_motility_analytics(analysis_dict, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    motility_analysis_dict = analysis_dict['motility']
    for regime_sid, regime_analysis_info in motility_analysis_dict.items():
        regime_name = regime_analysis_info['name']
        regime_analysis_dict = regime_analysis_info['val']
        for idx, (param_sid, param_info) in enumerate(regime_analysis_dict.items()):
            param_name = param_info['name']
            param_dict = param_info['val']

            fname = f'{regime_sid}_{path_safe_str(regime_name)}__{param_sid}_{path_safe_str(param_name)}.csv'
            fname = os.path.join(save_dir, fname)
            save_data_dict(param_dict, fname)


def save_acc_movement_analytics(analysis_dict, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    am_analysis_dict = analysis_dict['acc_movement']

    for idx, (param_sid, param_info) in enumerate(am_analysis_dict.items()):
        param_name = param_info['name']
        param_dict = param_info['val']

        fname = f'{param_sid}_{path_safe_str(param_name)}.csv'
        fname = os.path.join(save_dir, fname)
        save_data_dict(param_dict, fname)


def save_behavior_analytics(analysis_dict, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    behavior_analysis_info = analysis_dict['behavior']

    for idx, (param_sid, param_info) in enumerate(behavior_analysis_info.items()):
        param_name = param_info['name']
        param_dict = param_info['val']

        fname = f'{param_sid}_{path_safe_str(param_name)}.csv'
        fname = os.path.join(save_dir, fname)
        save_data_dict(param_dict, fname)


def save_stats_analytics(analysis_dict, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    stats_analysis_info = analysis_dict['stats']
    for idx, (param_sid, param_info) in enumerate(stats_analysis_info.items()):
        param_name = param_info['name']
        param_dict = param_info['val']

        fname = f'{param_sid}_{path_safe_str(param_name)}.csv'
        fname = os.path.join(save_dir, fname)
        save_data_dict(param_dict, fname)


def save_analytics(analysis_dict, save_dir='.\\ds_analytics'):
    os.makedirs(save_dir, exist_ok=True)
    save_pckl(analysis_dict, os.path.join(save_dir, 'analysis_dict.pckl'))

    save_motility_analytics(analysis_dict=analysis_dict, save_dir=os.path.join(save_dir, 'motility'))
    save_acc_movement_analytics(analysis_dict=analysis_dict, save_dir=os.path.join(save_dir, 'acc_movement'))
    save_behavior_analytics(analysis_dict=analysis_dict, save_dir=os.path.join(save_dir, 'behavior'))
    save_stats_analytics(analysis_dict=analysis_dict, save_dir=os.path.join(save_dir, 'stats'))


def set_plot_sizes():
    plt.rcParams.update({'axes.labelsize': 20})
    plt.rcParams.update({'xtick.labelsize': 12})
    plt.rcParams.update({'ytick.labelsize': 12})
    plt.rcParams.update({'legend.fontsize': 16})
    plt.rcParams.update({'axes.titlesize': 20})
    plt.rcParams.update({'font.size': 20})


def plot_motility_comparisson_regime(analysis_dict, regime_sid, mode, stat):
    regime_analysis_info = analysis_dict['motility'][regime_sid]
    regime_name = regime_analysis_info['name']
    regime_analysis_dict = regime_analysis_info['val']

    n_plot = len(regime_analysis_dict)
    n_row = (n_plot + 2 - 1) // 2
    n_col = 2
    fig, ax = plt.subplots(nrows=n_row, ncols=n_col, figsize=(9 * n_col, 5 * n_row), dpi=100)

    for idx, (param_sid, param_info) in enumerate(regime_analysis_dict.items()):
        try:
            r = idx // n_col
            c = idx - r * n_col
            ax_ij = ax[r][c]

            param_name = param_info['name']
            param_dict = param_info['val']

            plot_df = pd.DataFrame(param_dict)

            if mode == 'bar':
                sns.barplot(data=plot_df, x='group', y='val', ax=ax_ij)
            else:
                sns.violinplot(x='group', y='val', data=plot_df, cut=0, ax=ax_ij)
            # see section compare motility params, manual¶

            types = list(set(plot_df['group']))
            pairs = [(p1, p2) for i, p1 in enumerate(types[:-1]) for j, p2 in enumerate(types[i + 1:])]

            ax_ij.set(ylabel=param_name, xlabel='')
            labels_formatted = [label.get_text() if i % 2 == 0 else '\n\n' + label.get_text() for
                                i, label in enumerate(ax_ij.xaxis.get_majorticklabels())]
            ax_ij.set_xticklabels(labels_formatted)

            if param_sid == 'mi':
                ax_ij.set_ylim(bottom=1, top=5)
            if param_sid == 'dir':
                ax_ij.set_ylim(bottom=0, top=1)

            if stat:
                annotator = Annotator(ax_ij, pairs, data=plot_df, x='group', y='val', verbose=False)
                annotator.configure(test='t-test_ind', text_format='star', loc='inside')
                annotator.apply_and_annotate()
        except Exception as ex:
            continue

    fig.suptitle(regime_name)
    plt.tight_layout(pad=1.5, h_pad=0, w_pad=1)
    return fig


def plot_acc_movement_comparisson(analysis_dict, mode, stat):
    am_analysis_dict = analysis_dict['acc_movement']

    n_plot = len(am_analysis_dict)
    n_row = (n_plot + 2 - 1) // 2
    n_col = 2
    fig, ax = plt.subplots(nrows=n_row, ncols=n_col, figsize=(9 * n_col, 5 * n_row), dpi=100)

    if n_row == 1:
        ax = [ax]
    for idx, (param_sid, param_info) in enumerate(am_analysis_dict.items()):
        try:
            r = idx // n_col
            c = idx - r * n_col
            ax_ij = ax[r][c]

            param_name = param_info['name']
            param_dict = param_info['val']

            plot_df = pd.DataFrame(param_dict)

            if mode == 'bar':
                sns.barplot(data=plot_df, x='group', y='val', ax=ax_ij)
            else:
                sns.violinplot(x='group', y='val', data=plot_df, cut=0, ax=ax_ij)
            # see section compare motility params, manual¶

            types = list(set(plot_df['group']))
            pairs = [(p1, p2) for i, p1 in enumerate(types[:-1]) for j, p2 in enumerate(types[i + 1:])]

            ax_ij.set(ylabel=param_name, xlabel='')
            labels_formatted = [label.get_text() if i % 2 == 0 else '\n\n' + label.get_text() for
                                i, label in enumerate(ax_ij.xaxis.get_majorticklabels())]
            ax_ij.set_xticklabels(labels_formatted)

            if stat:
                annotator = Annotator(ax_ij, pairs, data=plot_df, x='group', y='val', verbose=False)
                annotator.configure(test='t-test_ind', text_format='star', loc='inside')
                annotator.apply_and_annotate()
        except Exception as ex:
            continue

    fig.suptitle('Accelerated movement')
    plt.tight_layout(pad=1.5, h_pad=0, w_pad=1)
    return fig


def plot_behavior_distribution(analysis_dict):
    behavior_analysis_info = analysis_dict['behavior']

    n_plot = len(behavior_analysis_info)
    n_row = (n_plot + 2 - 1) // 2
    n_col = 2
    fig, ax = plt.subplots(nrows=n_row, ncols=n_col, figsize=(9 * n_col, 5 * n_row), dpi=100)
    if n_row == 1:
        ax = [ax]

    for idx, (param_sid, param_info) in enumerate(behavior_analysis_info.items()):
        try:
            r = idx // n_col
            c = idx - r * n_col
            ax_ij = ax[r][c]

            param_name = param_info['name']
            param_dict = param_info['val']

            plot_df = pd.DataFrame(param_dict)

            sns.barplot(data=plot_df, x='behavior', y='val', hue='group', ax=ax_ij)

            max_val = np.max(param_dict['val'])
            if max_val < 1:
                ax_ij.set_ylim(0, .5 if max_val < 0.5 else 1.)

            ax_ij.set_ylabel(param_name)
            ax_ij.set_xlabel('')
            labels_formatted = [label.get_text() if i % 2 == 0 else '\n\n' + label.get_text() for
                                i, label in enumerate(ax_ij.xaxis.get_majorticklabels())]
            ax_ij.set_xticklabels(labels_formatted)

            # x_coords = [p.get_x() + 0.5*p.get_width() for p in ax_ij.patches]
            # y_coords = [p.get_height() for p in ax_ij.patches]
            # ax_ij.errorbar(x=x_coords, y=y_coords, yerr=param_dict['err'], fmt="none", c= "k", alpha=0.3)
            # ax_ij.set_title(param_sid)
        except Exception as ex:
            continue

    fig.suptitle('Behavior')
    plt.tight_layout(pad=1.5, h_pad=0, w_pad=1)
    return fig


def plot_stats_distribution(analysis_dict, mode, stat):
    stats_analysis_info = analysis_dict['stats']

    n_plot = len(stats_analysis_info)
    n_row = (n_plot + 2 - 1) // 2
    n_col = 2
    fig, ax = plt.subplots(nrows=n_row, ncols=n_col, figsize=(9 * n_col, 5 * n_row), dpi=100)

    for idx, (param_sid, param_info) in enumerate(stats_analysis_info.items()):
        try:
            r = idx // n_col
            c = idx - r * n_col
            ax_ij = ax[r][c]

            param_name = param_info['name']
            param_dict = param_info['val']

            plot_df = pd.DataFrame(param_dict)

            if mode == 'bar':
                sns.barplot(data=plot_df, x='group', y='val', ax=ax_ij)
            else:
                sns.violinplot(data=plot_df, x='group', y='val', bw=.25, cut=0, ax=ax_ij)

            # max_val = np.max(param_dict['val'])

            ax_ij.set_ylabel(param_name)
            ax_ij.set_xlabel('')
            labels_formatted = [label.get_text() if i % 2 == 0 else '\n\n' + label.get_text() for
                                i, label in enumerate(ax_ij.xaxis.get_majorticklabels())]
            ax_ij.set_xticklabels(labels_formatted)

            ax_ij.set_title(param_sid)

            types = list(set(plot_df['group']))
            pairs = [(p1, p2) for i, p1 in enumerate(types[:-1]) for j, p2 in enumerate(types[i + 1:])]

            ax_ij.set(ylabel=param_name, xlabel='')

            if stat:
                annotator = Annotator(ax_ij, pairs, data=plot_df, x='group', y='val', verbose=False)
                annotator.configure(test='t-test_ind', text_format='star', loc='inside')
                annotator.apply_and_annotate()
        except Exception as ex:
            continue

    fig.suptitle('Stats')
    plt.tight_layout(pad=1.5, h_pad=0, w_pad=1)
    return fig


def plot_group_comparisson(analysis_dict,
                           save_dir='.\\ds_plot', filename='motility_info',
                           mode='violin', stat=True,  # or 'bar'
                           show=False, palette=cp_3t_a):
    os.makedirs(save_dir, exist_ok=True)
    if palette is not None:
        sns.set_palette(palette)
    sfx = '_s' if stat else ''
    fn = os.path.join(save_dir, filename + f'_{mode}{sfx}.pdf')
    set_plot_sizes()
    with PdfPages(fn) as pdf:
        # 1. Motility
        regime_sids = list(analysis_dict['motility'].keys())
        for regime_sid in regime_sids:
            fig = plot_motility_comparisson_regime(analysis_dict, regime_sid, mode=mode, stat=stat)
            pdf.savefig(fig)
            if show:
                plt.show()
            plt.close()

        # 2. AM
        fig = plot_acc_movement_comparisson(analysis_dict, mode=mode, stat=stat)
        pdf.savefig(fig)
        if show:
            plt.show()
        plt.close()

        # 3. Behavior
        fig = plot_behavior_distribution(analysis_dict)
        pdf.savefig(fig)
        if show:
            plt.show()
        plt.close()

        # 4. Stats
        fig = plot_stats_distribution(analysis_dict, mode=mode, stat=stat)
        pdf.savefig(fig)
        if show:
            plt.show()
        plt.close()


# # Proc pipeline

def proc_resolve_xings(datasets_ids, datasets_dir,
                       use_ds_priors=False, skip_processed_xing=True
                       ):
    # 1. resolve xings
    t_0 = timer()
    priors = get_priors_from_datasets(datasets_ids, path=datasets_dir) if use_ds_priors else None
    t_pr = timer()
    print(f'priors filled, {(t_pr - t_0):.1f} sec')

    resolved_tracks, skipped_ids = resolve_track_xings_datasets(datasets_ids, priors,
                                                                path=datasets_dir, skip_processed=skip_processed_xing)

    t_1 = timer()
    print(f'total merging time, {(t_1 - t_0):.1f} sec')

    # 2. select tracks
    ok_tracks, long_tracks = filter_ok_long_tracks(resolved_tracks, n_nodes_ok=6, n_nodes_long=30)
    (long_tracks_fv, all_tracks_fv,
     long_tracks_phf, long_tracks_fv_phf,
     all_tracks_phf, all_tracks_fv_phf) = filter_fv_physflow_tracks(ok_tracks, long_tracks)

    n = get_total_tracks_num(long_tracks)
    print('Total tracks:', n)

    return (long_tracks_fv, all_tracks_fv,
            long_tracks_phf, long_tracks_fv_phf,
            all_tracks_phf, all_tracks_fv_phf,
            skipped_ids)


def proc_plot_track_info(all_tracks_fv, long_tracks_phf, long_tracks_fv_phf, all_tracks_fv_phf,
                         condition_id_to_condition_name, condition_id_to_ds_id, ds_id_to_condition_id,
                         skipped_ids,
                         plot_dir
                         ):
    os.makedirs(plot_dir, exist_ok=True)
    plot_dataset_speed_distributions(long_tracks_fv_phf)
    plot_tracks_datasets_all_vis({k: v for k, v in long_tracks_phf.items() if k not in skipped_ids})

    plot_datasets_att_det_distribution(ds_id_to_condition_id,
                                       condition_id_to_condition_name,
                                       all_tracks_fv,
                                       save_dir=plot_dir)
    plot_datasets_att_det_distribution(ds_id_to_condition_id,
                                       condition_id_to_condition_name,
                                       all_tracks_fv, t_max=50,
                                       save_dir=plot_dir)

    plot_groups_att_det_distribution(condition_id_to_condition_name=condition_id_to_condition_name,
                                     condition_id_to_ds_id=condition_id_to_ds_id,
                                     tracks=all_tracks_fv_phf, t_max=200,
                                     save_dir=plot_dir)
    plot_groups_att_det_distribution(condition_id_to_condition_name=condition_id_to_condition_name,
                                     condition_id_to_ds_id=condition_id_to_ds_id,
                                     tracks=all_tracks_fv_phf, t_max=50,
                                     save_dir=plot_dir)

    plot_phf_detachment_rate(condition_id_to_condition_name=condition_id_to_condition_name,
                             condition_id_to_ds_id=condition_id_to_ds_id,
                             tracks=all_tracks_fv_phf, save_dir=plot_dir)


def proc_analyze_tracks(all_tracks_fv_phf, long_tracks_fv_phf, all_tracks_fv,
                        datasets_dir, plot_dir,
                        skip_processed=False, no_transm_mode=False):
    neighbour_effect_radius = cfgm.NEIGHBOR_TRACK_DIST  # um, neighbours search radius
    prob_dr = cfgm.PROBING_MAX_DISPLACEMENT

    # plot only aggregated info
    cls_not_a_cell = Classificator.load(cfgm.CLASSIFIER_NAC)  # 'not_a_cell_classifier_024.lrp'
    cls_detached = None if no_transm_mode else Classificator.load(cfgm.CLASSIFIER_DET)  # 'detached_classifier_024.lrp'

    effs = []

    tas_all = {}
    for ds_id in all_tracks_fv_phf:
        tas_filename = os.path.join(datasets_dir, f'{ds_id}', 'tracks.tas')
        if skip_processed and os.path.exists(tas_filename):
            tas = load_pckl(tas_filename)
        else:
            trx, trx_long, trx_all = all_tracks_fv_phf[ds_id], long_tracks_fv_phf[ds_id], all_tracks_fv[ds_id]

            n_ch = NeighborhoodChecker(neighbour_effect_radius)
            n_ch.fill_container_points(trx_all)

            TrackAnalyzer.reset()

            tf_max = np.max([t.get_last_time() for t in trx] if len(trx) else [0])

            n_ok = 0
            eff = 0.
            errs = []
            tas = []
            for t in trx:
                n = t.get_num_nodes()
                if n > 6:
                    # try:
                    ta = TrackAnalyzer(t, tf_max=tf_max,
                                       classificator_det=cls_detached, classificator_not_a_cell=cls_not_a_cell,
                                       neighbourhood_checker=n_ch,
                                       probing_max_displ=prob_dr,
                                       )
                    ta.analyze()

                    if ta.is_ok:
                        n_ok += 1
                        tas.append(ta)

                    # noinspection PyRedundantParentheses
                    err_i = ((~ta.performed_run) * 1 +
                             (ta.not_a_cell) * 2 +
                             (ta.detected_diap_ineff) * 4 +
                             (ta.detected_tracking_ineff) * 8 +
                             (ta.performed_detachment) * 16
                             )
                    errs.append(err_i)

            n_long = len(trx_long)
            if n_long:
                eff = n_ok / n_long
                effs.append(eff)

            save_pckl(tas, tas_filename)

            print('dataset', ds_id, f'n_ok={n_ok}, n_tot={len(trx)}, n_long={len(trx_long)}, eff_long={eff}')

        tas_all[ds_id] = tas

    effs = np.array(effs)

    mask = effs > 0.6
    if len(effs):
        print(f'fraction accepted datasets: {sum(mask) / len(effs):.3f}, ' +
              f'mean efficiency {np.mean(effs[mask]):.3f}')

    save_detachment_rois(tas_all, datasets_dir=datasets_dir)
    
    if plot_dir is not None:
        plot_neighbour_tracks_info(tas_all, save_dir=plot_dir)

    return tas_all


def process_datasets(datasets_ids,
                     condition_id_to_condition_name, condition_id_to_ds_id, ds_id_to_condition_id,
                     datasets_dir, plot_dir,

                     use_ds_priors=False,
                     skip_processed_xing=True, skip_processed_tr_ana=False,
                     no_transm_mode=False  # option for Binding assys: True - no transmigration in this case possible.
                     ):
    # 1. resolve xings
    (long_tracks_fv, all_tracks_fv,
     long_tracks_phf, long_tracks_fv_phf,
     all_tracks_phf, all_tracks_fv_phf,
     skipped_ids) = proc_resolve_xings(datasets_ids, datasets_dir,
                                       use_ds_priors=use_ds_priors, skip_processed_xing=skip_processed_xing)

    # 2. make & save plots:
    if plot_dir is not None:
        proc_plot_track_info(all_tracks_fv, long_tracks_phf, long_tracks_fv_phf, all_tracks_fv_phf,
                             condition_id_to_condition_name, condition_id_to_ds_id, ds_id_to_condition_id,
                             skipped_ids, plot_dir
                             )

    # 3. analyze tracks
    tas_all = proc_analyze_tracks(all_tracks_fv_phf, long_tracks_fv_phf, all_tracks_fv,
                                  datasets_dir=datasets_dir, plot_dir=plot_dir,
                                  skip_processed=skip_processed_tr_ana,
                                  no_transm_mode=no_transm_mode
                                  )

    return tas_all


def analyze_group(tas_all, comparisson_groups, condition_id_to_ds_id, study_dir, palette=cp_3t_a, show=False):
    analytics_dict = get_analytics_dict(tas_all, comparisson_groups, condition_id_to_ds_id)
    save_analytics(analytics_dict, save_dir=study_dir)
    for stat in [False, True]:
        for mode in ['violin', 'bar']:
            plot_group_comparisson(analytics_dict, save_dir=study_dir, show=show, mode=mode, stat=stat, palette=palette)
