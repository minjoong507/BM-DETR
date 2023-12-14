import torch
from torch.utils.data import Dataset
import numpy as np
import random
import logging
from os.path import join
from utils.basic_utils import load_jsonl, load_json, l2_normalize_np_array, visual_feature_sampling
from utils.tensor_utils import pad_sequences_1d
from bm_detr.span_utils import span_xx_to_cxw
import os

logger = logging.getLogger(__name__)


class StartEndDataset(Dataset):
    """One line in data loaded from data_path."
    {
      "qid": 7803,
      "query": "Man in gray top walks from outside to inside.",
      "duration": 150,
      "vid": "RoripwjYFp8_360.0_510.0",
      "relevant_clip_ids": [13, 14, 15, 16, 17],
      "relevant_windows": [[26, 36]]
    }

    Note that only annotations in QVHighlights have relevant_clip_ids and saliency_scores.
    """

    def __init__(self, dset_name, data_path, v_feat_dirs, q_feat_dir,
                 q_feat_type="clip", max_q_l=32, max_v_l=75, data_ratio=1.0, ctx_mode="video",
                 normalize_v=False, normalize_t=False, load_labels=True,
                 clip_len=2, max_windows=5, span_loss_type="l1", txt_drop_ratio=0, window_size=2,
                 use_temporal_shifting=False):
        self.dset_name = dset_name
        self.data_path = data_path
        self.data_ratio = data_ratio
        self.v_feat_dirs = v_feat_dirs \
            if isinstance(v_feat_dirs, list) else [v_feat_dirs]
        self.q_feat_dir = q_feat_dir
        self.q_feat_type = q_feat_type
        self.max_q_l = max_q_l
        self.max_v_l = max_v_l
        self.fixed_length = False
        self.ctx_mode = ctx_mode
        self.use_tef = "tef" in ctx_mode
        self.use_video = "video" in ctx_mode
        self.normalize_t = normalize_t
        self.normalize_v = normalize_v
        self.load_labels = load_labels
        self.clip_len = clip_len
        self.max_windows = max_windows  # maximum number of windows to use as labels
        self.window_size = window_size
        self.span_loss_type = span_loss_type
        self.txt_drop_ratio = txt_drop_ratio
        self.use_nq = False if "val" in data_path or "test" in data_path else True
        self.use_temporal_shifting = use_temporal_shifting
        if "val" in data_path or "test" in data_path:
            assert txt_drop_ratio == 0

        # data
        self.vid_list = []
        self.data = self.load_data()

    def load_data(self):
        datalist = load_jsonl(self.data_path)

        if 'hl' == self.dset_name:
            if self.data_ratio != 1:
                n_examples = int(len(datalist) * self.data_ratio)
                datalist = datalist[:n_examples]
                logger.info("Using {}% of the data: {} examples"
                            .format(self.data_ratio * 100, n_examples))

            return datalist

        # Check all vid features are available.
        filtered_datalist = []
        for _feat_dir in self.v_feat_dirs:
            self.vid_list.extend(os.listdir(_feat_dir))
        self.vid_list = list(set(self.vid_list))
        self.vid_list = [vid.split('.')[0] for vid in self.vid_list]

        for data in datalist:
            moment_length = abs(data['relevant_windows'][0][1] - data['relevant_windows'][0][0])
            if moment_length >= data['duration']:
                continue
            if data['vid'] in self.vid_list:
                filtered_datalist.append(data)

        print('origin: {}, filtered: {}, dropped: {}'.format(len(datalist), len(filtered_datalist),
                                                             len(datalist) - len(filtered_datalist)))

        if self.data_ratio != 1:
            n_examples = int(len(filtered_datalist) * self.data_ratio)
            filtered_datalist = filtered_datalist[:n_examples]
            logger.info("Using {}% of the data: {} examples"
                        .format(self.data_ratio * 100, n_examples))

        return filtered_datalist

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        meta = self.data[index]

        model_inputs = dict()
        model_inputs["query_feat"] = self._get_query_feat_by_qid(meta["qid"])  # (Dq, ) or (Lq, Dq)

        if self.use_video:
            model_inputs["video_feat"] = self._get_video_feat_by_vid(meta["vid"])  # (Lv, Dv)
            ctx_l = len(model_inputs["video_feat"])
        else:
            ctx_l = self.max_v_l

        if self.use_tef:
            tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
            tef_ed = tef_st + 1.0 / ctx_l
            tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)

            if self.use_video:
                model_inputs["video_feat"] = torch.cat(
                    [model_inputs["video_feat"], tef], dim=1)  # (Lv, Dv+2)
            else:
                model_inputs["video_feat"] = tef

        if self.load_labels:
            model_inputs["relevant_windows"] = self.get_relevant_windows(meta["relevant_windows"], ctx_l)
            model_inputs["span_labels"] = self.get_span_labels(meta["relevant_windows"], ctx_l)  # (#windows, 2)

            if "hl" == self.dset_name:
                model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"] = \
                    self.get_saliency_labels(meta["relevant_clip_ids"], meta["saliency_scores"], ctx_l)
            else:
                model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"] = \
                    self.get_saliency_labels_sub_as_query(meta["relevant_windows"][0], ctx_l, meta=meta)  # only one gt

        if self.use_nq:
            if 'neg_query_ids' not in meta.keys():
                qid_pool = [anno['qid'] for anno in self.data if anno['vid'] != meta['vid']]
                neg_qid = random.choice(qid_pool)
                model_inputs["neg_query_feats"] = self._get_query_feat_by_qid(neg_qid)
            else:
                if len(meta['neg_query_ids']) > 0:
                    neg_qid = random.sample(meta['neg_query_ids'], k=1)[0]
                    model_inputs["neg_query_feats"] = self._get_query_feat_by_qid(neg_qid)
                else:
                    qid_pool = [anno['qid'] for anno in self.data if anno['vid'] != meta['vid']]
                    neg_qid = random.choice(qid_pool)
                    model_inputs["neg_query_feats"] = self._get_query_feat_by_qid(neg_qid)

        if self.use_temporal_shifting:
            _gt_window = self.get_relevant_windows(meta["relevant_windows"], ctx_l)
            shifted_vid_feat, shifted_window = self.get_temporal_shifted_feats(model_inputs["video_feat"].clone(),
                                                                               meta["relevant_windows"][0], ctx_l)
            shifted_span_labels = self.get_span_labels([shifted_window], ctx_l)
            shifted_saliency_pos_labels, shifted_saliency_neg_labels, = \
                self.get_saliency_labels_sub_as_query(shifted_window, ctx_l)
            model_inputs.update(shifted_video_feat=shifted_vid_feat,
                                shifted_relevant_windows=shifted_window,
                                shifted_span_labels=shifted_span_labels,
                                shifted_saliency_pos_labels=shifted_saliency_pos_labels,
                                shifted_saliency_neg_labels=shifted_saliency_neg_labels)

        return dict(meta=meta, model_inputs=model_inputs)

    def _use_tef(self, vid_feat, ctx_l):
        tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
        tef_ed = tef_st + 1.0 / ctx_l
        tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
        if self.use_video:
            return torch.cat([vid_feat, tef], dim=1)  # (Lv, Dv+2)
        else:
            return tef

    def get_relevant_windows(self, gt_windows, ctx_l):
        moment_lengths = []
        _gt_windows = []
        for gt_window in gt_windows:
            if gt_window[0] > gt_window[1]:
                gt_window[1] = gt_window[0]
            moment_lengths.append(gt_window[1] - gt_window[0])
            gt_st = int(gt_window[0] / self.clip_len)
            gt_ed = max(0, min(int(gt_window[1] / self.clip_len), ctx_l) - 1)

            if gt_st > gt_ed:
                gt_st = gt_ed

            _gt_windows.append([gt_st, gt_ed])

        return _gt_windows

    def get_temporal_shifted_feats(self, vid_feat, gt_window, ctx_l):
        st = int(gt_window[0] / self.clip_len)
        ed = max(0, min(int(gt_window[1] / self.clip_len), ctx_l) - 1)

        if st > ed:
            st = ed

        moment_length = ed - st
        if moment_length + 1 == len(vid_feat):
            return vid_feat, gt_window

        pos_pool = list(np.arange(st, ed + 1))
        neg_pool = list(range(0, st)) + list(range(ed + 1, ctx_l))

        shifted_st = random.sample(range(len(neg_pool)), k=1)[0]
        shifted_ed = shifted_st + moment_length
        shifted_frame_indices = neg_pool[:shifted_st] + pos_pool + neg_pool[shifted_st:]
        shifted_vid_feat = vid_feat[shifted_frame_indices]
        assert len(shifted_vid_feat) == len(vid_feat), 'shifting function error 1.'
        assert torch.sum(vid_feat[st:ed] - shifted_vid_feat[shifted_st: shifted_ed]) == 0, 'shifting function error 2.'
        return shifted_vid_feat, [shifted_st, shifted_ed]

    def get_shifted_saliency_labels(self, gt_windows, ctx_l, max_n=2, meta=None):
        pos_clip_indices = []
        for gt_window in gt_windows:
            gt_st = int(gt_window[0] / self.clip_len)
            gt_ed = max(0, min(int(gt_window[1] / self.clip_len), ctx_l) - 1)

            if gt_st > gt_ed:
                gt_st = gt_ed
            pos_clip_indices.extend(list(range(gt_st, gt_ed + 1)))

        neg_pool = list(set(list(range(0, ctx_l))) - set(pos_clip_indices))
        pos_clip_indices = random.sample(pos_clip_indices, k=max_n)
        # neg_clip_indices = []

        if len(neg_pool) == 1:
            neg_clip_indices = random.sample(neg_pool, k=1) * 2
        else:
            neg_clip_indices = random.sample(neg_pool, k=max_n)

        return pos_clip_indices, neg_clip_indices

    def get_saliency_labels_sub_as_query(self, gt_window, ctx_l, max_n=2, meta=None):
        gt_st = int(gt_window[0] / self.clip_len)
        gt_ed = max(0, min(int(gt_window[1] / self.clip_len), ctx_l) - 1)

        if gt_st > gt_ed:
            gt_st = gt_ed

        if gt_st != gt_ed:
            pos_clip_indices = random.sample(range(gt_st, gt_ed + 1), k=max_n)
        else:
            pos_clip_indices = [gt_st, gt_st]

        neg_pool = list(range(0, gt_st)) + list(range(gt_ed + 1, ctx_l))

        if len(neg_pool) == 0:
            neg_clip_indices = [0, 0]
            # raise ValueError("qid: {}, st: {}, ed: {}, ctx_l: {}".format(meta['qid'], gt_st, gt_ed, ctx_l), gt_window, meta)
        elif len(neg_pool) == 1:
            neg_clip_indices = random.sample(neg_pool, k=1) * 2
        else:
            neg_clip_indices = random.sample(neg_pool, k=max_n)

        return pos_clip_indices, neg_clip_indices

    def get_saliency_labels(self, rel_clip_ids, scores, ctx_l, max_n=1, add_easy_negative=True):
        """Sum the scores from the three annotations, then take the two clips with the
        maximum scores as positive, and two with the minimum scores as negative.
        Args:
            rel_clip_ids: list(int), list of relevant clip ids
            scores: list([anno1_score, anno2_score, anno3_score]),
            ctx_l: int
            max_n: int, #clips to use as positive and negative, for easy and hard negative, respectively.
            add_easy_negative: bool, if True, sample eay negative outside the relevant_clip_ids.
        """
        # indices inside rel_clip_ids
        scores = np.array(scores)  # (#rel_clips, 3)
        agg_scores = np.sum(scores, 1)  # (#rel_clips, )
        sort_indices = np.argsort(agg_scores)  # increasing

        # indices in the whole video
        # the min(_, ctx_l-1) here is incorrect, but should not cause
        # much troubles since this should be rarely used.
        hard_pos_clip_indices = [min(rel_clip_ids[idx], ctx_l - 1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(rel_clip_ids[idx], ctx_l - 1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)) - set(rel_clip_ids))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices
        return pos_clip_indices, neg_clip_indices

    def get_span_labels(self, windows, ctx_l):
        """
        windows: list([st, ed]) in seconds. E.g. [[26, 36]], corresponding st_ed clip_indices [[13, 17]] (inclusive)
            Note a maximum of `self.max_windows` windows are used.
        returns Tensor of shape (#windows, 2), each row is [center, width] normalized by video length
        """
        if len(windows) > self.max_windows:
            random.shuffle(windows)
            windows = windows[:self.max_windows]
        if self.span_loss_type == "l1":
            windows = torch.Tensor(windows) / (ctx_l * self.clip_len)  # normalized windows in xx
            windows = span_xx_to_cxw(windows)  # normalized windows in cxw
        elif self.span_loss_type == "ce":
            windows = torch.Tensor([
                [int(w[0] / self.clip_len), min(int(w[1] / self.clip_len), ctx_l) - 1]
                for w in windows]).long()  # inclusive
        else:
            raise NotImplementedError
        return windows

    def _get_query_feat_by_qid(self, qid, type="last_hidden_state"):
        q_feat_path = join(self.q_feat_dir, f"qid{qid}.npz")
        q_feat = np.load(q_feat_path)[type].astype(np.float32)[:self.max_q_l]

        if self.normalize_t:
            q_feat = l2_normalize_np_array(q_feat)
        if self.txt_drop_ratio > 0:
            q_feat = self.random_drop_rows(q_feat)
        if type == 'pooler_output':
            q_feat = q_feat.squeeze(0)

        return torch.from_numpy(q_feat)  # (D, ) or (Lq, D)

    def random_drop_rows(self, embeddings):
        """randomly mask num_drop rows in embeddings to be zero.
        Args:
            embeddings: np.ndarray (L, D)
        """
        num_drop_rows = round(len(embeddings) * self.txt_drop_ratio)
        if num_drop_rows > 0:
            row_indices = np.random.choice(
                len(embeddings), size=num_drop_rows, replace=False)
            embeddings[row_indices] = 0
        return embeddings

    def _get_video_feat_by_vid(self, vid):
        v_feat_list = []
        for _feat_dir in self.v_feat_dirs:
            if 'c3d' in _feat_dir or 'i3d' in _feat_dir or 'vgg' in _feat_dir:
                _feat_path = join(_feat_dir, f"{vid}.npy")
                if self.dset_name == 'tacos':
                    _feat_path = join(_feat_dir, f"{vid}.avi.npy")
                _feat = np.load(_feat_path)[:self.max_v_l].astype(np.float32)
            else:
                _feat_path = join(_feat_dir, f"{vid}.npz")
                if self.dset_name == 'tacos':
                    _feat_path = join(_feat_dir, f"{vid}-cam-002.npz")
                _feat = np.load(_feat_path)["features"][:self.max_v_l].astype(np.float32)
            if self.normalize_v:
                _feat = l2_normalize_np_array(_feat)
            v_feat_list.append(_feat)
        # some features are slightly longer than the others
        min_len = min([len(e) for e in v_feat_list])
        v_feat_list = [e[:min_len] for e in v_feat_list]
        v_feat = np.concatenate(v_feat_list, axis=1)
        return torch.from_numpy(v_feat)  # (Lv, D)

class FixedVidLengthDataset(Dataset):
    """One line in data loaded from data_path."
    {
      "qid": 7803,
      "query": "Man in gray top walks from outside to inside.",
      "duration": 150,
      "vid": "RoripwjYFp8_360.0_510.0",
      "relevant_clip_ids": [13, 14, 15, 16, 17],
      "relevant_windows": [[26, 36]]
    }
    """

    def __init__(self, dset_name, data_path, v_feat_dirs, q_feat_dir,
                 q_feat_type="clip",
                 max_q_l=32, max_v_l=75, data_ratio=1.0, ctx_mode="video",
                 normalize_v=False, normalize_t=False, load_labels=True,
                 max_windows=5, span_loss_type="l1", txt_drop_ratio=0,
                 use_nq=False, use_temporal_shifting=False):
        self.dset_name = dset_name
        self.data_path = data_path
        self.data_ratio = data_ratio
        self.v_feat_dirs = v_feat_dirs
        self.q_feat_dir = q_feat_dir
        self.q_feat_type = q_feat_type
        self.max_q_l = max_q_l
        self.max_v_l = max_v_l
        self.ctx_mode = ctx_mode
        self.use_tef = "tef" in ctx_mode
        self.use_video = "video" in ctx_mode
        self.normalize_t = normalize_t
        self.normalize_v = normalize_v
        self.load_labels = load_labels
        self.max_windows = max_windows
        self.span_loss_type = span_loss_type
        self.txt_drop_ratio = txt_drop_ratio
        self.use_nq = use_nq
        self.use_temporal_shifting = use_temporal_shifting
        if "val" in data_path or "test" in data_path:
            assert txt_drop_ratio == 0

        # data
        self.vid_list = []
        self.data = self.load_data()


    def load_data(self):
        datalist = load_jsonl(self.data_path)

        # Check all vid features are available.
        filtered_datalist = []
        for _feat_dir in self.v_feat_dirs:
            self.vid_list.extend(os.listdir(_feat_dir))
        self.vid_list = list(set(self.vid_list))
        self.vid_list = [vid.split('.')[0] for vid in self.vid_list]

        for data in datalist:
            if data['vid'] in self.vid_list:
                filtered_datalist.append(data)

        print('origin: {}, filtered: {}, dropped: {}'.format(len(datalist), len(filtered_datalist), len(datalist) - len(filtered_datalist)))

        if self.data_ratio != 1:
            n_examples = int(len(filtered_datalist) * self.data_ratio)
            filtered_datalist = filtered_datalist[:n_examples]
            logger.info("Using {}% of the data: {} examples"
                        .format(self.data_ratio * 100, n_examples))

        return filtered_datalist

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        meta = self.data[index]
        duration = meta['duration']

        model_inputs = dict()
        model_inputs["query_feat"] = self._get_query_feat_by_qid(meta["qid"])  # (Dq, ) or (Lq, Dq)

        if self.use_video:
            model_inputs["video_feat"] = self._get_video_feat_by_vid(meta["vid"])  # (Lv, Dv)
            ctx_l = len(model_inputs["video_feat"])
        else:
            ctx_l = self.max_v_l

        if self.use_tef:
            tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
            tef_ed = tef_st + 1.0 / ctx_l
            tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)

            if self.use_video:
                model_inputs["video_feat"] = torch.cat(
                    [model_inputs["video_feat"], tef], dim=1)  # (Lv, Dv+2)
            else:
                model_inputs["video_feat"] = tef

        if self.load_labels:
            model_inputs["relevant_windows"] = self.get_relevant_windows(meta["relevant_windows"], ctx_l, duration)
            model_inputs["span_labels"] = self.get_span_labels(meta["relevant_windows"], ctx_l, duration)  # (#windows, 2)
            model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"] = \
                self.get_saliency_labels_sub_as_query(meta["relevant_windows"][0], ctx_l, duration)  # only one gt

        if self.use_nq:
            if 'neg_query_ids' not in meta.keys():
                qid_pool = [anno['qid'] for anno in self.data if anno['vid'] != meta['vid']]
                neg_qid = random.choice(qid_pool)
                model_inputs["neg_query_feats"] = self._get_query_feat_by_qid(neg_qid)
            else:
                if len(meta['neg_query_ids']) > 0:
                    neg_qid = random.sample(meta['neg_query_ids'], k=1)[0]
                    model_inputs["neg_query_feats"] = self._get_query_feat_by_qid(neg_qid)
                else:
                    qid_pool = [anno['qid'] for anno in self.data if anno['vid'] != meta['vid']]
                    neg_qid = random.choice(qid_pool)
                    model_inputs["neg_query_feats"] = self._get_query_feat_by_qid(neg_qid)

        if self.use_temporal_shifting:
            shifted_vid_feat, shifted_window = self.get_temporal_shifted_feats(model_inputs["video_feat"].clone(),
                                                                               meta["relevant_windows"][0], ctx_l, duration)
            shifted_span_labels = self.get_span_labels([shifted_window], ctx_l, duration)
            shifted_saliency_pos_labels, shifted_saliency_neg_labels, = \
                self.get_saliency_labels_sub_as_query(shifted_window, ctx_l, duration)
            model_inputs.update(shifted_video_feat=shifted_vid_feat,
                                shifted_relevant_windows=shifted_window,
                                shifted_span_labels=shifted_span_labels,
                                shifted_saliency_pos_labels=shifted_saliency_pos_labels,
                                shifted_saliency_neg_labels=shifted_saliency_neg_labels)

        return dict(meta=meta, model_inputs=model_inputs)

    def _use_tef(self, vid_feat, ctx_l):
        tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
        tef_ed = tef_st + 1.0 / ctx_l
        tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
        if self.use_video:
            return torch.cat([vid_feat, tef], dim=1)  # (Lv, Dv+2)
        else:
            return tef

    def get_relevant_windows(self, gt_windows, ctx_l, duration):
        _gt_windows = []
        clip_len = float(duration / ctx_l)

        for gt_window in gt_windows:
            if gt_window[0] > gt_window[1]:
                gt_window[1] = gt_window[0]
            gt_st = int(gt_window[0] / clip_len)
            gt_ed = max(0, min(int(gt_window[1] / clip_len), ctx_l) - 1)

            if gt_st > gt_ed:
                gt_st = gt_ed

            _gt_windows.append([gt_st, gt_ed])

        return _gt_windows

    def get_temporal_shifted_feats(self, vid_feat, gt_window, ctx_l, duration):
        clip_len = float(duration / ctx_l)
        st = int(gt_window[0] / clip_len)
        ed = max(0, min(int(gt_window[1] / clip_len), ctx_l) - 1)

        if st > ed:
            st = ed

        moment_length = ed - st
        if moment_length + 1 == len(vid_feat):
            return vid_feat, gt_window

        pos_pool = list(np.arange(st, ed + 1))
        neg_pool = list(range(0, st)) + list(range(ed + 1, ctx_l))

        shifted_st = random.sample(range(len(neg_pool)), k=1)[0]
        shifted_ed = shifted_st + moment_length
        shifted_frame_indices = neg_pool[:shifted_st] + pos_pool + neg_pool[shifted_st:]
        shifted_vid_feat = vid_feat[shifted_frame_indices]
        assert len(shifted_vid_feat) == len(vid_feat), 'shifting function error 1.'
        assert torch.sum(vid_feat[st:ed] - shifted_vid_feat[shifted_st: shifted_ed]) == 0, 'shifting function error 2.'
        return shifted_vid_feat, [shifted_st, shifted_ed]

    def get_saliency_labels_sub_as_query(self, gt_window, ctx_l, duration, max_n=2):
        clip_len = float(duration / ctx_l)
        gt_st = int(gt_window[0] / clip_len)
        gt_ed = max(0, min(int(gt_window[1] / clip_len), ctx_l) - 1)

        if gt_st > gt_ed:
            gt_st = gt_ed

        if gt_st != gt_ed:
            pos_clip_indices = random.sample(range(gt_st, gt_ed + 1), k=max_n)
        else:
            pos_clip_indices = [gt_st, gt_st]

        neg_pool = list(range(0, gt_st)) + list(range(gt_ed + 1, ctx_l))

        if len(neg_pool) == 0:
            neg_clip_indices = [0, 0]

        elif len(neg_pool) == 1:
            neg_clip_indices = random.sample(neg_pool, k=1) * 2
        else:
            neg_clip_indices = random.sample(neg_pool, k=max_n)

        return pos_clip_indices, neg_clip_indices

    def get_saliency_labels(self, rel_clip_ids, scores, ctx_l, max_n=1, add_easy_negative=True):
        """Sum the scores from the three annotations, then take the two clips with the
        maximum scores as positive, and two with the minimum scores as negative.
        Args:
            rel_clip_ids: list(int), list of relevant clip ids
            scores: list([anno1_score, anno2_score, anno3_score]),
            ctx_l: int
            max_n: int, #clips to use as positive and negative, for easy and hard negative, respectively.
            add_easy_negative: bool, if True, sample eay negative outside the relevant_clip_ids.
        """
        # indices inside rel_clip_ids
        scores = np.array(scores)  # (#rel_clips, 3)
        agg_scores = np.sum(scores, 1)  # (#rel_clips, )
        sort_indices = np.argsort(agg_scores)  # increasing

        # indices in the whole video
        # the min(_, ctx_l-1) here is incorrect, but should not cause
        # much troubles since this should be rarely used.
        hard_pos_clip_indices = [min(rel_clip_ids[idx], ctx_l - 1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(rel_clip_ids[idx], ctx_l - 1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)) - set(rel_clip_ids))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices
        return pos_clip_indices, neg_clip_indices

    def get_span_labels(self, windows, ctx_l, duration):
        """
        windows: list([st, ed]) in seconds. E.g. [[26, 36]], corresponding st_ed clip_indices [[13, 17]] (inclusive)
            Note a maximum of `self.max_windows` windows are used.
        returns Tensor of shape (#windows, 2), each row is [center, width] normalized by video length
        """
        if len(windows) > self.max_windows:
            random.shuffle(windows)
            windows = windows[:self.max_windows]
        if self.span_loss_type == "l1":
            windows = torch.Tensor(windows) / duration  # normalized windows in xx
            windows = span_xx_to_cxw(windows)  # normalized windows in cxw
        elif self.span_loss_type == "ce":
            windows = torch.Tensor([
                [int(w[0] / self.clip_len), min(int(w[1] / self.clip_len), ctx_l) - 1]
                for w in windows]).long()  # inclusive
        else:
            raise NotImplementedError
        return windows

    def _get_query_feat_by_qid(self, qid, type="last_hidden_state"):
        q_feat_path = join(self.q_feat_dir, f"qid{qid}.npz")
        q_feat = np.load(q_feat_path)[type].astype(np.float32)[:self.max_q_l]

        if self.normalize_t:
            q_feat = l2_normalize_np_array(q_feat)
        if self.txt_drop_ratio > 0:
            q_feat = self.random_drop_rows(q_feat)
        if type == 'pooler_output':
            q_feat = q_feat.squeeze(0)

        return torch.from_numpy(q_feat)  # (D, ) or (Lq, D)

    def random_drop_rows(self, embeddings):
        """randomly mask num_drop rows in embeddings to be zero.
        Args:
            embeddings: np.ndarray (L, D)
        """
        num_drop_rows = round(len(embeddings) * self.txt_drop_ratio)
        if num_drop_rows > 0:
            row_indices = np.random.choice(
                len(embeddings), size=num_drop_rows, replace=False)
            embeddings[row_indices] = 0
        return embeddings

    def _get_video_feat_by_vid(self, vid):
        _feat_path = join(self.v_feat_dirs[0], f"{vid}.npy")
        _feat = np.load(_feat_path).astype(np.float32)
        _feat = visual_feature_sampling(visual_feature=_feat,
                                        max_num_clips=self.max_v_l)
        return torch.from_numpy(_feat)  # (Lv, D)


def get_match_labels(relevant_windows, vid_len, device, non_blocking):
    match_labels = np.zeros(shape=(len(relevant_windows), vid_len), dtype=np.int32)
    for idx, span in enumerate(relevant_windows):
        span = span['spans']
        if isinstance(span[0], list):
            for st, ed in span:
                match_labels[idx][st:(ed + 1)] = 1
        else:
            st, ed = span[0], span[1]
            match_labels[idx][st:(ed + 1)] = 1
    return torch.tensor(match_labels, dtype=torch.long).to(device, non_blocking=non_blocking)


def start_end_collate(batch):
    batch_meta = [e["meta"] for e in batch]  # seems no need to collate ?
    model_inputs_keys = batch[0]["model_inputs"].keys()
    batched_data = dict()
    for k in model_inputs_keys:
        if "relevant_windows" in k or "span_labels" in k:
            batched_data[k] = [dict(spans=e["model_inputs"][k]) for e in batch]
            continue

        if "pos_labels" in k or "neg_labels" in k:
            batched_data[k] = torch.LongTensor([e["model_inputs"][k] for e in batch])
            continue

        batched_data[k] = pad_sequences_1d(
            [e["model_inputs"][k] for e in batch], dtype=torch.float32, fixed_length=None)

    return batch_meta, batched_data


def prepare_batch_inputs(batched_model_inputs, device, non_blocking=False, max_v_l=75):
    model_inputs = dict(
        src_txt=batched_model_inputs["query_feat"][0].to(device, non_blocking=non_blocking),
        src_txt_mask=batched_model_inputs["query_feat"][1].to(device, non_blocking=non_blocking),
        src_vid=batched_model_inputs["video_feat"][0].to(device, non_blocking=non_blocking),
        src_vid_mask=batched_model_inputs["video_feat"][1].to(device, non_blocking=non_blocking),
    )

    targets = {}

    if "span_labels" in batched_model_inputs:
        targets["span_labels"] = [
            dict(spans=e["spans"].to(device, non_blocking=non_blocking))
            for e in batched_model_inputs["span_labels"]
        ]

    if "saliency_pos_labels" in batched_model_inputs:
        for name in ["saliency_pos_labels", "saliency_neg_labels"]:
            targets[name] = batched_model_inputs[name].to(device, non_blocking=non_blocking)

    if "relevant_windows" in batched_model_inputs:
        bsz, length, _ = batched_model_inputs["video_feat"][0].shape
        match_labels = get_match_labels(batched_model_inputs['relevant_windows'], length, device, non_blocking)
        model_inputs.update(match_labels=match_labels)
        targets['match_labels'] = match_labels
        targets['logit_mask'] = model_inputs['src_vid_mask']

    if "neg_query_feats" in batched_model_inputs:
        model_inputs.update(
            neg_src_txt=batched_model_inputs["neg_query_feats"][0].to(device, non_blocking=non_blocking),
            neg_src_txt_mask=batched_model_inputs["neg_query_feats"][1].to(device, non_blocking=non_blocking),
        )

    if "shifted_video_feat" in batched_model_inputs:
        model_inputs.update(
            shifted_vid=batched_model_inputs["shifted_video_feat"][0].to(device, non_blocking=non_blocking),
            shifted_vid_mask=batched_model_inputs["shifted_video_feat"][1].to(device, non_blocking=non_blocking)
        )

        targets['shifting'] = dict(
            span_labels=[dict(spans=e["spans"].to(device, non_blocking=non_blocking)) for e in
                         batched_model_inputs["shifted_span_labels"]],
            saliency_pos_labels=batched_model_inputs['shifted_saliency_pos_labels'].to(device, non_blocking=non_blocking),
            saliency_neg_labels=batched_model_inputs['shifted_saliency_neg_labels'].to(device, non_blocking=non_blocking),
            logit_mask=model_inputs['shifted_vid_mask']
        )

        if "shifted_relevant_windows" in batched_model_inputs:
            bsz, length, _ = batched_model_inputs["shifted_video_feat"][0].shape
            shifted_match_labels = get_match_labels(batched_model_inputs['shifted_relevant_windows'],
                                                    length,
                                                    device,
                                                    non_blocking)
            model_inputs.update(shifted_match_labels=shifted_match_labels)
            targets['shifting']['match_labels'] = shifted_match_labels

    targets = None if len(targets) == 0 else targets

    return model_inputs, targets


def build_dataset(config):
    # return FixedVidLengthDataset(**config)
    return StartEndDataset(**config)
