# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
    DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
import math
import random
from torch import nn
from bm_detr.span_utils import generalized_temporal_iou, span_cxw_to_xx
from bm_detr.matcher import build_matcher
from bm_detr.transformer import build_transformer, inverse_sigmoid, AttentivePooling
from bm_detr.position_encoding import build_position_encoding
from bm_detr.misc import accuracy


class BMDETR(nn.Module):
    """ This is the BM-DETR model that performs video moment retrieval tasks. (borrowed from Moment-DETR)"""

    def __init__(self, transformer, position_embed, txt_position_embed, txt_dim, vid_dim,
                 num_queries, input_dropout, max_v_l=75, span_loss_type="l1", t_feat_type="clip",
                 use_txt_pos=False, n_input_proj=2, contrastive_hdim=256):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            position_embed: torch module of the position_embedding, See position_encoding.py
            txt_position_embed: position_embedding for text
            txt_dim: int, text query input dimension
            vid_dim: int, video feature input dimension
            num_queries: number of learnable spans, ie detection slot. This is the maximal number of predictions
            max_v_l: int, maximum #clips in videos
            span_loss_type: str, one of [l1, ce]
                l1: (center-x, width) regression.
                ce: (st_idx, ed_idx) classification.
            # foreground_thd: float, intersection over prediction >= foreground_thd: labeled as foreground
            # background_thd: float, intersection over prediction <= background_thd: labeled background
        """

        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.position_embed = position_embed
        self.txt_position_embed = txt_position_embed
        hidden_dim = transformer.d_model
        self.span_loss_type = span_loss_type
        self.t_feat_type = t_feat_type
        self.max_v_l = max_v_l
        span_pred_dim = 2 if span_loss_type == "l1" else max_v_l * 2
        self.span_embed = MLP(hidden_dim, hidden_dim, span_pred_dim, 3)
        self.class_embed = nn.Linear(hidden_dim, 2)  # 0: background, 1: foreground
        self.use_txt_pos = use_txt_pos
        self.n_input_proj = n_input_proj
        self.vid_att_pooling = AttentivePooling(hidden_dim)
        self.txt_att_pooling = AttentivePooling(hidden_dim)
        self.query_embed = nn.Embedding(num_queries, 2) # for learnable spans

        relu_args = [True] * 3
        relu_args[n_input_proj - 1] = False
        self.input_txt_proj = nn.Sequential(*[LinearLayer(txt_dim, hidden_dim, layer_norm=True,
                                                          dropout=input_dropout, relu=relu_args[0]),
                                              LinearLayer(hidden_dim, hidden_dim, layer_norm=True,
                                                          dropout=input_dropout, relu=relu_args[1]),
                                              LinearLayer(hidden_dim, hidden_dim, layer_norm=True,
                                                          dropout=input_dropout, relu=relu_args[2])
                                              ][:n_input_proj])

        self.input_vid_proj = nn.Sequential(*[LinearLayer(vid_dim, hidden_dim, layer_norm=True,
                                                          dropout=input_dropout, relu=relu_args[0]),
                                              LinearLayer(hidden_dim, hidden_dim, layer_norm=True,
                                                          dropout=input_dropout, relu=relu_args[1]),
                                              LinearLayer(hidden_dim, hidden_dim, layer_norm=True,
                                                          dropout=input_dropout, relu=relu_args[2])
                                              ][:n_input_proj])

        self.saliency_proj = nn.Linear(hidden_dim, 1)
        self.FPM = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2, bias=False),
                                 nn.Tanh(),
                                 nn.Linear(hidden_dim // 2, 1, bias=False)
                                 )

        self.contrastive_align_vid_proj = nn.Linear(hidden_dim, contrastive_hdim)
        self.contrastive_align_txt_proj = nn.Linear(hidden_dim, contrastive_hdim)

    def forward(self, src_txt, src_txt_mask, src_vid, src_vid_mask, **kwargs):

        """The forward expects two tensors:
               - src_txt: [batch_size, L_txt, D_txt]
               - src_txt_mask: [batch_size, L_txt], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer
               - src_vid: [batch_size, L_vid, D_vid]
               - src_vid_mask: [batch_size, L_vid], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer

            + Additional inputs:
               - negative_src_txt: [batch_size, L_txt, D_txt] or [batch_size, D_txt]
               - negative_src_txt_mask: [batch_size, L_txt], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer
               - shifted_vid: [batch_size, L_vid, D_vid]
               - shifted_vid_mask: [batch_size, L_vid], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer

            It returns a dict with the following elements:
               - "pred_spans": The normalized boxes coordinates for all queries, represented as
                               (center_x, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "shifting": Optional, only returned when temporal shifted videos are loaded. It has same output formats.
        """

        hs, reference, memory, vid_weights = self._forward(src_vid=src_vid, src_vid_mask=src_vid_mask,
                                                           src_txt=src_txt, src_txt_mask=src_txt_mask,
                                                           additional_inputs=kwargs)
        vid_logits, vid_att_weights = vid_weights[0], vid_weights[1]
        outputs_class = self.class_embed(hs)  # (#layers, batch_size, #queries, #classes)
        reference_before_sigmoid = inverse_sigmoid(reference)
        tmp = self.span_embed(hs)
        outputs_coord = tmp + reference_before_sigmoid

        if self.span_loss_type == "l1":
            outputs_coord = outputs_coord.sigmoid()

        out = {'pred_logits': outputs_class[-1], 'pred_spans': outputs_coord[-1]}
        out['vid_logits'] = vid_logits
        out["saliency_scores"] = self.saliency_proj(memory[:, :src_vid.shape[1]]).squeeze(-1)  # (bsz, L_vid)
        out["hs"] = hs

        vid_mem = memory[:, :src_vid.shape[1]]  # (bsz, L_vid, d)
        txt_mem = memory[:, src_vid.shape[1]:]  # (bsz, L_txt, d)

        pooled_vid, _ = self.vid_att_pooling(vid_mem, src_vid_mask)
        pooled_txt, _ = self.txt_att_pooling(txt_mem, src_txt_mask)

        pooled_vid = self.contrastive_align_vid_proj(pooled_vid)
        pooled_txt = self.contrastive_align_txt_proj(pooled_txt)

        proj_vid_mem = F.normalize(pooled_vid, p=2, dim=-1)
        proj_txt_mem = F.normalize(pooled_txt, p=2, dim=-1)
        out.update(dict(
            proj_txt_mem=proj_txt_mem,
            proj_vid_mem=proj_vid_mem
        ))

        shifted_vid, shifted_vid_mask, shifted_match_labels = self._load_feats(kwargs, keyword='shifted')

        if shifted_vid is not None:
            """
                If model inputs contain shifted video features,
                We get the model outputs from shifted video features ('s_' means 'shifted features').
            """

            s_hs, s_reference, s_memory, s_vid_weights = self._forward(src_vid=shifted_vid, src_vid_mask=shifted_vid_mask,
                                                                       src_txt=src_txt, src_txt_mask=src_txt_mask,
                                                                       additional_inputs=kwargs)
            s_vid_logits, s_vid_att_weights = s_vid_weights[0], s_vid_weights[1]
            s_outputs_class = self.class_embed(s_hs)  # (#layers, batch_size, #queries, #classes)
            s_reference_before_sigmoid = inverse_sigmoid(s_reference)
            tmp = self.span_embed(s_hs)  # (#layers, bsz, #queries, 2 or max_v_l * 2)
            s_outputs_coord = tmp + s_reference_before_sigmoid

            if self.span_loss_type == "l1":
                s_outputs_coord = s_outputs_coord.sigmoid()

            out['shifting'] = {'pred_logits': s_outputs_class[-1], 'pred_spans': s_outputs_coord[-1]}
            out['shifting']['vid_logits'] = s_vid_logits
            out['shifting']['saliency_scores'] = self.saliency_proj(s_memory[:, :shifted_vid.shape[1]]).squeeze(-1)  # (bsz, L_vid)
            out['shifting']['hs'] = s_hs  # (bsz, L_vid)

            s_vid_mem = s_memory[:, :shifted_vid.shape[1]]  # (bsz, L_vid, d)
            s_txt_mem = s_memory[:, shifted_vid.shape[1]:]  # (bsz, L_txt, d)

            s_pooled_vid, _ = self.vid_att_pooling(s_vid_mem, shifted_vid_mask)
            s_pooled_txt, _ = self.txt_att_pooling(s_txt_mem, src_txt_mask)
            s_pooled_vid = self.contrastive_align_vid_proj(s_pooled_vid)
            s_pooled_txt = self.contrastive_align_txt_proj(s_pooled_txt)

            s_proj_vid_mem = F.normalize(s_pooled_vid, p=2, dim=-1)
            s_proj_txt_mem = F.normalize(s_pooled_txt, p=2, dim=-1)
            out['shifting'].update(dict(
                proj_txt_mem=s_proj_txt_mem,
                proj_vid_mem=s_proj_vid_mem
            ))

        return out

    def _forward(self, src_vid, src_vid_mask, src_txt, src_txt_mask, additional_inputs=None):
        src_vid = self.input_vid_proj(src_vid)
        pos_vid = self.position_embed(src_vid, src_vid_mask)  # (bsz, L_vid, d)
        src_txt = self.input_txt_proj(src_txt)
        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)  # (bsz, L_txt, d)

        hs, reference, memory, target_frame_probs = self.calculate_probs(src_vid=src_vid, src_vid_mask=src_vid_mask, pos_vid=pos_vid,
                                                                         src_txt=src_txt, src_txt_mask=src_txt_mask, pos_txt=pos_txt,
                                                                         returns=True)

        # Compute probs with neg query
        neg_src_txt, neg_src_txt_mask = self._load_feats(additional_inputs, keyword='neg')

        if neg_src_txt is not None:
            neg_src_txt = self.input_txt_proj(neg_src_txt)
            pos_neg_txt = self.txt_position_embed(neg_src_txt) if self.use_txt_pos else torch.zeros_like(neg_src_txt)  # (bsz, L_txt, d)
            neg_frame_probs = self.calculate_probs(src_vid=src_vid, src_vid_mask=src_vid_mask, pos_vid=pos_vid,
                                                         src_txt=neg_src_txt, src_txt_mask=neg_src_txt_mask,
                                                         pos_txt=pos_neg_txt,
                                                         reverse=True)
            target_frame_probs = target_frame_probs * neg_frame_probs

        vid_logits = target_frame_probs * src_vid_mask
        vid_att_weight = F.softmax(vid_logits, dim=1)  # (bsz, L_vid)

        vid_weights = [vid_logits, vid_att_weight]
        vid_mem = memory[:, :src_vid.shape[1]]
        txt_mem = memory[:, src_vid.shape[1]:]
        vid_att_mem = torch.einsum("bl,bld->bld", vid_att_weight, vid_mem)

        _memory = torch.zeros_like(memory)
        _memory[:, :src_vid.shape[1]] = vid_att_mem.clone()
        _memory[:, src_vid.shape[1]:] = txt_mem.clone()

        return hs, reference, _memory, vid_weights

    def _load_feats(self, feats, keyword='neg'):
        keys = feats.keys()
        if 'neg' == keyword:
            if 'neg_src_txt' in keys:
                return feats['neg_src_txt'], feats['neg_src_txt_mask']
            else:
                return None, None

        if 'match_labels' == keyword:
            return feats['match_labels']

        if 'shifted' == keyword:
            if 'shifted_vid' in keys:
                return feats['shifted_vid'], feats['shifted_vid_mask'], feats['shifted_match_labels']
            else:
                return None, None, None

    def calculate_probs(self, src_vid, src_vid_mask, pos_vid, src_txt, src_txt_mask, pos_txt,
                        reverse=False, returns=False):
        src = torch.cat([src_vid, src_txt], dim=1)
        mask = torch.cat([src_vid_mask, src_txt_mask], dim=1).bool()
        pos = torch.cat([pos_vid, pos_txt], dim=1)

        hs, references, memory = self.transformer(src, ~mask, self.query_embed.weight, pos)
        vid_mem = memory[:, :src_vid.shape[1]]
        frame_probs = self.FPM(vid_mem)
        frame_probs = frame_probs.sigmoid().squeeze(2)

        if reverse:
            frame_probs = torch.ones_like(frame_probs, device=frame_probs.device) - frame_probs

        if returns:
            return hs, references, memory, frame_probs

        return frame_probs


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, matcher, weight_dict, eos_coef, losses, temperature, span_loss_type, max_v_l,
                 additional_losses, saliency_margin=1, prob_thd=1):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            temperature: float, temperature for NCE loss
            span_loss_type: str, [l1, ce]
            max_v_l: int,
            additional_losses: list of additional losses to be applied. It will be set to None if there are no additional losses.
            saliency_margin: float
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.temperature = temperature
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        self.additional_losses = additional_losses
        self.saliency_margin = saliency_margin
        self.prob_thd = prob_thd

        # foreground and background classification
        self.foreground_label = 0
        self.background_label = 1
        self.eos_coef = eos_coef
        empty_weight = torch.ones(2)
        empty_weight[-1] = self.eos_coef  # lower weight for background (index 1, foreground index 0)
        self.register_buffer('empty_weight', empty_weight)

    def loss_spans(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "spans" containing a tensor of dim [nb_tgt_spans, 2]
           The target spans are expected in format (center_x, w), normalized by the image size.
        """
        assert 'pred_spans' in outputs
        targets = targets["span_labels"]

        idx = self._get_src_permutation_idx(indices)
        src_spans = outputs['pred_spans'][idx]  # (#spans, max_v_l * 2)
        tgt_spans = torch.cat([t['spans'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # (#spans, 2)
        if self.span_loss_type == "l1":
            loss_span = F.l1_loss(src_spans, tgt_spans, reduction='none')
            loss_giou = 1 - torch.diag(generalized_temporal_iou(span_cxw_to_xx(src_spans), span_cxw_to_xx(tgt_spans)))
        else:  # ce
            n_spans = src_spans.shape[0]
            src_spans = src_spans.view(n_spans, 2, self.max_v_l).transpose(1, 2)
            loss_span = F.cross_entropy(src_spans, tgt_spans, reduction='none')
            loss_giou = loss_span.new_zeros([1])

        losses = {}
        losses['loss_span'] = loss_span.mean()
        losses['loss_giou'] = loss_giou.mean()
        return losses

    def loss_labels(self, outputs, targets, indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # TODO add foreground and background classifier.  use all non-matched as background.
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # (batch_size, #queries, #classes=2)
        # idx is a tuple of two 1D tensors (batch_idx, src_idx), of the same length == #objects in batch
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(src_logits.shape[:2], self.background_label,
                                    dtype=torch.int64, device=src_logits.device)  # (batch_size, #queries)
        target_classes[idx] = self.foreground_label
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction="none")
        losses = {'loss_label': loss_ce.mean()}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['loss_class'] = 100 - accuracy(src_logits[idx], self.foreground_label)[0]
        return losses

    def loss_saliency(self, outputs, targets, indices, log=True):
        """higher scores for positive clips"""
        if "saliency_pos_labels" not in targets:
            return {"loss_saliency": 0}
        saliency_scores = outputs["saliency_scores"]  # (N, L)
        pos_indices = targets["saliency_pos_labels"]  # (N, #pairs)
        neg_indices = targets["saliency_neg_labels"]  # (N, #pairs)
        num_pairs = pos_indices.shape[1]  # typically 2 or 4
        batch_indices = torch.arange(len(saliency_scores)).to(saliency_scores.device)
        pos_scores = torch.stack(
            [saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        neg_scores = torch.stack(
            [saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        loss_saliency = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum() \
                        / (len(pos_scores) * num_pairs) * 2  # * 2 to keep the loss the same scale

        return {"loss_saliency": loss_saliency}

    def loss_prob_guidance(self, outputs, targets, indices, log=True):
        if 'vid_logits' not in outputs:
            return {"loss_prob_guidance": 0}
        vid_logits = outputs['vid_logits']  # (bsz, 1, L_vid)

        mask = targets['logit_mask']
        match_labels = targets['match_labels']
        unmatch_labels = mask - match_labels
        foreground_probs = match_labels * vid_logits
        background_probs = unmatch_labels * vid_logits

        match_labels = torch.sum(match_labels, dim=-1)
        unmatch_labels = torch.sum(unmatch_labels, dim=-1)
        for i in range(len(unmatch_labels)):
            if unmatch_labels[i] == 0:
                unmatch_labels[i] = 1

        foreground_term = torch.sum(foreground_probs, dim=-1) / match_labels
        background_term = torch.sum(background_probs, dim=-1) / unmatch_labels
        prob_thd = torch.ones_like(foreground_term) * self.prob_thd
        loss_prob_guidance = prob_thd - foreground_term + background_term

        losses = {"loss_prob_guidance": loss_prob_guidance.mean(0)}
        return losses

    def loss_contrastive_align(self, outputs, targets, indices, log=True):
        """encourage higher scores between matched query span and input text"""
        if "proj_vid_mem" not in outputs or "proj_txt_mem" not in outputs:
            return {"loss_contrastive_align": 0}

        normalized_vid_embed = outputs["proj_vid_mem"]  # (bsz, d)
        normalized_txt_embed = outputs["proj_txt_mem"]  # (bsz, d)

        similarity = torch.matmul(normalized_vid_embed, normalized_txt_embed.t()) / self.temperature
        labels = torch.arange(len(similarity), device=similarity.device)
        losses = {"loss_contrastive_align": F.cross_entropy(similarity, labels, reduction='mean')}

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx  # two 1D tensors of the same length

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            "spans": self.loss_spans,
            "labels": self.loss_labels,
            "saliency": self.loss_saliency,
            "contrastive_align": self.loss_contrastive_align,
            "loss_prob_guidance": self.loss_prob_guidance,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k not in self.additional_losses}
        targets_without_aux = {k: v for k, v in targets.items() if k not in self.additional_losses}

        # Retrieve the matching between the outputs of the last layer and the targets
        # list(tuples), each tuple is (pred_span_indices, tgt_span_indices)
        indices = self.matcher(outputs_without_aux, targets_without_aux)
        # Compute all the requested losses

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets_without_aux, indices))

        aux_losses = {}

        for additional_loss in self.additional_losses:
            if additional_loss in outputs:
                additional_loss_dict = {}
                _outputs = outputs[additional_loss]
                _targets = targets[additional_loss]
                _indices = self.matcher(_outputs, _targets)
                for loss in self.losses:
                    additional_loss_dict.update(self.get_loss(loss, _outputs, _targets, _indices))

                aux_losses[additional_loss] = additional_loss_dict

        return losses, aux_losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class GELU(nn.Module):
    def forward(self, input_):
        output = gelu(input_)
        return output


def gelu(x):
    """ Original Implementation of the gelu activation function
        in Google Bert repo when initialy created.
        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi)
            * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


def mask_logits(target, mask):
    return target * mask + (1 - mask)


def build_model(args):
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/moment_detr/issues/108#issuecomment-650269223
    device = torch.device(args.device)

    transformer = build_transformer(args)
    position_embedding, txt_position_embedding = build_position_encoding(args)

    model = BMDETR(
        transformer,
        position_embedding,
        txt_position_embedding,
        txt_dim=args.t_feat_dim,
        vid_dim=args.v_feat_dim,
        num_queries=args.num_queries,
        input_dropout=args.input_dropout,
        span_loss_type=args.span_loss_type,
        t_feat_type=args.t_feat_type,
        use_txt_pos=args.use_txt_pos,
        n_input_proj=args.n_input_proj,
        contrastive_hdim=args.contrastive_hdim
    )

    matcher = build_matcher(args)
    weight_dict = {"loss_span": args.span_loss_coef,
                   "loss_giou": args.giou_loss_coef,
                   "loss_label": args.label_loss_coef,
                   "loss_class": args.class_loss_coef,
                   "loss_saliency": args.lw_saliency,
                   "loss_contrastive_align": args.lw_contrastive_loss_coef,
                   "loss_prob_guidance": args.lw_prob_loss_coef,
                   }

    excepted_losses = []
    for k, v in weight_dict.items():
        if v == 0:
            excepted_losses.append(k)

    for loss in excepted_losses:
        del weight_dict[loss]

    losses = ['spans', 'labels', 'saliency', 'contrastive_align', 'loss_prob_guidance']

    criterion = SetCriterion(
        matcher=matcher, weight_dict=weight_dict, losses=losses,
        eos_coef=args.eos_coef, temperature=args.temperature,
        span_loss_type=args.span_loss_type, max_v_l=args.max_v_l,
        additional_losses=args.additional_losses,
        saliency_margin=args.saliency_margin,
        prob_thd=args.prob_thd
    )

    criterion.to(device)
    return model, criterion
