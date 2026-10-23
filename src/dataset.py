from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Callable, Any
import hashlib
import numpy as np
import torch

from torch_brain.data.dataset import DatasetIndex
from torch_brain.data import Dataset
from torch_brain.data import collate as tbrain_collate
from temporaldata import Interval
import temporaldata as td
import brainsets.descriptions as bsd

from .utils import move_to_device
from .logger import get_cli_logger

SamplingIntervals = Dict[str, Interval]

logger = get_cli_logger()


def get_window(dataset, index):
    """Read a half-open window with stable floating-point boundary comparisons.

    Sampler additions can put a boundary a few ulps above a frame timestamp.
    A 1 ns left shift keeps the first frame and excludes the right endpoint;
    it does not crop or pad the resulting samples.
    """
    return dataset.get(index.recording_id, index.start - 1e-9, index.end - 1e-9)


class TwoViewDataset(Dataset):
    r"""A dataset class that extends TBDataset to support sampling more than 1 data slice.

    This dataset class adds two key features on top of TBDataset:
    1. A common_transform that can be applied to a batch of samples
    2. Support for both single index and iterable of indices for __getitem__

    The common_transform is particularly useful for operations that need to be applied
    to multiple samples together, like creating paired views of the data.
    """

    common_transform: Callable[[List[ViewData]], TwoViewData]

    def _get_one_item(self, index: DatasetIndex) -> ViewData:
        sample = get_window(self, index)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample  # type: ignore

    def __getitem__(self, index: DatasetIndex):
        index2 = self._get_second_view(index)
        if index2 is None:
            return None

        samples = [self._get_one_item(index), self._get_one_item(index2)]
        samples = self.common_transform(samples)
        return samples

    def get_sampling_intervals(self):
        """Keep block boundaries even when adjacent blocks share a domain."""
        domains = super().get_sampling_intervals()
        for rec, domain in domains.items():
            data = self._get_data_object(rec)
            if not hasattr(data, "blocks"):
                continue  # Original Bugeon files have only domain intervals.
            starts, ends = [], []
            for start, end in zip(data.blocks.start, data.blocks.end):
                part = domain & Interval(float(start), float(end))
                starts.extend(part.start)
                ends.extend(part.end)
            domains[rec] = Interval(np.array(starts), np.array(ends))
        return domains

    def set_second_view_props(
        self,
        window_length: float,
        max_time_diff: float,
        seed: int,
        pairing: Optional[dict] = None,
        deterministic: bool = False,
    ):
        self.pairing_mode = (pairing or {}).get("mode", "local")
        if self.pairing_mode not in {"recording", "local", "within_condition"}:
            raise ValueError(f"Unknown pairing mode: {self.pairing_mode}")
        self.window_length = window_length
        self.max_time_diff = (pairing or {}).get("max_distance", max_time_diff)
        self.sanitized_sampling_intervals = self.get_sampling_intervals()
        self.init_seed = seed
        self.deterministic_pairing = deterministic
        self.np_rng = np.random.default_rng(seed)
        self._sanitize_intervals(self.sanitized_sampling_intervals, window_length)

    def _get_second_view(self, v: DatasetIndex):
        # Validate view 1 as well as view 2: neither may cross a block boundary.
        samp_interval = self.sanitized_sampling_intervals[v.recording_id]
        inside = (samp_interval.start <= v.start + 1e-8) & (
            samp_interval.end >= v.end - 1e-8
        )
        if inside.sum() != 1:
            raise ValueError("View 1 must be inside exactly one valid block")
        if self.pairing_mode == "local" and self.max_time_diff == 0.0:
            return v
        if self.pairing_mode == "within_condition":
            blocks = self._get_data_object(v.recording_id).blocks
            first = (blocks.start <= v.start + 1e-8) & (blocks.end >= v.end - 1e-8)
            if first.sum() != 1:
                raise ValueError("View 1 must be inside exactly one block")
            same = blocks.condition == blocks.condition[first][0]
            allowed = np.array(
                [
                    np.any(
                        (blocks.start[same] <= a + 1e-8)
                        & (blocks.end[same] >= b - 1e-8)
                    )
                    for a, b in zip(samp_interval.start, samp_interval.end)
                ]
            )
            samp_interval = Interval(
                samp_interval.start[allowed], samp_interval.end[allowed]
            )
        if self.pairing_mode == "local" and self.max_time_diff is not None:
            samp_interval &= Interval(
                start=v.start - self.max_time_diff,
                end=v.end + self.max_time_diff,
            )

        # Validation pairs depend only on the seed and window, never worker/epoch state.
        rng = self.np_rng
        if self.deterministic_pairing:
            key = (
                f"{self.init_seed}:{v.recording_id}:"
                f"{float(v.start).hex()}:{float(v.end).hex()}"
            ).encode()
            seed = int.from_bytes(
                hashlib.blake2b(key, digest_size=16).digest(), "little"
            )
            rng = np.random.default_rng(seed)

        # 2. sample and check
        v2_interval = self._sample_window_from_interval(
            interval=samp_interval,
            window_length=self.window_length,
            rng=rng,
        )
        if v2_interval is None:
            return None

        return DatasetIndex(
            recording_id=v.recording_id,
            start=v2_interval.start[0],  # type: ignore
            end=v2_interval.end[0],  # type: ignore
        )

    @staticmethod
    def _sample_window_from_interval(
        interval: Interval,
        window_length: float,
        rng: np.random.Generator,
    ) -> Optional[Interval]:
        """
        Algo:
            An interval is made of smaller periods:
            [(intrvl.start[0], intrvl.end[0]), (intrvl.start[1], intrvl.end[1]), ...]
            So first we decide which period to sample from, and then sample from it.
        """

        # 1. Measure duration from which "start" time can be sampled from each period
        start_sample_durs: np.ndarray = interval.end - interval.start - window_length  # type: ignore
        if not (start_sample_durs > 1e-8).any():
            exact = np.flatnonzero(np.abs(start_sample_durs) <= 1e-8)
            if not len(exact):
                return None
            period_idx = rng.choice(exact)
            start = interval.start[period_idx]
            return Interval(start, start + window_length)

        # 2. Choose uniformly over all admissible window start times.
        start_sample_durs = start_sample_durs.clip(0)
        probs = start_sample_durs / sum(start_sample_durs)
        period_idx = rng.choice(len(interval), p=probs)
        period_start, period_end = interval.start[period_idx], interval.end[period_idx]  # type: ignore

        # 3. Sample from the chosen period
        start = rng.uniform(period_start, period_end - window_length)
        return Interval(start, start + window_length)

    @staticmethod
    def _sanitize_intervals(sampling_intervals: SamplingIntervals, min_length: float):
        """Remove interval periods that are shorter than min_length"""
        for rec_id, interval in sampling_intervals.items():
            valid_idx = (interval.end - interval.start) >= min_length  # type: ignore

            if not valid_idx.any():
                raise ValueError(
                    f"None of the periods in sampling interval for {rec_id} "
                    "are long enough."
                )

            sampling_intervals[rec_id] = Interval(
                interval.start[valid_idx], interval.end[valid_idx]  # type: ignore
            )


class ViewData:
    def __init__(
        self,
        data: dict[str, dict[str, Any]],
        unit_ids: np.ndarray,
        recording_ids: np.ndarray,
    ):
        self.data = data
        self.unit_ids = unit_ids
        self.recording_ids = recording_ids

        # Set programatically during collation
        self._unit_seqlen: Optional[List[int]] = None
        self._unit_boundaries: Optional[List] = None
        self._collated: bool = False
        self._batch_size: Optional[int] = None
        self.device: Optional[torch.device] = None

    @property
    def unit_seqlen(self) -> List[int]:
        assert self._unit_seqlen is not None
        return self._unit_seqlen

    def to(self, device: torch.device) -> ViewData:
        self.data = move_to_device(self.data, device)  # type: dict[str, dict[str, Any]]
        self.device = device
        return self

    @staticmethod
    def collate(data: List[ViewData]) -> ViewData:
        data_clean = [d for d in data if len(d.unit_ids) > 0 and d.unit_ids.ndim == 1]
        if len(data) > len(data_clean):
            print(f"Found some empty samples")
        assert len(data_clean) > 0, data

        collated_data = ViewData(
            data=tbrain_collate([d.data for d in data_clean]),  # type: ignore
            unit_ids=np.concatenate([d.unit_ids for d in data_clean]),
            recording_ids=np.concatenate([d.recording_ids for d in data_clean]),
        )
        collated_data._unit_seqlen = [len(d.unit_ids) for d in data_clean]
        collated_data._batch_size = len(data_clean)
        collated_data._collated = True
        return collated_data


class TwoViewData:
    def __init__(
        self,
        view1: ViewData,
        view2: ViewData,
        match_idx: torch.Tensor | List[torch.Tensor],
    ):
        self.view1 = view1
        self.view2 = view2
        self.match_idx = match_idx

        # Set during collation
        self._collated = False
        self._batch_size: Optional[int] = None
        self.device: Optional[torch.device] = None

    def to(self, device: torch.device) -> "TwoViewData":
        self.view1.to(device)
        self.view2.to(device)
        self.match_idx = move_to_device(
            self.match_idx, device
        )  # type: torch.Tensor | List[torch.Tensor]
        self.device = device
        return self

    @staticmethod
    def from_two_views(views: Tuple[ViewData, ViewData]) -> TwoViewData | None:
        # Used as combined_transform in the dataset
        # Used for data-pipeline efficiency purposes
        # Goal: To find the match indices at the dataloader worker so that
        # the main collate function doesn't have to
        view1, view2 = views
        is_clean = (
            isinstance(view1.unit_ids, np.ndarray)
            and isinstance(view2.unit_ids, np.ndarray)
            and (len(view1.unit_ids) > 10)
            and (len(view2.unit_ids) > 10)
        )
        if not is_clean:
            return None

        match = view1.unit_ids[:, None] == view2.unit_ids[None, :]
        return TwoViewData(
            view1=view1,
            view2=view2,
            match_idx=torch.from_numpy(np.vstack(np.where(match))),
        )

    @staticmethod
    def collate(data: List[TwoViewData | None]) -> TwoViewData:
        data_clean: List[TwoViewData] = [d for d in data if d is not None]
        # ^ need to be resillient to this
        # sometimes 2nd view is not found, and then the data is None
        assert len(data_clean) > 0, data

        collated_data = TwoViewData(
            view1=ViewData.collate([d.view1 for d in data_clean]),
            view2=ViewData.collate([d.view2 for d in data_clean]),
            match_idx=[d.match_idx for d in data_clean],  # type: ignore
        )
        collated_data._batch_size = len(data_clean)
        collated_data._collated = True
        return collated_data

    @property
    def collated(self) -> bool:
        return self._collated and (self._batch_size is not None)

    @property
    def batch_size(self) -> int:
        assert self._batch_size is not None
        return self._batch_size

    def get_match_idx_with_offset(self):
        assert self.collated, "Can only call this on collated TwoViewData"

        match_idx = [elem.clone() for elem in self.match_idx]
        enc1_running_idx, enc2_running_idx = 0, 0
        for b in range(self.batch_size):
            match_idx[b][0] += enc1_running_idx
            match_idx[b][1] += enc2_running_idx
            enc1_running_idx += self.view1.unit_seqlen[b]
            enc2_running_idx += self.view2.unit_seqlen[b]
        match_idx = torch.cat(match_idx, dim=-1)
        return match_idx


# -- A type template for data
class Spikes(td.IrregularTimeSeries):
    unit_index: np.ndarray
    timestamps: np.ndarray


class SpikeData(td.Data):
    spikes: Spikes
    session: bsd.SessionDescription
    brainset: bsd.BrainsetDescription
    units: td.ArrayDict


class CalciumTraces(td.IrregularTimeSeries):
    df_over_f: np.ndarray
    timestamps: np.ndarray


class CalciumData(td.Data):
    calcium_traces: CalciumTraces
    session: bsd.SessionDescription
    brainset: bsd.BrainsetDescription
    units: td.ArrayDict
