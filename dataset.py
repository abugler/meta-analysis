import numpy as np
import torch
from collections import OrderedDict
from stempeg import read_stems
from librosa.effects import _signal_to_frame_nonsilent
from nussl import AudioSignal
from nussl.datasets import MUSDB18
from nussl.core.utils import musdb_track_to_audio_signals
from tqdm import tqdm

"""
Calling this file via command line will generate a segmentation file,
with predetermined locations where there exists noise.
"""

root = "/exports/projects/computer-audition/data/musdb/raw/stems/"
segmentation_file = "segmentation.npy"

class MUSDB18Segmented(MUSDB18):
    """
    Generates a list of segments in each track where each source has sound.

    A number of assumptions are made here:
     - Sample rate is 44.1kHz
     - Stereo

    Args:
      segment_length (int): Length in samples of each segment.
      segment_hop (int): Distance between each possible start of segment.
      top_db (int): The threshold (in decibels) below reference to consider
        as silence
      toy (int): Number of tracks to load. (Make this small for a "toy" dataset)
        if None, load all tracks.
      kwargs (dict): Arguments to ``nussl.datasets.MUSDB18``.
    """
    def __init__(self, segment_length=352_800, segment_hop=176_400,
                 top_db=8, load_seg=False, save_seg=False, toy=None, **kwargs):
        self.segment_length = segment_length
        self.segment_hop = segment_hop
        self.top_db = top_db
        self.toy = toy
        self.load_seg = load_seg
        self.save_seg = save_seg
        if save_seg and toy is not None:
            raise ValueError("Segmentations will only be saved if toy is None.")
        if load_seg:
            self.load_segmentations()
        super().__init__(**kwargs)
        if load_seg:
            self.musdb.tracks = sorted(self.musdb.tracks, key=lambda t: t.path)

    def _segments_of_sources(self, sources):
        good_frames = None
        for name, signal in sources.items():
            # Exclude the last frame
            good_frames_ = _signal_to_frame_nonsilent(
                            signal.audio_data,
                            top_db=self.top_db,
                            frame_length=self.segment_length,
                            hop_length=self.segment_hop,
                            ref=np.max)[:-1]
            if good_frames is None:
                good_frames = good_frames_
            else:
                good_frames &= good_frames_
        segments = np.array([
                self.segment_hop * i for i, non_silent
                in enumerate(good_frames) if non_silent
            ])
        return segments

    def make_segmentations(self):
        segment_mappings = OrderedDict()
        self.musdb.tracks = sorted(self.musdb.tracks, key=lambda t: t.path)
        segments = []
        num_intervals = 0
        for idx, track in tqdm(enumerate(self.musdb), desc="Loading Dataset"):
            if self.toy == idx:
                break
            mix, sources = musdb_track_to_audio_signals(track)
            track_segments = self._segments_of_sources(sources)
            segments.append(track_segments)
            segment_mappings[num_intervals] = idx
            num_intervals += track_segments.shape[0]

        self.segment_mappings = segment_mappings
        self.segments = np.array(segments, dtype=object)
        self.num_intervals = num_intervals
        if self.save_seg:
            data = np.array(
                [self.segment_mappings, self.segments, self.num_intervals],
                dtype=object
            )
            np.save(segmentation_file, data)

    def load_segmentations(self):
        self.segment_mappings, self.segments, self.num_intervals = \
            np.load(segmentation_file, allow_pickle=True)
        if self.toy is None:
            return
        # remove extraneous tracks, if only some tracks will be loaded via toy.
        self.segment_mappings = {k: v for i, (k, v) in
                                 enumerate(self.segment_mappings.items())
                                 if i < self.toy}
        self.segments = self.segments[0:self.toy]
        self.num_intervals = sum([len(a) for a in self.segments])

    def __len__(self):
        return self.num_intervals

    def get_items(self, _):
        if getattr(self, 'num_intervals', None) is None:
            self.make_segmentations()
        return list(range(self.num_intervals))

    def _get_musdb_idx(self, item):
        curr_idx = -1
        # This could be a binary search, but it is not worth it.
        keys = self.segment_mappings.keys()
        for key in keys: # keys are sorted
            if item >= key:
                curr_idx += 1
            else: break
        return key, curr_idx

    def process_item(self, item):
        if item < 0 or item >= self.num_intervals:
            raise IndexError
        first_frame, musdb_idx = self._get_musdb_idx(item)
        segment_idx = item - first_frame
        track = self.musdb[musdb_idx]
        start = self.segments[musdb_idx][segment_idx]
        # Taken from `nussl.core.utils.musdb_track_to_audio_signals`
        mixture, _ = read_stems(
            filename=track.path,
            stem_id=track.stem_id,
            start=start / 44_100,
            duration=self.segment_length / 44_100,
            info=track.info
        )
        mixture = mixture.T.astype(np.float32)

        stems, _ = read_stems(
            filename=track.path,
            start=start / 44_100,
            duration=self.segment_length / 44_100,
            info=track.info
        )
        sources = OrderedDict()

        for k, v in sorted(track.sources.items(), key=lambda x: x[1].stem_id):
            sources[k] = AudioSignal(
                audio_data_array=stems[v.stem_id].T
                .astype(np.float32),
                sample_rate=track.rate
            )
            sources[k].path_to_input_file = f'musdb/{track.name}_{k}.wav'
            self._setup_audio_signal(sources[k])

        output = {
            'mix': torch.tensor(mixture),
            'sources': sources
        }
        return output

if __name__ == "__main__":
    # This populates a segmentation file.
    musdb = MUSDB18Segmented(folder=root, is_wav=False, subsets=['test'], save_seg=True)
    # musdb = MUSDB18Segmented(folder=root, is_wav=False, load_seg=True, save_seg=False, subsets=['test'], toy=1)