import numpy as np
from collections import OrderedDict
from librosa.effects import _signal_to_frame_nonsilent
from nussl import AudioSignal
from nussl.datasets import MUSDB18
from nussl.core.utils import musdb_track_to_audio_signals

root = "/exports/projects/computer-audition/data/musdb/raw/stems/"

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
      kwargs (dict): Arguments to ``nussl.datasets.MUSDB18``.
    """
    def __init__(self, segment_length=352_800, segment_hop=176_400,
                 top_db=40, segmentations_file=None, **kwargs):
        self.segment_length = segment_length
        self.segment_hop = segment_hop
        self.top_db = top_db
        super().__init__(**kwargs)

    def _segments_of_sources(self, sources):
        good_frames = None
        for name, signal in sources.items():
            good_frames_ = _signal_to_frame_nonsilent(
                            signal.audio_data,
                            top_db=self.top_db,
                            frame_length=self.segment_length,
                            hop_length=self.segment_hop,
                            ref=np.max)
            if good_frames is None:
                good_frames = good_frames_
            else:
                good_frames = good_frames & good_frames_
        segments = np.array([
                self.segment_hop * i for i, non_silent
                in enumerate(good_frames) if non_silent
            ])
        return segments

    def make_segmentations(self):
        segment_mappings = OrderedDict()
        segments = []
        num_intervals = 0
        for idx, track in enumerate(self.musdb):
            mix, sources = musdb_track_to_audio_signals(track)
            track_segments = self._segments_of_sources(sources)
            segments.append(track_segments)
            segment_mappings[num_intervals] = idx
            num_intervals += track_segments.shape[0]
        self.segment_mappings = segment_mappings
        self.segments = np.array(segments)
        self.num_intervals = num_intervals

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
        for key in keys:
            if item >= key:
                curr_idx += 1
            else: break
        return curr_idx

    def process_item(self, item):
        if item < 0 or item >= self.num_intervals:
            raise IndexError
        musdb_idx = self._get_musdb_idx(item)
        segment_idx = item - musdb_idx
        track = self.musdb[musdb_idx]
        start = self.segments[musdb_idx, segment_idx]
        # Taken from `nussl.core.utils.musdb_track_to_audio_signals`
        mixture = AudioSignal(audio_data_array=track.audio,
                              sample_rate=track.rate,
                              offset=start/track.rate,
                              duration=self.segment_length/track.rate)
        self._setup_audio_signal(mixture)
        mixture.path_to_input_file = track.name
        stems = track.stems
        sources = {}

        for k, v in sorted(track.sources.items(), key=lambda x: x[1].stem_id):
            sources[k] = AudioSignal(
                audio_data_array=stems[v.stem_id],
                sample_rate=track.rate,
                offset=start/track.rate,
                duration=self.segment_length/track.rate
            )
            sources[k].path_to_input_file = f'musdb/{track.name}_{k}.wav'
            self._setup_audio_signal(sources[k])

        output = {
            'mix': mixture,
            'sources': sources
        }

if __name__ == "__main__":
    musdb = MUSDB18Segmented(folder=root, is_wav=False, subsets=['test'])