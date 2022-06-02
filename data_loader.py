import h5py
import numpy as np
from os import PathLike
from pathlib import Path
from torch.utils.data import Dataset
from typing import Any, Dict, List, Union, Optional

class DSRDataLoader:
    def __init__(
        self,
        data_path: Union[str, PathLike],
        split: str,
        num_frames: int = 10,
        subseq_len: Optional[int] = None
    ):
        """
        Create DataLoader
        data_path: the path to the dataset(in the format provided by the authors)
        split: 'train' or 'test'
        num_frames: the number of frames for each data(10 for authors')
        subseq_len: the length of output sequence(equal to num_frames if not given)
        """
        if isinstance(data_path, str):
            data_path = Path(data_path)
        elif not isinstance(data_path, PathLike):
            raise TypeError("data_path is neither str or os.PathLike")

        self.data_path = data_path
        self.num_frames = num_frames
        if subseq_len is None:
            self.subseq_len = num_frames
            self.num_subseqs_in_seq = 1
        else:
            assert num_frames % subseq_len == 0 # subseq_len <= num_frames instead?
            self.subseq_len = subseq_len
            self.num_subseqs_in_seq = num_frames // subseq_len
        self.num_directions = 8
        self.volume_size = [128, 128, 48]
        with open(self.data_path / f'{split}.txt', 'r') as f:
            self.idx_list = [line.strip() for line in f.readlines()]

    def data_count(self) -> int:
        return len(self.idx_list) * self.num_subseqs_in_seq

    def load_data(self, index: int) -> List[Dict[str, Any]]:
        frame_list = []
        # Map index to part of sequence
        index_video = index // self.num_subseqs_in_seq
        frame_offset = index % self.num_subseqs_in_seq * self.subseq_len
        # convert the index to file index based on the .txt file
        data_idx = self.idx_list[index_video]
        for frame_idx in range(frame_offset, frame_offset+self.subseq_len):
            filename = f'{data_idx}_{frame_idx}.hdf5'
            frame_data = h5py.File(self.data_path/filename)
            frame_dict = {}

            # action, one-hot encoded with shape [8, W, H]
            direction, row, col = frame_data['action']
            frame_dict['action'] = np.zeros(shape=[self.num_directions, self.volume_size[0], self.volume_size[1]], dtype=np.float32)
            frame_dict['action'][direction, row, col] = 1

            # color_image, [W, H, 3]
            frame_dict['color_image'] = np.asarray(frame_data['color_image_small'], dtype=np.uint8)

            # color_heightmap(for visualiztion), [128, 128, 3]
            # NOTE: Unlike the authors' code, drawing arrow isn't done here
            frame_dict['color_heightmap'] = np.asarray(frame_data['color_heightmap'], dtype=np.uint8)

            # tsdf, [S1, S2, S3]
            frame_dict['tsdf'] = np.asarray(frame_data['tsdf'], dtype=np.float32)

            # mask_3d, [S1, S2, S3]
            frame_dict['mask_3d'] = np.asarray(frame_data['mask_3d'], dtype=np.int)

            # scene_flow_3d, [3, S1, S2, S3]
            scene_flow_3d = np.asarray(frame_data['scene_flow_3d'], dtype=np.float32).transpose([3, 0, 1, 2])
            frame_dict['scene_flow_3d'] = scene_flow_3d

            frame_list.append(frame_dict)

        return frame_list

# Helper class for PyTorch Dataset API, which supports automatic batching
class DSRDataset(Dataset):
    def __init__(
        self,
        data_path: Union[str, PathLike],
        split: str,
        num_frames: int = 10,
        subseq_len: Optional[int] = None
    ):
        super(DSRDataset, self).__init__()
        self.data_loader = DSRDataLoader(data_path, split, num_frames, subseq_len)

    def __len__(self) -> int:
        return self.data_loader.data_count()

    def __getitem__(self, index: int) -> List[Dict[str, Any]]:
        return self.data_loader.load_data(index)
