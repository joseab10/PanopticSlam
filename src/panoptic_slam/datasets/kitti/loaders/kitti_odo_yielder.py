import numpy as np

from panoptic_slam.datasets.kitti.utils import KittiConfig
from base_kitti_yielder import BaseKittiYielder


class OdoKittiYielder(BaseKittiYielder):

    def __init__(self, seq, **kwargs):
        super(OdoKittiYielder, self).__init__(**kwargs)

        self.seq = seq

        self._timestamps = None

    def get_timestamps(self):
        frame_indexes = None
        if self._timestamps is None:
            if self.is_frame_delimited:
                frame_indexes = np.arange(self.frame_start, self.frame_end, self.frame_step)
            self._timestamps = KittiConfig.load_odo_timestamps(self.seq, frame_indexes)




if __name__ == "__main__":
    kitti = OdoKittiYielder(8)

    a = kitti.get_timestamps()
