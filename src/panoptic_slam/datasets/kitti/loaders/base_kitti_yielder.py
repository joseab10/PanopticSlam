from panoptic_slam.datasets.kitti.utils import KittiConfig


class BaseKittiYielder:

    def __init__(self, **kwargs):

        if "kitti_dir" in kwargs:
            KittiConfig.set_root_dir(kwargs['kitti_dir'])

        frame_step = kwargs.get("frame_step")
        self.frame_start = kwargs.get("start_frame")
        self.frame_end = kwargs.get("end_frame")
        self.frame_step = 1 if frame_step is None else frame_step

        self._time_offset_conf = kwargs.get("time_offset", None)
        self.time_offset = None

    @property
    def is_frame_delimited(self):
        return self.frame_start is not None and \
               self.frame_end is not None and \
               self.frame_step is not None

    def in_frame_range(self, frame):
        if not self.is_frame_delimited:
            return True

        if self.frame_start is not None:
            if frame < self.frame_start:
                return False

        if self.frame_end is not None:
            if frame > self.frame_end:
                return False

        if self.frame_step is None:
            return True

        if self.frame_step == 1:
            return True

        return (frame - self.frame_start) % self.frame_step == 0

    def frame_range(self, max_frame=None):
        if max_frame is None:
            max_frame = 100000

        i = self.frame_start if self.frame_start is not None else 0
        step = self.frame_step if self.frame_step is not None else 1

        if self.frame_end is not None:
            end = self.frame_end
        else:
            if isinstance(max_frame, int):
                end = max_frame
            else:
                try:
                    end = (len(max_frame) * step) + i
                except Exception:
                    raise TypeError("Invalid type for the max_frame parameter ({})."
                                    "Only int and iterables supported.".format(type(max_frame), max_frame))

        while i <= end:
            yield i
            i += step
