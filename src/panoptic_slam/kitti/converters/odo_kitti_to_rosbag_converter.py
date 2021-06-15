from sensor_msgs.msg import PointField
from tqdm import tqdm

from panoptic_slam.kitti.converters.kitti_to_rosbag_converter import Kitti2RosBagConverter
from panoptic_slam.kitti.data_loaders import KittiOdomDataYielder
import panoptic_slam.kitti.utils.utils as ku
import panoptic_slam.ros.utils as ru


class OdoKitti2RosBagConverter(Kitti2RosBagConverter):

    def __init__(self, bag, kitti_dir, seq, **kwargs):

        kitti_loader = KittiOdomDataYielder(kitti_dir, seq, **kwargs)
        Kitti2RosBagConverter.__init__(self, bag, kitti_loader, **kwargs)

    def convert(self):
        if self._convert_velodyne:
            self.convert_velodyne()

    def convert_velodyne(self):
        print("Exporting Velodyne Data.")
        topic, frame_id = self.get_topic_and_frame("velo")
        fields = [
            PointField("x",          0, PointField.FLOAT32, 1),
            PointField('y',          4, PointField.FLOAT32, 1),
            PointField('z',          8, PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.FLOAT32, 1),
            PointField('ring',      16, PointField.UINT16,  1)
        ]

        for t, scan in tqdm(self.kitti_loader.yield_velodyne(), total=len(self.kitti_loader.get_timestamps("velo"))):
            stamp = ru.stamp_to_rospy(t)
            parsed_scan = ku.parse_velo(scan)
            header = ru.build_header_msg(stamp, frame_id)
            velo_msg = ru.build_pcl2_msg(header, fields, parsed_scan, is_dense=True)
            self.bag.write(topic, velo_msg, t=stamp)

