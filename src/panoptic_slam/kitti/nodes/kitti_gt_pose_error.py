# Standard Libraries
from os import path

# Third Party Libraries
import numpy as np
import rospy
from std_srvs.srv import Trigger, TriggerResponse
from nav_msgs.msg import Path
from scipy.spatial.transform.rotation import Rotation

# Project Libraries
from panoptic_slam.kitti.data_analysers.kitti_pose_error import compute_pose_error, load_gt_poses, match_timestamps


class KittiGTPoseError:

    def __init__(self, kitti_dir, kitti_seq, output_dir, **kwargs):

        self._gt_poses_loader, self._gt_timestamps, self._gt_poses = load_gt_poses(kitti_dir, kitti_seq)
        self._output_dir = output_dir

        self._steps = 0  # Number of received trajectories.

        self._field_sep = kwargs.get("field_sep", " ")
        self._line_sep = kwargs.get("line_sep", "\n")

        self._file_desc = {
            # Each trajectory a list of: [step, frame, ts, x, y, z, r, p, y]
            "trajectories": {
                "filename": "trajectories.csv",
                "data": np.empty((0, 9)),
                "header": self._field_sep.join(["step", "frame",
                                                "ts_[s]", "x_[m]", "y_[m]", "z_[m]",
                                                "roll_[rad]", "pitch_[rad]", "yaw_[rad]"])
            },
            # Errors for each step in a trajectory as list of [step, frame, ts, ts_err, tra_err, rot_err]
            "errors": {
                "filename": "errors.csv",
                "data": np.empty((0, 6)),
                "header": self._field_sep.join(["step", "frame",
                                                "ts_[s]", "ts_err_[ns]",
                                                "tra_err_[m]", "rot_err_[rad]"])
            },
            # Relative transformations between GT and Estimated poses for each trajectory
            "transforms": {
                "filename": "transforms.csv",
                "data": np.empty((0, 13)),
                "header": self._field_sep.join(["step"] +
                                               ["T_{},{}".format(j, i) for i in range(3) for j in range(4)])
            }
        }

        self._path_subscriber = rospy.Subscriber("/lio_sam/mapping/path", Path, self._path_callback)
        self._save_service_provider = rospy.Service("/save_data", Trigger, self._save_callback)
        rospy.on_shutdown(self.save_output)

    def _path_callback(self, msg):

        est_timestamps = [p.header.stamp.to_nsec() for p in msg.poses]
        est_positions = np.array([[p.pose.position.x, p.pose.position.y, p.pose.position.z] for p in msg.poses])
        est_quaternions = [[p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z, p.pose.orientation.w]
                           for p in msg.poses]
        est_rotations = Rotation.from_quat(est_quaternions).as_euler("xyz")
        est_poses = np.concatenate((est_positions, est_rotations), axis=1)

        gt_pose_indexes, gt_ts_match_err = match_timestamps(self._gt_timestamps, est_timestamps)

        num_poses = len(gt_pose_indexes)
        tra_err, rot_err, tf, _ = compute_pose_error(self._gt_poses[gt_pose_indexes], est_poses)

        est_ts_s = np.array(est_timestamps, dtype=np.float64) * 1e-9

        self._steps += 1
        frame_col = np.ones(num_poses) * self._steps
        frame_col = frame_col.reshape((-1, 1))
        gt_pose_indexes = gt_pose_indexes.reshape((-1, 1))
        est_ts_s = est_ts_s.reshape((-1, 1))
        gt_ts_match_err = gt_ts_match_err.reshape((-1, 1))
        tra_err = tra_err.reshape((-1, 1))
        rot_err = rot_err.reshape((-1, 1))
        tf = tf[:3, :].reshape((1, 12))
        tf_step = np.ones((1, 1)) * self._steps

        step_trajectories = np.concatenate((frame_col, gt_pose_indexes,
                                            est_ts_s, est_poses), axis=1)
        step_errors = np.concatenate((frame_col, gt_pose_indexes,
                                      est_ts_s, gt_ts_match_err,
                                      tra_err, rot_err), axis=1)
        step_tf = np.concatenate((tf_step, tf), axis=1)

        self._file_desc['trajectories']['data'] = np.concatenate((step_trajectories,
                                                                  self._file_desc['trajectories']['data']), axis=0)
        self._file_desc['errors']['data'] = np.concatenate((step_errors,
                                                            self._file_desc['errors']['data']), axis=0)
        self._file_desc['transforms']['data'] = np.concatenate((step_tf,
                                                                self._file_desc['transforms']['data']), axis=0)

    def _save_callback(self, req):
        _ = req
        success, message = self.save_output()
        response = TriggerResponse()
        response.success = success
        response.message = message
        return response

    def _save_output_file(self, filename, array, header):
        try:
            file_path = path.join(self._output_dir, filename)
            np.savetxt(file_path, array, header=header, comments="#",
                       delimiter=self._field_sep, newline=self._line_sep)
        except Exception as e:
            return False, e

        return True, "Saved file {}.\n".format(file_path)

    def save_output(self):
        success = True
        message = ""

        for _, f in self._file_desc.items():
            tmp_suc, tmp_msg = self._save_output_file(f['filename'], f['data'], f['header'])
            success = success and bool(tmp_suc)
            message = message + str(tmp_msg)

        return success, message
