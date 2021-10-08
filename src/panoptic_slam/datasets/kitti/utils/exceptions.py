"""Provides exception classes for dealing with the KITTI Dataset"""


class KittiError(Exception):
    """Base class for KITTI related exceptions"""
    pass


class KittiGTError(KittiError):
    """Class for errors when KITTI has no Ground Truth"""
    pass


class KittiTimeError(KittiError):
    """Class for errors when KITTI sequence does not have a given time stamp"""
    pass

