# ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
   packages=[
       'panoptic_slam',
       'panoptic_slam.kitti', 'panoptic_slam.kitti.converters', 'panoptic_slam.kitti.data_loaders',
       'panoptic_slam.ros', 'panoptic_slam.ros.utils'
   ],
   package_dir={
       '': 'src'
   },
)

setup(**setup_args)
