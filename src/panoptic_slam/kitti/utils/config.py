import datetime as dt

# Mapping between KITTI Raw Drives and KITTI Odometry Sequences
KITTI_RAW_SEQ_MAPPING = {
    0:  {'date': "2011_10_03", 'drive': 27, 'start_frame':    0, 'end_frame': 4540},
    1:  {'date': "2011_10_03", 'drive': 42, 'start_frame':    0, 'end_frame': 1100},
    2:  {'date': "2011_10_03", 'drive': 34, 'start_frame':    0, 'end_frame': 4660},
    3:  {'date': "2011_09_26", 'drive': 67, 'start_frame':    0, 'end_frame':  800},
    4:  {'date': "2011_09_30", 'drive': 16, 'start_frame':    0, 'end_frame':  270},
    5:  {'date': "2011_09_30", 'drive': 18, 'start_frame':    0, 'end_frame': 2760},
    6:  {'date': "2011_09_30", 'drive': 20, 'start_frame':    0, 'end_frame': 1100},
    7:  {'date': "2011_09_30", 'drive': 27, 'start_frame':    0, 'end_frame': 1100},
    8:  {'date': "2011_09_30", 'drive': 28, 'start_frame': 1100, 'end_frame': 5170},
    9:  {'date': "2011_09_30", 'drive': 33, 'start_frame':    0, 'end_frame': 1590},
    10: {'date': "2011_09_30", 'drive': 34, 'start_frame':    0, 'end_frame': 1200},
}

# Camera Configuration
RAW_KITTI_CAMERA_CFG = {
    'GRAY': {
        'L': 0,
        'R': 1,
    },
    'COLOR': {
        'L': 2,
        'R': 3,
    }
}

# KITTI Strings formatting and validation
KITTI_STR = {
    # KEY          Allowed Datatypes       Format String      Validation REGEX
    'seq':        {'types': [int, str],         'fmt': "{:02d}",   'valid': r"([01][0-9])|(2[01])"},
    'date':       {'types': [dt.datetime, str], 'fmt': "%Y_%m_%d", 'valid': r"(2011)_((09_((2[689])|(30)))|(10_03))"},
    'drive':      {'types': [int, str],         'fmt': "{:04d}",   'valid': r"0[0-9]{3}"},
    'raw camera': {'types': [int, str],         'fmt': "{:02d}",   'valid': r"0[0-3]"},
    'raw image':  {'types': [int, str],         'fmt': "{:010d}",  'valid': r"[0-9]{10}"},
    'raw oxts':   {'types': [int, str],         'fmt': "{:010d}",  'valid': r"[0-9]{10}"},
    'rawe velo':  {'types': [int, str],         'fmt': "{:010d}",  'valid': r"[0-9]{10}"},
    'raws velo':  {'types': [int, str],         'fmt': "{:010d}",  'valid': r"[0-9]{10}"},
    'odo labels': {'types': [int, str],         'fmt': "{:06d}",   'valid': r"[0-9]{6}"},
    'odo velo':   {'types': [int, str],         'fmt': "{:06d}",   'valid': r"[0-9]{6}"},
    # Directories
    'raw camera directory': {'types': [int, str], 'fmt': "image_{:02d}",  'valid': r"image_0[0-3]"},
    # Files
    'raw image file':  {'types': [int, str], 'fmt': "{:010d}.png",  'valid': r"[0-9]{10}\.png"},
    'raw oxts file':   {'types': [int, str], 'fmt': "{:010d}.txt",  'valid': r"[0-9]{10}\.txt"},
    'rawe velo file':  {'types': [int, str], 'fmt': "{:010d}.txt",  'valid': r"[0-9]{10}\.txt"},
    'raws velo file':  {'types': [int, str], 'fmt': "{:010d}.bin",  'valid': r"[0-9]{10}\.bin"},
    'odo labels file': {'types': [int, str], 'fmt': "{:06d}.label", 'valid': r"[0-9]{6}\.label"},
    'odo velo file':   {'types': [int, str], 'fmt': "{:06d}.bin",   'valid': r"[0-9]{6}\.bin"},
    'poses odo file':  {'types': [int, str], 'fmt': "{:02d}.txt",   'valid': r"(0[0-9])(10)\.txt"},
    'poses sem file':  {'types': [object],   'fmt': "poses.txt",    'valid': r"poses\.txt"},
}

# KITTI Raw Dataset download url
RAW_KITT_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/"


# Kitti Raw available datasets
RAW_KITTI_DATASETS = {
    '2011_09_26': {
        # City
        1:   ['extract', 'sync', 'tracklets'],
        2:   ['extract', 'sync', 'tracklets'],
        5:   ['extract', 'sync', 'tracklets'],
        9:   ['extract', 'sync', 'tracklets'],
        11:  ['extract', 'sync', 'tracklets'],
        13:  ['extract', 'sync', 'tracklets'],
        14:  ['extract', 'sync', 'tracklets'],
        17:  ['extract', 'sync', 'tracklets'],
        18:  ['extract', 'sync', 'tracklets'],
        48:  ['extract', 'sync', 'tracklets'],
        51:  ['extract', 'sync', 'tracklets'],
        56:  ['extract', 'sync', 'tracklets'],
        57:  ['extract', 'sync', 'tracklets'],
        59:  ['extract', 'sync', 'tracklets'],
        60:  ['extract', 'sync', 'tracklets'],
        84:  ['extract', 'sync', 'tracklets'],
        91:  ['extract', 'sync', 'tracklets'],
        93:  ['extract', 'sync', 'tracklets'],
        95:  ['extract', 'sync'],
        96:  ['extract', 'sync'],
        104: ['extract', 'sync'],
        106: ['extract', 'sync'],
        113: ['extract', 'sync'],
        117: ['extract', 'sync'],
        # Residential
        19:  ['extract', 'sync', 'track'],
        20:  ['extract', 'sync', 'track'],
        22:  ['extract', 'sync', 'track'],
        23:  ['extract', 'sync', 'track'],
        35:  ['extract', 'sync', 'track'],
        36:  ['extract', 'sync', 'track'],
        39:  ['extract', 'sync', 'track'],
        46:  ['extract', 'sync', 'track'],
        61:  ['extract', 'sync', 'track'],
        64:  ['extract', 'sync', 'track'],
        79:  ['extract', 'sync', 'track'],
        86:  ['extract', 'sync', 'track'],
        87:  ['extract', 'sync', 'track'],
        # Road
        15:  ['extract', 'sync', 'track'],
        27:  ['extract', 'sync', 'track'],
        28:  ['extract', 'sync', 'track'],
        29:  ['extract', 'sync', 'track'],
        32:  ['extract', 'sync', 'track'],
        52:  ['extract', 'sync', 'track'],
        70:  ['extract', 'sync', 'track'],
        101: ['extract', 'sync'],
        # Calibration
        119: ['extract'],
    },
    '2011_09_28': {
        # City
        2:   ['extract', 'sync'],
        # Campus
        16:  ['extract', 'sync'],
        21:  ['extract', 'sync'],
        34:  ['extract', 'sync'],
        35:  ['extract', 'sync'],
        37:  ['extract', 'sync'],
        38:  ['extract', 'sync'],
        39:  ['extract', 'sync'],
        43:  ['extract', 'sync'],
        45:  ['extract', 'sync'],
        47:  ['extract', 'sync'],
        # Person
        53:  ['extract', 'sync'],
        54:  ['extract', 'sync'],
        57:  ['extract', 'sync'],
        65:  ['extract', 'sync'],
        66:  ['extract', 'sync'],
        68:  ['extract', 'sync'],
        70:  ['extract', 'sync'],
        71:  ['extract', 'sync'],
        75:  ['extract', 'sync'],
        77:  ['extract', 'sync'],
        78:  ['extract', 'sync'],
        80:  ['extract', 'sync'],
        82:  ['extract', 'sync'],
        86:  ['extract', 'sync'],
        87:  ['extract', 'sync'],
        89:  ['extract', 'sync'],
        90:  ['extract', 'sync'],
        94:  ['extract', 'sync'],
        95:  ['extract', 'sync'],
        96:  ['extract', 'sync'],
        98:  ['extract', 'sync'],
        100: ['extract', 'sync'],
        102: ['extract', 'sync'],
        103: ['extract', 'sync'],
        104: ['extract', 'sync'],
        106: ['extract', 'sync'],
        108: ['extract', 'sync'],
        110: ['extract', 'sync'],
        113: ['extract', 'sync'],
        117: ['extract', 'sync'],
        119: ['extract', 'sync'],
        121: ['extract', 'sync'],
        122: ['extract', 'sync'],
        125: ['extract', 'sync'],
        126: ['extract', 'sync'],
        128: ['extract', 'sync'],
        132: ['extract', 'sync'],
        134: ['extract', 'sync'],
        135: ['extract', 'sync'],
        136: ['extract', 'sync'],
        138: ['extract', 'sync'],
        141: ['extract', 'sync'],
        143: ['extract', 'sync'],
        145: ['extract', 'sync'],
        146: ['extract', 'sync'],
        149: ['extract', 'sync'],
        153: ['extract', 'sync'],
        154: ['extract', 'sync'],
        155: ['extract', 'sync'],
        156: ['extract', 'sync'],
        160: ['extract', 'sync'],
        161: ['extract', 'sync'],
        162: ['extract', 'sync'],
        165: ['extract', 'sync'],
        166: ['extract', 'sync'],
        167: ['extract', 'sync'],
        168: ['extract', 'sync'],
        171: ['extract', 'sync'],
        174: ['extract', 'sync'],
        177: ['extract', 'sync'],
        179: ['extract', 'sync'],
        183: ['extract', 'sync'],
        184: ['extract', 'sync'],
        185: ['extract', 'sync'],
        186: ['extract', 'sync'],
        187: ['extract', 'sync'],
        191: ['extract', 'sync'],
        192: ['extract', 'sync'],
        195: ['extract', 'sync'],
        198: ['extract', 'sync'],
        199: ['extract', 'sync'],
        201: ['extract', 'sync'],
        204: ['extract', 'sync'],
        205: ['extract', 'sync'],
        208: ['extract', 'sync'],
        209: ['extract', 'sync'],
        214: ['extract', 'sync'],
        216: ['extract', 'sync'],
        220: ['extract', 'sync'],
        222: ['extract', 'sync'],
        # Calibration
        225: ['extract'],
    },
    '2011_09_29': {
        # City
        26:  ['extract', 'sync'],
        71:  ['extract', 'sync'],
        # Road
        4:   ['extract', 'sync'],
        # Calibration
        108: ['extract'],
    },
    '2011_09_30': {
        # Residential
        18:  ['extract', 'sync'],
        20:  ['extract', 'sync'],
        27:  ['extract', 'sync'],
        28:  ['extract', 'sync'],
        33:  ['extract', 'sync'],
        34:  ['extract', 'sync'],
        # Road
        16:  ['extract', 'sync'],
        # Calibration
        72:  ['extract'],
    },
    '2011_10_03': {
        # Residential
        27:  ['extract', 'sync'],
        34:  ['extract', 'sync'],
        # Road
        42:  ['extract', 'sync'],
        47:  ['extract', 'sync'],
        # Calibration
        58:  ['extract'],
    },
}

SEMANTIC_LABELS = {
    0: "unlabeled",
    1: "outlier",

    10: "car",
    11: "bicycle",
    13: "bus",
    15: "motorcycle",
    16: "on-rails",
    18: "truck",
    20: "other vehicle",

    30: "person",
    31: "bicyclist",
    32: "motorcyclist",

    40: "road",
    44: "parking",
    48: "sidewalk",
    49: "other ground",

    50: "building",
    51: "fence",
    52: "other structure",
    60: "lane marking",

    70: "vegetation",
    71: "trunk",
    72: "terrain",

    80: "pole",
    81: "traffic sign",
    99: "other object",

    252: "moving car",
    253: "moving bicyclist",
    254: "moving person",
    255: "moving motorcyclist",
    256: "moving on-rails",
    257: "moving bus",
    258: "moving truck",
    259: "moving other vehicle"
}
