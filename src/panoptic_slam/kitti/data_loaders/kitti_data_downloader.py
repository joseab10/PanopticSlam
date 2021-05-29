from multiprocessing import Pool
from os import path, remove

from tqdm import tqdm
from urllib2 import urlopen, URLError, HTTPError
import zipfile

from panoptic_slam.kitti.exceptions import KittiError
import panoptic_slam.kitti.utils as ku

_RAW_KITT_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/"
_RAW_KITTI_DATASETS = {
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
        16:  ['extract', 'sync'],
        42:  ['extract', 'sync'],
        47:  ['extract', 'sync'],
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
        # Calibration
        72:  ['extract'],
    },
    '2011_10_03': {
        # Residential
        27:  ['extract', 'sync'],
        34:  ['extract', 'sync'],
        # Calibration
        58:  ['extract'],
    },
}

_pbar_position = 0

def download_zip_data(download_dir, url, **kwargs):

    file_name = path.basename(url)
    file_path = path.join(download_dir, file_name)

    chunk_size = kwargs.get("chunk_size", 256 * 1024)

    pbar_position = kwargs.get('proc_num', 0)

    try:
        f = urlopen(url)

        print("Downloading file:\n{}\nto:\n{}".format(url, download_dir))

        total_bytes = int(f.info().getheader('Content-Length').strip())
        div_factor = 1e9 if total_bytes > 1e10 else (1e6 if total_bytes > 1e7 else 1e4)
        total_bytes = float(total_bytes) / div_factor
        downloaded_bytes = 0

        with open(file_path, "wb") as local_file:
            with tqdm(total=total_bytes, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', desc=file_name) as pbar:
                while True:
                    chunk = f.read(chunk_size)
                    dchunk_size = len(chunk)
                    downloaded_bytes += dchunk_size

                    if not chunk:
                        break

                    local_file.write(chunk)
                    pbar.update(float(dchunk_size)/div_factor)

        extract = kwargs.get('extract', True)
        extract_dir = kwargs.get('extract_dir', download_dir)
        remove_zip = kwargs.get('remove_zip', False)

        if extract:
            print("Extracting file: {}.".format(file_path))
            with zipfile.ZipFile(file_path) as f:
                f.extractall(extract_dir)

        if remove_zip:
            print("Deleting file: {}.".format(file_path))
            remove(file_path)

    except HTTPError as e:
        print("HTTP Error: ", e.code, url)

    except URLError as e:
        print("URL Error: ", e.reason, url)


def download_raw_data(download_dir, **kwargs):

    if not(("date" in kwargs and "drive" in kwargs) ^ ("seq" in kwargs)):
        raise RuntimeError("Either date and drive or a sequence must be passed to the download function.")

    date = kwargs.get("date", None)
    drive = kwargs.get("drive", None)

    if "seq" in kwargs:
        seq = kwargs.get("seq")
        date = ku.get_raw_seq_date(seq)
        drive = ku.get_raw_seq_drive(seq)

    if date is None or drive is None:
        raise RuntimeError("No date or drive could be determined.")

    if date not in _RAW_KITTI_DATASETS:
        raise KittiError("Error, no url configuration found for date {}.".format(date))

    date_set = _RAW_KITTI_DATASETS[date]

    if drive not in date_set:
        raise KittiError("No URL configuration found for date {} and drive {}.".format(date, drive))

    drive_s = date + "_drive_" + ku.format_drive(drive)
    drive_set = date_set[drive]

    extension = kwargs.get("extension", "zip")

    default_subsets = {
        'sync':    {'download': True,  'file_suffix': "_sync"},
        'extract': {'download': True,  'file_suffix': "_extract"},
        'track':   {'download': False, 'file_suffix': "_tracklets"},
    }
    download_urls = [
        _RAW_KITT_URL + drive_s + "/" + drive_s + default_subsets[k]['file_suffix'] + "." + extension
        for k in drive_set
        if kwargs.get(k, default_subsets[k]['download'])
    ]

    max_processes = kwargs.get('max_processes', None)
    pool = Pool(processes=max_processes)

    calib = kwargs.get("calib", True)
    if calib:
        download_urls.append(_RAW_KITT_URL + date + "_calib" + "." + extension)

    procs = []
    for i, url in enumerate(download_urls):
        kwargs["proc_num"] = i
        proc = pool.apply_async(download_zip_data, args=(download_dir, url), kwds=kwargs)
        procs.append(proc)

    for proc in procs:
        proc.get()


def download_odom_dataset(**kwargs):
    # TODO
    pass


if __name__ == "__main__":

    dir = "/home/jose/Documents/Master_Thesis/dat/Kitti/raw/"
    seq = 0

    download_raw_data(dir, seq=seq, remove_zip=True)
