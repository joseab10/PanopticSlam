from multiprocessing import Pool
from os import path, remove

from tqdm import tqdm
from urllib2 import urlopen, URLError, HTTPError
import zipfile

from panoptic_slam.kitti.utils.exceptions import KittiError
import panoptic_slam.kitti.utils.config as kc
import panoptic_slam.kitti.utils.utils as ku


def download_zip_data(download_dir, url, **kwargs):

    file_name = path.basename(url)
    file_path = path.join(download_dir, file_name)

    chunk_size = kwargs.get("chunk_size", 256 * 1024)

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

    if date not in kc.RAW_KITTI_DATASETS:
        raise KittiError("Error, no url configuration found for date {}.".format(date))

    date_set = kc.RAW_KITTI_DATASETS[date]

    if drive not in date_set:
        raise KittiError("No URL configuration found for date {} and drive {}.".format(date, drive))

    drive_s = date + "_drive_" + ku.format_raw_drive(drive)
    drive_set = date_set[drive]

    extension = kwargs.get("extension", "zip")

    default_subsets = {
        'sync':    {'download': True,  'file_suffix': "_sync"},
        'extract': {'download': True,  'file_suffix': "_extract"},
        'track':   {'download': False, 'file_suffix': "_tracklets"},
    }
    download_urls = [
        kc.RAW_KITT_URL + drive_s + "/" + drive_s + default_subsets[k]['file_suffix'] + "." + extension
        for k in drive_set
        if kwargs.get("dl_" + k, default_subsets[k]['download'])
    ]

    max_processes = kwargs.get('max_processes', None)
    pool = Pool(processes=max_processes)

    calib = kwargs.get("dl_calib", True)
    if calib:
        download_urls.append(kc.RAW_KITT_URL + date + "_calib" + "." + extension)

    procs = []
    for url in download_urls:
        proc = pool.apply_async(download_zip_data, args=(download_dir, url), kwds=kwargs)
        procs.append(proc)

    for proc in procs:
        proc.get()


def download_odom_dataset(**kwargs):
    # TODO
    pass
