from multiprocessing import Pool
from os import path, remove

from tqdm import tqdm
from urllib2 import urlopen, URLError, HTTPError
import zipfile

from panoptic_slam.kitti.utils.exceptions import KittiError
import panoptic_slam.kitti.utils.config as kc
import panoptic_slam.kitti.utils.utils as ku


def download_zip_file(download_dir, url, **kwargs):
    """
    Downloads a zip file from a given URL to a given directory and (optionally) extracts it and deletes the original
    zip.

    :param download_dir: (path) Directory where the zip file will be downloaded to.
    :param url: (str) URL from where the zip file will be downloaded.
    :param kwargs:
        * filename: (str)(Default: None) Name of the local zip file. If None, the file will get its name from the URL.
        * chunk_size: (int)(Default: 256Kb (256*1024)) Size of the chunks in bytes in which the file will be downloaded.
        * extract: (bool)(Default: True) Extract the zip file if true.
        * extract_dir: (path)(Default: download_dir) Path where the extracted contents will be saved.
        * remove_zip: (bool)(Default: False) Delete the zip file after extracting its contents.

    :return: None
    """

    file_name = kwargs.get("filename", path.basename(url))
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


def download_zip_files(download_dir, urls, **kwargs):
    """
    Downloads a set of zip files from a given list of URLs to a given directory and (optionally) extracts them and
    deletes the original zips.

    :param download_dir: (path) Directory where the zip files will be downloaded to.
    :param urls: (list(str)) List of URLs from where the zip files will be downloaded.
    :param kwargs:
        * max_processes: (int) Number of parallel processes to run when downloading multiple files.
        * chunk_size: (int)(Default: 256Kb (256*1024)) Size of the chunks in bytes in which the file will be downloaded.
        * extract: (bool)(Default: True) Extract the zip file if true.
        * extract_dir: (path)(Default: download_dir) Path where the extracted contents will be saved.
        * remove_zip: (bool)(Default: False) Delete the zip file after extracting its contents.

    :return: None
    """

    max_processes = kwargs.get('max_processes', None)

    # Sequential download
    if max_processes == 1:
        for url in urls:
            download_zip_file(download_dir, url, **kwargs)
        return

    # Parallel downloads
    pool = Pool(processes=max_processes)
    procs = []
    for url in urls:
        proc = pool.apply_async(download_zip_file, args=(download_dir, url), kwds=kwargs)
        procs.append(proc)

    for proc in procs:
        proc.get()


def download_raw_data(download_dir, **kwargs):
    """
    Download the Raw KITTI dataset for EITHER a given sequence or a date/drive.

    :param download_dir: (path) Directory where the zip file will be downloaded to.
    :param kwargs:
        * seq: (int) KITTI dataset sequence.
        * date: (str) KITTI dataset drive date.
        * drive: (int) KITTI dataset drive number.
        * extension: (str)(Default: zip) Extension of the file to be downloaded from the URL.
        * dl_sync, dl_extract, dl_track, dl_calib: (bool)(Default: True, True, False, True) Download the sync, extract,
            tracklets and calibration data for the given date/drive.
        * max_processes: (int) Number of parallel processes to run when downloading multiple files.
        * chunk_size: (int)(Default: 256Kb (256*1024)) Size of the chunks in bytes in which the file will be downloaded.
        * extract: (bool)(Default: True) Extract the zip file if true.
        * extract_dir: (path)(Default: download_dir) Path where the extracted contents will be saved.
        * remove_zip: (bool)(Default: False) Delete the zip file after extracting its contents.

    :return: None
    """

    if not(("date" in kwargs and "drive" in kwargs) ^ ("seq" in kwargs)):
        raise RuntimeError("Either date and drive or a sequence must be passed to the download function.")

    date = kwargs.get("date", None)
    drive = kwargs.get("drive", None)

    if "seq" in kwargs:
        seq = kwargs.get("seq")
        print("Downloading SEQ{:02d}.".format(seq))
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

    print("Downloading data for Date {} / Drive {}.".format(date, drive))

    extension = kwargs.get("extension", "zip")

    default_subsets = {
        'sync':    {'download': True,  'file_suffix': "_sync"},
        'extract': {'download': True,  'file_suffix': "_extract"},
        'track':   {'download': False, 'file_suffix': "_tracklets"},
    }
    download_subsets = [
        k for k in default_subsets.keys()
        if kwargs.get("dl_" + k, default_subsets[k]['download'])
    ]
    download_urls = [
        kc.RAW_KITT_URL + drive_s + "/" + drive_s + default_subsets[k]['file_suffix'] + "." + extension
        for k in download_subsets
    ]

    calib = kwargs.get("dl_calib", True)
    if calib:
        download_subsets.append("calib")
        download_urls.append(kc.RAW_KITT_URL + date + "_calib" + "." + extension)

    print("Donwloading subsets {}.".format(",".join(download_subsets)))

    download_zip_files(download_dir, download_urls, **kwargs)


def download_odom_data(download_dir, **kwargs):
    """
    Download the KITTI Odometry dataset.

    :param download_dir: (path) Directory where the zip file will be downloaded to.
    :param kwargs:
        * dl_calib, dl_odo, dl_velo: (bool)(Default: True, True, True) Download the calibration, odometry and velodyne
            data respectively.
        * max_processes: (int) Number of parallel processes to run when downloading multiple files.
        * chunk_size: (int)(Default: 256Kb (256*1024)) Size of the chunks in bytes in which the file will be downloaded.
        * extract: (bool)(Default: True) Extract the zip file if true.
        * extract_dir: (path)(Default: download_dir) Path where the extracted contents will be saved.
        * remove_zip: (bool)(Default: False) Delete the zip file after extracting its contents.

    :return: None
    """

    default_subsets = {
        'calib': {'download': True},
        'odo':   {'download': True},
        'velo':  {'download': True},
    }
    download_subsets = [
        k for k in default_subsets.keys()
        if kwargs.get("dl_" + k, default_subsets[k]['download'])
    ]
    download_urls = [kc.KITTI_ODO_URLS[k] for k in download_subsets]

    print("Donwloading subsets {}.".format(",".join(download_subsets)))

    download_zip_files(download_dir, download_urls, **kwargs)


def download_data(download_dir, dataset, **kwargs):
    """
    Wrapper and selector of the specific download functions.

    :param download_dir: (path) Directory where the zip file will be downloaded to.
    :param dataset: (str) Kitti dataset to be downloaded
    :param kwargs: See specific download function.

    :return: None
    """

    dl_fun = {
        "raw": download_raw_data,
        "odom": download_odom_data
    }

    if dataset not in dl_fun.keys():
        raise ValueError("Invalid dataset {}. Supported values: {}.".format(dataset, dl_fun.keys()))

    return dl_fun[dataset](download_dir, **kwargs)
