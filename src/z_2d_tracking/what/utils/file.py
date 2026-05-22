import os.path
import hashlib

import progressbar
import urllib.request

class __MyProgressBar__():
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar=progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()

def download_file(file_name, path, url, hash):
    print('Downloading', os.path.join(path, file_name))

    if not os.path.exists(path):
        os.makedirs(path)

    try:
        try:
            urllib.request.urlretrieve(url, os.path.join(path, file_name), __MyProgressBar__())
        except Exception as e:
            print(e)
    except (Exception, KeyboardInterrupt):
        if os.path.exists(os.path.join(path, file_name)):
            os.remove(os.path.join(path, file_name))
        raise

    # Validate download if succeeded and user provided an expected hash
    sha256_hash = hashlib.sha256()
    with open(os.path.join(path, file_name),"rb") as f:
        for byte_block in iter(lambda: f.read(4096),b""):
            sha256_hash.update(byte_block)
        if (sha256_hash.hexdigest() == hash):
            print("Model {} downloaded".format(os.path.join(path, file_name)))
        else:
            print('Incomplete or corrupted file detected.')

def get_file(file_name, path, url, hash):
    if os.path.isfile(os.path.join(path, file_name)):
        sha256_hash = hashlib.sha256()
        with open(os.path.join(path, file_name),"rb") as f:
            for byte_block in iter(lambda: f.read(4096),b""):
                sha256_hash.update(byte_block)
            if (sha256_hash.hexdigest() == hash):
                print("Model {} already exists".format(os.path.join(path, file_name)))
            else:
                redownload = input("File corrupted, Redownload? [Y/n]")
                if len(redownload) == 0:
                    redownload = 'y'
                else:
                    while not redownload in ['y', 'n']:
                        redownload = input("File corrupted, Redownload? [y or n]")

                if redownload == 'y':
                    download_file(file_name, path, url, hash)

    else:
        download_file(file_name, path, url, hash)
