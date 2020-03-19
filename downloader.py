import argparse
import os
import hashlib
import time
import subprocess

from helpers import logger


parser = argparse.ArgumentParser(description="Downloader")
parser.add_argument('--user', type=str, default=None)
parser.add_argument('--host', type=str, default=None)
parser.add_argument('--path', type=str, default=None)
args = parser.parse_args()


def download(args):
    # Create unique destination dir name
    hash_ = hashlib.sha1()
    hash_.update(str(time.time()).encode('utf-8'))
    dst = "downloads/logs_{}".format(hash_.hexdigest()[:20])
    os.makedirs(dst, exist_ok=False)
    # Download the logs with rsync
    src = "{}@{}:{}".format(args.user, args.host, args.path)
    logger.info("src: {}".format(src))
    response = subprocess.run(["rsync", "-hvPt", "--recursive", src, dst],
                              stderr=subprocess.PIPE)
    logger.info("Download done.")
    logger.info(response)


if __name__ == "__main__":
    # Download
    download(args)
