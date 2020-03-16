import argparse
import os
import subprocess

from helpers import logger


parser = argparse.ArgumentParser(description="Log downloader")
parser.add_argument('--user', type=str, default=None)
parser.add_argument('--host', type=str, default=None)
parser.add_argument('--path', type=str, default=None)
args = parser.parse_args()


def download(args):
    dst = "downloads"
    os.makedirs(dst, exist_ok=True)
    src = "{}@{}:{}".format(args.user, args.host, args.path)
    logger.info("src: {}".format(src))
    stdout = subprocess.run(["rsync", "-hvPt", "--recursive", src, dst],
                            capture_output=True,
                            text=True).stdout
    logger.info("[INFO] Download done.")
    logger.info("[STDOUT]\n{}".format(stdout))


if __name__ == "__main__":
    # Download
    download(args)
