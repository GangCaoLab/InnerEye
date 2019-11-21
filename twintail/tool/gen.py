from os import path
import subprocess as subp

import logging
log = logging.getLogger(__file__)


HERE = path.dirname(path.abspath(__file__))
CONF_PATH = path.join(HERE, "../../conf.yaml")



def copy_file(src, dst):
    subp.check_call(['cp', src, dst])


def gen_file(src, dst, gen_func=copy_file):
    if path.exists(dst):
        log.info(f"{dst} already exists.")
    else:
        log.info(f"Generate {dst}.")
        gen_func(src, dst)


def gen(dst_dir="./"):
    """Generate files for pipeline"""
    fname = path.basename(CONF_PATH)
    dst = path.join(dst_dir, fname)
    gen_file(CONF_PATH, dst)
