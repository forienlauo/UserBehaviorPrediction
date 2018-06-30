# coding=utf-8
import os
import sys

if __name__ == "__main__":
    # OPT(20180630) support option
    mode = sys.argv[1]
    wkdir = sys.argv[2]
    # TODO(20180630) check args

    # set base conf(conf is constructed by base conf)
    import base_conf

    base_conf.mode = base_conf.Mode[mode]
    base_conf.wkdir = wkdir
    if os.path.exists(wkdir):
        raise IOError("Work dir already exists: %s" % wkdir)
    os.makedirs(wkdir)
