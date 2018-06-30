import os
import sys

if __name__ == "__main__":
    mode = sys.argv[1]
    wkdir = sys.argv[2]
    import base_conf

    base_conf.mode = base_conf.Mode[mode]
    base_conf.wkdir = wkdir
    if os.path.exists(wkdir):
        raise IOError("Work dir already exists: %s" % wkdir)
    os.makedirs(wkdir)
