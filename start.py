# coding=utf-8
import os
import sys


def main():
    # OPT(20180630) support option
    mode = sys.argv[1]
    logLevel = sys.argv[2]
    wkdir = sys.argv[3]
    # TODO(20180630) check args

    # set base conf(conf is constructed by base conf)
    import base_conf

    base_conf.mode = base_conf.Mode[mode.upper()]
    base_conf.logLevel = base_conf.LogLevel[logLevel.upper()]
    base_conf.wkdir = wkdir
    # REFACTOR(20180701) remove to check args
    if os.path.isdir(wkdir) and len(os.listdir(wkdir)) > 0:
        raise IOError("Non empty work dir: %s" % wkdir)
    if not os.path.isdir(wkdir):
        os.makedirs(wkdir)

    from src.driver import Driver
    Driver().run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
