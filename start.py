import sys

if __name__ == "__main__":
    mode = sys.argv[1]
    import run_mode

    run_mode.mode = run_mode.Mode[mode]
