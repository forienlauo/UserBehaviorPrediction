import os

from run_mode import mode

# base

PROJECT_DIR = os.path.basename(__file__)

# path
RESOURCE_DIR = os.path.join(PROJECT_DIR, "resource", mode.name)
CDR_DIR = os.path.join(RESOURCE_DIR, "cdr")
PROPERTY_DIR = os.path.join(RESOURCE_DIR, "property")
