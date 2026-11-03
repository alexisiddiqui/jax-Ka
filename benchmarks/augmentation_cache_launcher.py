"""Load the isolated pilot overlay against the frozen Isambard training source."""
import os
from pathlib import Path
import runpy
import sys

base=Path(os.environ['ANM_TRAINING_SOURCE'])
overlay=Path(os.environ['ANM_PILOT_OVERLAY'])
sys.path.insert(0,str(base))
import pkatrain
pkatrain.__path__.insert(0,str(overlay/'pkatrain'))
runpy.run_module('pkatrain.augmentation_cache_pilot',run_name='__main__')
