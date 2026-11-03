"""Run the isolated online-feature pilot against the frozen training source."""
import os
from pathlib import Path
import runpy
import sys

sys.path.insert(0,str(Path(os.environ['ANM_TRAINING_SOURCE'])))
import pkatrain
pkatrain.__path__.insert(0,str(Path(os.environ['ANM_PILOT_OVERLAY'])/'pkatrain'))
runpy.run_module('pkatrain.augmentation_online_features',run_name='__main__')
