import numpy as np
from pathlib import Path
import pytest, torch

################################################################################################   
### Out of 10k given structs, 9k are within 1 stdev of mean. Any random choices will almost
### always lie in this near-equilibrium distribution. So we first prune to a smaller subset 
### before running random split to give OOD samples a more even footing with 
### near-equilibrium samples
################################################################################################




def test_frame_ID(tmp_path: Path):

    pass
    
