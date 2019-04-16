# ADD from project_setup import * to any file which is supposed to work inside
import os
from pathlib import Path
from settings import *


def setup():
    global project_path, source_path, test_path, output_path, data_path
    project_path = Path(project_path)
    source_path = Path(source_path)
    test_path = Path(test_path)
    output_path = Path(output_path)
    data_path = Path(data_path)

    os.chdir(project_path)
