import logging
import multiprocessing

from src.extractor import load_data
from src.fitter import fit_curves_parallel
from src.logging_config import setup_logger
from src.preprocess import preprocess_data

# Set up logger for the main module
logger = setup_logger("main")
LOG_LEVEL = logging.INFO
logger.setLevel(LOG_LEVEL)


FOLDER_PATH_RAW_CURVES = r"data/test_data/raw_data"
METADATA_PATH = r"data/test_data/metadata_test.csv"

NMETA = 5
NCYCLE_INIT = 5
CT_THRESH = 35
FLUO_THRESH = 0.1
FLUO_LOW_THRESH = 0.25
FLUO_UP_THRESH = 1000

INITIAL_PARAMS = (1, 0, 0.5, 20, 100)
PARAM_BOUNDS = ((0, -0.3, 0, -30, 0), (10, 0.3, 2, 50, 100))
N_JOBS = multiprocessing.cpu_count()

# 1: LOAD DATA
df_raw = load_data(
    FOLDER_PATH_RAW_CURVES,
    METADATA_PATH,
    log_level=LOG_LEVEL,
)

# 2: PREPROCESS DATA
df_proc = preprocess_data(
    df_raw,
    NMETA,
    NCYCLE_INIT,
    CT_THRESH,
    FLUO_THRESH,
    FLUO_LOW_THRESH,
    FLUO_UP_THRESH,
    log_level=LOG_LEVEL,
)

# 3: FITTING DATA
df_fitted = fit_curves_parallel(
    df_proc,
    NMETA,
    INITIAL_PARAMS,
    PARAM_BOUNDS,
    N_JOBS,
    log_level=LOG_LEVEL,
)
