"""
Adds training data from excel sheet to the SQL table
"""

import os
import sys

import pandas as pd

from utils import get_current_datetime_cst

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


def setup_env():
    """
    Add project root to sys.path
    """
    global APP_ROOT

    environment = "local"
    print("ENV Running: ", environment)

    global ENV
    ENV = environment

    if environment == "local":
        # Project root is three levels up.
        file_path = os.path.abspath(__file__)
        root = os.path.abspath(os.path.join(file_path, "..", "..", ".."))
        APP_ROOT = root
        if os.path.isdir(root) and root not in sys.path:
            sys.path.insert(0, root)
            print(f"[INFO] Added to sys.path: {root}")

    else:
        # ─── Locate the folder that contains config.py ───────────────────────────────
        candidates = [os.path.join(os.environ.get("HOME", "/home"), "site", "wwwroot"), "/app", os.getcwd()]

        for root in candidates:
            if os.path.isfile(os.path.join(root, "config.py")):
                sys.path.insert(0, root)
                APP_ROOT = root
                print(f"[INFO] Added to sys.path: {root}")
                break
        else:
            raise ImportError("Cannot find config.py in any of the known locations: " + ", ".join(candidates))


setup_env()

from config import Config
from sdp import SDP


def add_classifier_training_data(sdp: SDP, rows: list | None = None) -> pd.DataFrame:
    query = """
                INSERT INTO [AIML].[IVCE_XCTN_CLSFR_TRNL_DESC_REF]
                    ([IVCE_PRDT_LDSC]
                    ,[CLS_NM]
                    ,[TRNG_DAT_VRSN_NUM]
                    ,[REC_ACTV_IND]
                    ,[REC_CRTD_BY_ID]
                    ,[REC_CRTD_DTTM]
                    ,[REC_UPDD_BY_ID]
                    ,[REC_UPDD_DTTM]
                    ,[RNTL_IND])
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
    with sdp.sqlalchemy_engine.begin() as conn:
        raw_conn = conn.connection
        raw_cursor = raw_conn.cursor()
        raw_cursor.fast_executemany = True
        raw_cursor.executemany(query, rows)


def to_tuple(df_row) -> tuple:
    # cls_nm = f"RENTAL_{df_row["Classification"].upper()}" if df_row["Rental?"] == "y" else f"NON_RENTAL_{df_row["Classification"].upper()}"
    current_time = get_current_datetime_cst()
    rental = df_row["Rental?"]
    if df_row["Classification"] not in ["material", "unknown", "tax", "fee", "freight"]:
        rental = None
    return (
        df_row["ITM_LDSC"],
        df_row["Classification"].upper(),
        "NEW",
        "Y",
        "Anthony",
        current_time,
        "Anthony",
        current_time,
        rental,
    )


df = pd.read_excel(r"C:\Users\BuchholzAnthony\Downloads\merged_rentals.xlsx")
rows = [to_tuple(x) for _, x in df.iterrows()]

config = Config(app_root=APP_ROOT)  # Load application configuration
sdp = SDP(config=config)
add_classifier_training_data(sdp, rows)
print("Completed")
