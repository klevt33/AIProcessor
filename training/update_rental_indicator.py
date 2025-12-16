import os
import sys
from typing import Optional

import pandas as pd

source_excel_file = r"C:\\Users\\VamsiMalneedi(Aspire\\Downloads\\Rentals_classification_cleaned.xlsx"
INVOICE_HEADER_COL = "IVCE_DTL_UID"
RENTAL_IND_COL = "rental?"
ALLOWED_VALS = ("y", "n")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


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
        root = os.path.abspath(os.path.join(file_path, "..", ".."))
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

df = pd.read_excel(source_excel_file)

df[RENTAL_IND_COL] = df[RENTAL_IND_COL].str.lower()
df = df[df[RENTAL_IND_COL].isin(ALLOWED_VALS)]
rows = df[[INVOICE_HEADER_COL, RENTAL_IND_COL]].apply(tuple, axis=1).tolist()
rows = [(row[0], row[1], "AI_USER") for row in rows]
print(f"Found {len(rows)} to update.")


def update_ivce_dtl_rntl_ind(rows: Optional[list] = None) -> pd.DataFrame:

    # from sqlalchemy import create_engine
    # sqlalchemy_engine = create_engine("mssql+pyodbc://svc_aks-ai-spendreport-prod:qc8pjfMb3ABpYqj3sojBuKdp@akssdp.35a92a58d879.database.windows.net,1433/SDPDWH?driver=ODBC+Driver+17+for+SQL+Server")

    from config import Config
    from sdp import SDP

    config = Config(app_root=APP_ROOT)  # Load application configuration
    sdp = SDP(config=config)
    sqlalchemy_engine = sdp.sqlalchemy_engine

    with sqlalchemy_engine.begin() as conn:
        # Step 1: Create temp table
        conn.exec_driver_sql(
            """
            CREATE TABLE #UpdateRentalRefTable (
                IVCE_DTL_UID INT,
                RNTL_IND VARCHAR(1),
                REC_UPDD_BY_ID [varchar](20) NULL,
                REC_UPDD_DTTM [datetime] NULL
            )
        """
        )
        print("created temp table")
        raw_conn = conn.connection
        raw_cursor = raw_conn.cursor()
        raw_cursor.fast_executemany = True
        raw_cursor.executemany(
            """
            INSERT INTO #UpdateRentalRefTable (IVCE_DTL_UID, RNTL_IND, REC_UPDD_BY_ID, REC_UPDD_DTTM)
            VALUES (?, ?, ?, GETDATE())
        """,
            rows,
        )
        print("Inserted data")
        # Step 3: Execute join-based update
        conn.exec_driver_sql(
            """
            UPDATE TGT
            SET TGT.RNTL_IND = TMP.RNTL_IND,
                TGT.REC_UPDD_BY_ID = TMP.REC_UPDD_BY_ID,
                TGT.REC_UPDD_DTTM = TMP.REC_UPDD_DTTM
            FROM RPAO.IVCE_DTL TGT
            JOIN #UpdateRentalRefTable TMP
                ON TGT.IVCE_DTL_UID = TMP.IVCE_DTL_UID
        """
        )
        print("Updated indicator")
        # Step 4: Drop temp table
        conn.exec_driver_sql("DROP TABLE #UpdateRentalRefTable")
        print("Dropped table")


print("Beginning SQL update")
update_ivce_dtl_rntl_ind(rows)
print("Finished SQL update")
