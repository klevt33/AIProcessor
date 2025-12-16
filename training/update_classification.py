import os
import sys

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


def main():
    try:
        # from config import Config
        # from sdp import SDP

        # config = Config(app_root=APP_ROOT)  # Load application configuration
        # sdp = SDP(config=config)
        # sqlalchemy_engine = sdp.sqlalchemy_engine

        from sqlalchemy import create_engine

        sqlalchemy_engine = create_engine(
            "mssql+pyodbc://svc_aks-ai-spendreport-prod:qc8pjfMb3ABpYqj3sojBuKdp@akssdp.35a92a58d879.database.windows.net,1433/SDPDWH?driver=ODBC+Driver+17+for+SQL+Server"
        )

        rows = [(uid, category_id, "AI_USER") for uid in ref_uuid_list]

        with sqlalchemy_engine.begin() as conn:
            # Step 1: Create temp table
            conn.exec_driver_sql(
                """
                CREATE TABLE #UpdateRefTable (
                    [IVCE_DTL_UID] [int] NOT NULL,
                    [REC_UPDD_DTTM] [datetime] NULL,
                    [REC_UPDD_BY_ID] [varchar](20) NULL,
                    [CTGY_ID] [varchar](20) NULL
                )
            """
            )
            print("created temp table")

            # Step 2: Insert using raw DBAPI connection (for fast_executemany)
            raw_conn = conn.connection
            raw_cursor = raw_conn.cursor()
            raw_cursor.fast_executemany = True
            raw_cursor.executemany(
                """
                INSERT INTO #UpdateRefTable (
                    IVCE_DTL_UID,
                    CTGY_ID,
                    REC_UPDD_BY_ID,
                    REC_UPDD_DTTM
                )
                VALUES (?, ?, ?, GETDATE())
            """,
                rows,
            )
            print("Inserted data")

            # Step 3: Execute join-based update
            conn.exec_driver_sql(
                """
                UPDATE TGT
                SET
                    TGT.REC_UPDD_DTTM = TMP.REC_UPDD_DTTM,
                    TGT.REC_UPDD_BY_ID = TMP.REC_UPDD_BY_ID,
                    TGT.CTGY_ID = TMP.CTGY_ID
                FROM RPAO.IVCE_DTL TGT
                JOIN #UpdateRefTable TMP
                    ON TGT.IVCE_DTL_UID = TMP.IVCE_DTL_UID
            """
            )
            print("Updated indicator")

            # Step 4: Drop temp table
            conn.exec_driver_sql("DROP TABLE #UpdateRefTable")
            print("Dropped table")

    except Exception as e:
        raise e


if __name__ == "__main__":
    # Update these for any category data update
    ref_uuid_list = []
    category_id = "-"

    print("Beginning SQL update")
    main()
    print("Finished SQL update")
