import json

import pandas as pd
from training_data import TrainingData

from constants import DatabaseObjects
from sdp import SDP
from utils import get_current_datetime_cst


async def update_classifier_training_data_version(sdp, training_data: list[TrainingData]):
    now = get_current_datetime_cst()
    rows = [(datum.invoice_desc_uid, json.dumps(datum.training_version), "AI_USER", now) for datum in training_data]

    try:
        with sdp.sqlalchemy_engine.begin() as conn:
            # Step 1: Create temp table
            conn.exec_driver_sql(
                """
                CREATE TABLE #UpdateClassifierRefTable (
                    [IVCE_XCTN_CLSFR_TRNL_DESC_REF_UID] [int] NOT NULL,
                    [TRNG_DAT_VRSN_NUM] [varchar](MAX) NOT NULL,
                    [REC_UPDD_BY_ID] [varchar](20) NULL,
                    [REC_UPDD_DTTM] [datetime] NULL
                )
            """
            )

            # Step 2: Insert using raw DBAPI connection (for fast_executemany)
            raw_conn = conn.connection
            raw_cursor = raw_conn.cursor()
            raw_cursor.fast_executemany = True
            raw_cursor.executemany(
                """
                INSERT INTO #UpdateClassifierRefTable (
                    IVCE_XCTN_CLSFR_TRNL_DESC_REF_UID,
                    TRNG_DAT_VRSN_NUM,
                    REC_UPDD_BY_ID,
                    REC_UPDD_DTTM
                )
                VALUES (?, ?, ?, ?)
            """,
                rows,
            )

            # Step 3: Execute join-based update
            conn.exec_driver_sql(
                f"""
                UPDATE TGT
                SET
                    TGT.TRNG_DAT_VRSN_NUM = TMP.TRNG_DAT_VRSN_NUM,
                    TGT.REC_UPDD_BY_ID = TMP.REC_UPDD_BY_ID,
                    TGT.REC_UPDD_DTTM = TMP.REC_UPDD_DTTM
                FROM {DatabaseObjects.TBL_IVCE_XCTN_CLSFR_TRNL_DESC_REF} TGT
                JOIN #UpdateClassifierRefTable TMP
                    ON TGT.IVCE_XCTN_CLSFR_TRNL_DESC_REF_UID = TMP.IVCE_XCTN_CLSFR_TRNL_DESC_REF_UID
            """
            )

            # Step 4: Drop temp table
            conn.exec_driver_sql("DROP TABLE #UpdateClassifierRefTable")

    except Exception as e:
        raise e


async def get_all_classifier_training_data(sdp: SDP, classes: list[str] | None = None) -> pd.DataFrame:
    """Retrieve ALL classifier training data

    Args:
        sdp (SDP): SDP object to submit SQL queries
        classes (list[str] | None, optional): optional list of classes to filter on

    Raises:
        e: any error returned from the SQL query fetching

    Returns:
        pd.DataFrame: resulting training data
    """
    query = f"""
                SELECT [IVCE_XCTN_CLSFR_TRNL_DESC_REF_UID]
                    ,[IVCE_PRDT_LDSC]
                    ,[CLS_NM]
                    ,[TRNG_DAT_VRSN_NUM]
                    ,[RNTL_IND]
                FROM {DatabaseObjects.TBL_IVCE_XCTN_CLSFR_TRNL_DESC_REF}
                WHERE [REC_ACTV_IND] = 'Y'
            """
    if classes:
        formatted_classes = "', '".join(classes)
        query += f"AND [CLS_NM] IN ('{formatted_classes}')"
    try:
        df = await sdp.fetch_data(query)

    except Exception as e:
        raise e
    return df
