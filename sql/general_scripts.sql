--# FROM MDM TEAM
--  select top(100) * from
--  AKSPROF.DBO.[PRDT_PRT_NUM_RAW] where unspsc_rt_nm is not Null
--  SDPDWH.INRS.CTRT_PRCG_MSTR - map using UPC_CD

--  sDPDWH.[TPRT].[GO_UPC_PRDT_OTPT_DTL]
--  SDPDWH.[TPRT].[PRDT_CTLG_DTL]
--  SDPDWH.[TPRT].[TRD_SVC_PRDT_OTPT_DTL]
--  SDPDWH.INRS.CTRT_PRCG_MSTR

--   128k - golden table - MDMT.PRDT_MSTR
--   invc_tbl - invc description - akp

-- MFR_NM
-- MFR_PRT_NUM
-- UNSPSC_CD

----------------------------------------------------------------------------------------------
--# QUERY FROM VIDHYA FOR GOOD RECORDS FROM 128K RECORDS

SELECT A.AKS_LGCY_PRT_NUM, A.AKS_LGCY_UPC_CD, A.UNSPSC_CD, B.MFR_NM, B.MFR_PRT_NUM
FROM
  (SELECT *
  FROM SDPDMF.MDMT.PRDT_MSTR
  WHERE UNSPSC_CD IS NOT NULL AND
    AKS_LGCY_UPC_CD NOT IN (select AKS_LGCY_UPC_CD
    from aksprof.data.vprdt_mstr
    WHERE IS_AKS_PRT_NUM_GEN_IND='Y'))A
  LEFT JOIN
  SDPDMF.MDMT.PRDT_MFR_DTL B ON A. AKS_LGCY_PRT_NUM=B.AKS_LGCY_PRT_NUM
WHERE B.MFR_NM IS NOT NULL AND B.MFR_PRT_NUM IS NOT NULL;
----------------------------------------------------------------------------------------------

-- # TEMP SCRIPTS
--   select  top (1000) * from [SDPDWH].[IRSH].[PriceAgreements]
--   select  count(distinct PartNumber) from [SDPDWH].[IRSH].[PriceAgreements]
--   where PartNumber is not Null

--   SELECT COUNT(DISTINCT UPC) AS DuplicateUPCCount
-- FROM [SDPDWH].[IRSH].[PriceAgreements]
-- WHERE PartNumber IN (
--     SELECT PartNumber
--     FROM [SDPDWH].[IRSH].[PriceAgreements]
--     GROUP BY PartNumber
--     HAVING COUNT(DISTINCT UPC) > 1
-- );

----------------------------------------------------------------------------------------------
-- # SCRIPT TO MAP INVC TBL OVER TRNL_MFR TBL
SELECT
  t1.IVCE_DTL_UID,
  t1.IVCE_HDR_ID,
  t1.ITM_LDSC,
  t1.AKS_PRT_NUM,
  t2.IVCE_XCTN_LLM_TRNL_MFR_REF_UID,
  t2.MFR_PRT_NUM,
  t2.MFR_NM,
  t2.UPC_CD,
  t2.UNSPSC_CD
FROM [SDPDWH].[RPAO].[IVCE_DTL] AS t1
  INNER JOIN [SDPDWH].[AIML].[IVCE_XCTN_LLM_TRNL_MFR_REF] AS t2
  ON t1.AKS_PRT_NUM = t2.AKS_PRT_NUM
ORDER BY t1.IVCE_DTL_UID;
----------------------------------------------------------------------------------------------

SELECT
  t2.CLN_MFR_NM,
  --    t2.MFR_NM as IVCE_MFR_NM,
  t1.MFR_ID,
  t1.MFR_SHT_NM,
  t1.MFR_FULL_NM
FROM [SDPDWH].[TPRT].[TRD_SVC_MFR_MSTR] AS t1
  LEFT JOIN (SELECT distinct CLN_MFR_NM
  FROM [SDPDWH].[RPAO].[MFR_DTL]) AS t2
  ON t1.MFR_FULL_NM = t2.CLN_MFR_NM
WHERE CLN_MFR_NM is not null
----------------------------------------------------------------------------------------------
-- Fetch latest invoice details of a day for testing
SELECT *
FROM SDPDWH.RPAO.IVCE_DTL
WHERE CAST(REC_UPDD_DTTM AS DATE) = CAST(DATEADD(DAY, -3, GETDATE()) AS DATE)
order by REC_UPDD_DTTM desc
----------------------------------------------------------------------------------------------
SELECT COLUMN_NAME, DATA_TYPE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'YourTableName'

-----------------------------------------------------------------------------------------------

select *
from SDPDMF.MTMT.DAT_STD_ABRVN
where TRM_TXT like 'category'

select *
from SDPDMF.MTMT.DAT_STD_RPSTN

select *
from sdpdwh.[AIML].[IVCE_XCTN_LLM_TRNL_MFR_CONFL_REF]

-----------------------------------------------------------------------------------------------
-- Insert data into classifier mapping table
INSERT INTO SDPDWH.AIML.IVCE_XCTN_CLSFR_CTGY_DTL
  (CTGY_ID, CTGY_NM, PRNT_CTGY_ID, REC_CRTD_DTTM, REC_UPDD_DTTM, REC_CRTD_BY_ID, REC_UPDD_BY_ID)
VALUES
  ('1.2', 'GENERIC', '1', GETDATE(), GETDATE(), 'AI_USER', 'AI_USER')

-----------------------------------------------------------------------------------------------
-- Fetch the counts by invoice line status
SELECT
  IVCE_LNE_STAT,
  COUNT(*)
FROM
  sdpdwh.rpao.ivce_dtl
WHERE
  REC_CRTD_DTTM > '2025-07-29'
GROUP BY
  IVCE_LNE_STAT
ORDER BY
  IVCE_LNE_STAT;
-----------------------------------------------------------------------------------------------
-- Catalogue table details
SELECT TOP (5000)
  *
FROM SDPDWH.TPRT.PRDT_CTLG_DTL
WHERE VNDR_ID IS NOT NULL
order by rec_crtd_dttm

SELECT MFR_NM, COUNT(*) AS RecordCount
FROM SDPDWH.TPRT.PRDT_CTLG_DTL
WHERE VNDR_ID IS NULL
GROUP BY MFR_NM
ORDER BY RecordCount DESC;

SELECT VNDR_ID, COUNT(*) AS RecordCount
FROM SDPDWH.TPRT.PRDT_CTLG_DTL
GROUP BY VNDR_ID
ORDER BY RecordCount DESC;
-----------------------------------------------------------------------------------------------
-- Get counts by source name for MFR_REF
SELECT
  SRC_NM,
  COUNT(*) AS CNT
FROM
  SDPDWH.AIML.IVCE_XCTN_LLM_TRNL_MFR_REF
GROUP BY
    SRC_NM
-----------------------------------------------------------------------------------------------
ALTER TABLE [SDPDWH].[AIML].[IVCE_XCTN_CLSFR_TRNL_DESC_REF]
ADD RNTL_IND VARCHAR(1);
-----------------------------------------------------------------------------------------------
USE SDPDWH;
GO

-- Step 1: Rename column (schema.table.column, NOT fully qualified with db twice)
EXEC sp_rename 'AIML.IVCE_XCTN_CLSFR_TRNL_DESC_REF.TRNG_DAT_VRSN_NUM',
               'TRNG_DAT_VRSN_DESC',
               'COLUMN';

-- Step 2: Change datatype
ALTER TABLE SDPDWH.AIML.IVCE_XCTN_CLSFR_TRNL_DESC_REF
ALTER COLUMN TRNG_DAT_VRSN_DESC VARCHAR(MAX);
-----------------------------------------------------------------------------------------------
