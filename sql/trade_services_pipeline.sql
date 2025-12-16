
-- Source Tables
--      [SDPDWH].[TPRT].[TRD_SVC_ITM_DESC_DTL]
-- 	    [SDPDWH].[TPRT].[TRD_SVC_ITM_MSTR]
-- 	    [SDPDWH].[TPRT].[TRD_SVC_MFR_MSTR]
-- 	    [SDPDWH].RPAO.MFR_DTL

-- Destination Tables
--     [SDPDWH].[AIML].[IVCE_XCTN_LLM_TRNL_MFR_REF]
--     [SDPDWH].[AIML].[IVCE_XCTN_LLM_TRNL_PRDT_REF]

-- Steps
-- This is a 2 step process.
-- 1. Run Query 1. First the data needs to be inserted in table [SDPDWH].[AIML].[IVCE_XCTN_LLM_TRNL_MFR_REF].
--      1. a  ) Check if record already exists in table [SDPDWH].[AIML].[IVCE_XCTN_LLM_TRNL_MFR_REF]. Record is unique based on distinct values of Manufacture Part Number, Manufacture Name, and UPC Code. ([MFR_PRT_NUM], [MFR_NM], [UPC_CD]). If it already exists then do not copy it over.
--      1. b ) Cross refernence the Trade Service data with [SDPDWH].RPAO.MFR_DTL table to filter the records needed by clean manufacturer names.
-- "    1. c ) UNSPSC_CD is not empty.
-- "
--      1. d ) Length of Manufacture Catalog Number  [MFR_CTLG_NUM] is greater than  4.

-- 2. Run Query 2. Once the data is inserted in the above table then insert the data in table [SDPDWH].[AIML].[IVCE_XCTN_LLM_TRNL_PRDT_REF].
--     2. a) Description check for already existing records? Can descriptions be altered or remain same ? If altered , do we want to keep the old description and add a new one to the list ) - add record and indicator as "N " so that AI Model will pick up the variation for training.
--     2. b ) Ingest records where ITM_DESC_TYP = 'PFMS'  since PFMS type has Manufacture Part number in the description which is required for AI usecase.
--     2. c ) Join Condition : UPC, MFR_ID, ITM_ID. Please refer to Query 2  below.

-- QUERY 1
INSERT INTO [SDPDWH].[AIML].[IVCE_XCTN_LLM_TRNL_MFR_REF](
[MFR_PRT_NUM],
[MFR_NM],
[UPC_CD],
[UNSPSC_CD],
REC_CRTD_BY_ID,
REC_CRTD_DTTM,
REC_UPDD_BY_ID,
REC_UPDD_DTTM,
[AKS_PRT_NUM]) (
 SELECT Replace(
           REPLACE(
           REPLACE(
           REPLACE(
           REPLACE(
           REPLACE(
           REPLACE(itm_mstr.MFR_CTLG_NUM, '-', ''),  -- Remove hyphen
                   '/', ''),  -- Remove slash
                   '=', ''),  -- Remove equal sign
                   '"', ''),  -- Remove double quote
                   '''', ''),  -- Remove single quote
                   '|', ''), -- Remove pipe
	       ' ','')  -- Remove space
 AS [MFR_PRT_NUM],
 mfr_dtl.CLN_MFR_NM AS [MFR_NM],
 itm_mstr.UPC_NUM AS [UPC_CD],
 itm_mstr.UNSPSC_CD AS [UNSPSC_CD],
 'AI_USER' AS REC_CRTD_BY_ID,
 GETDATE() AS REC_CRTD_DTTM,
 'AI_USER' AS REC_UPDD_BY_ID,
 GETDATE() AS REC_UPDD_DTTM,
 prd_mstr.AKS_PRT_NUM AS [AKS_PRT_NUM]
FROM [SDPDWH].[TPRT].[TRD_SVC_ITM_DESC_DTL] itm_det
JOIN [SDPDWH].[TPRT].[TRD_SVC_ITM_MSTR] itm_mstr
ON itm_det.ITM_ID = itm_mstr.ITM_MSTR_ID
JOIN [SDPDWH].[TPRT].[TRD_SVC_MFR_MSTR] mfr_mstr
ON mfr_mstr.MFR_ID = itm_mstr.MFR_ID
JOIN (select distinct CLN_MFR_NM,MFR_NM from [SDPDWH].RPAO.MFR_DTL) mfr_dtl on mfr_mstr.MFR_FULL_NM = mfr_dtl.MFR_NM
LEFT JOIN [SDPDWH].[AIML].[IVCE_XCTN_LLM_TRNL_MFR_REF] prd_mstr on itm_mstr.UPC_NUM = prd_mstr.UPC_CD
WHERE itm_det.ITM_DESC_TYP = 'PFMS'
AND itm_mstr.UNSPSC_CD IS NOT NULL
AND LEN( itm_mstr.MFR_CTLG_NUM) > 4
AND AKS_PRT_NUM is NULL
)

-- QUERY 2
INSERT INTO [SDPDWH].[AIML].[IVCE_XCTN_LLM_TRNL_PRDT_REF](
      [IVCE_XCTN_LLM_TRNL_MFR_REF_UNQ_ID]
      ,[SRC_NM]
      ,[SRC_REC_ID]
      ,[IVCE_PRDT_LDSC]
      ,[TRNG_DAT_VRSN_NUM]
      ,[REC_ACTV_IND]
      ,[REC_CRTD_BY_ID]
      ,[REC_CRTD_DTTM]
      ,[REC_UPDD_BY_ID]
      ,[REC_UPDD_DTTM])
SELECT MR.IVCE_XCTN_LLM_TRNL_MFR_REF_UID,
            'TRD_SRVC' AS SRC_NM ,
            itm_det.ITM_DESC_ID ,
            itm_det.ITM_SMRY_LDSC,
            '2025-1' AS TRAN_VER,
            'Y' AS REC_ACT,
            'AI_USER' AS REC_CRTD_BY_ID,
	GETDATE() AS REC_CRTD_DTTM,
	'AI_USER' AS REC_UPDD_BY_ID,
	GETDATE() AS REC_UPDD_DTTM
FROM [SDPDWH].[TPRT].[TRD_SVC_ITM_DESC_DTL] itm_det
JOIN [SDPDWH].[TPRT].[TRD_SVC_ITM_MSTR] itm_mstr
ON itm_det.ITM_ID = itm_mstr.ITM_MSTR_ID
JOIN [SDPDWH].[TPRT].[TRD_SVC_MFR_MSTR] mfr_mstr
ON mfr_mstr.MFR_ID = itm_mstr.MFR_ID
JOIN [SDPDWH].[AIML].[IVCE_XCTN_LLM_TRNL_MFR_REF] MR
ON MR.UPC_CD = itm_mstr.UPC_NUM
WHERE itm_det.ITM_DESC_TYP = 'PFMS'
AND itm_mstr.UNSPSC_CD IS NOT NULL
AND LEN( itm_mstr.MFR_CTLG_NUM) > 4
