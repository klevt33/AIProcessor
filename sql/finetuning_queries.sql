-- Query to fetch all the ACTV_IND data from the training tables.
SELECT
    d.[IVCE_PRDT_LDSC],
    i.[MFR_PRT_NUM],
    i.[MFR_NM],
    i.[UNSPSC_CD],
    d.[TRNG_DAT_VRSN_NUM]
FROM
    SDPDWH.AIML.IVCE_XCTN_LLM_TRNL_PRDT_REF d
        JOIN SDPDWH.AIML.IVCE_XCTN_LLM_TRNL_MFR_REF i
    ON d.[IVCE_XCTN_LLM_TRNL_MFR_REF_UID] = i.[IVCE_XCTN_LLM_TRNL_MFR_REF_UID]
WHERE d.[REC_ACTV_IND] = 'Y';


-- Query to fetch all the data that is not used for fine-tuning after the first model.
SELECT
    d.[IVCE_PRDT_LDSC],
    i.[MFR_PRT_NUM],
    i.[MFR_NM],
    i.[UNSPSC_CD],
    d.[TRNG_DAT_VRSN_NUM]
FROM
    SDPDWH.AIML.IVCE_XCTN_LLM_TRNL_PRDT_REF d
        JOIN SDPDWH.AIML.IVCE_XCTN_LLM_TRNL_MFR_REF i
    ON d.[IVCE_XCTN_LLM_TRNL_MFR_REF_UID] = i.[IVCE_XCTN_LLM_TRNL_MFR_REF_UID]
WHERE d.[REC_ACTV_IND] = 'Y' AND d.[TRNG_DAT_VRSN_NUM] != '2025-1';


-- Query to fetch the number of items for each training version.
SELECT
    d.[TRNG_DAT_VRSN_NUM],
    count(d.[TRNG_DAT_VRSN_NUM])
FROM SDPDWH.AIML.IVCE_XCTN_LLM_TRNL_PRDT_REF d
    JOIN SDPDWH.AIML.IVCE_XCTN_LLM_TRNL_MFR_REF i
        ON d.[IVCE_XCTN_LLM_TRNL_MFR_REF_UID] = i.[IVCE_XCTN_LLM_TRNL_MFR_REF_UID]
WHERE
    d.[REC_ACTV_IND] = 'Y'
GROUP BY d.[TRNG_DAT_VRSN_NUM]


-- Query to fetch all the on-trained items ([TRNG_DAT_VRSN_NUM] - 'NEW') from the training tables.
SELECT
    d.[IVCE_PRDT_LDSC],
    i.[MFR_PRT_NUM],
    i.[MFR_NM],
    i.[UNSPSC_CD],
    d.[TRNG_DAT_VRSN_NUM]
FROM
    SDPDWH.AIML.IVCE_XCTN_LLM_TRNL_PRDT_REF d
        JOIN SDPDWH.AIML.IVCE_XCTN_LLM_TRNL_MFR_REF i
    ON d.[IVCE_XCTN_LLM_TRNL_MFR_REF_UID] = i.[IVCE_XCTN_LLM_TRNL_MFR_REF_UID]
WHERE d.[REC_ACTV_IND] = 'Y' AND d.[TRNG_DAT_VRSN_NUM] = 'NEW';


-- Sample Update query used to insert some sample data with TRNG_DAT_VRSN_NUM as 'NEW' to test fine-tuning
INSERT INTO
    SDPDWH.AIML.IVCE_XCTN_LLM_TRNL_PRDT_REF
        (
        IVCE_XCTN_LLM_TRNL_MFR_REF_UID,
        SRC_NM,
        SRC_REC_ID,
        IVCE_PRDT_LDSC,
        TRNG_DAT_VRSN_NUM,
        REC_ACTV_IND,
        REC_CRTD_BY_ID,
        REC_CRTD_DTTM,
        REC_UPDD_BY_ID,
        REC_UPDD_DTTM)
VALUES
    (3583271,
    'TRD_SRVC',
    '602320186',
    'KSTL KT-RDLED-6B-BN-TRIM LTG TRM END NEW',
    'NEW',
    'Y',
    'AI_USER',
    '2025-06-02 16:31:18.987',
    'AI_USER',
    '2025-06-02 16:31:18.987');


-- Sample Update query to update the rows that are picked for fine tuning LLM.
UPDATE SDPDWH.AIML.IVCE_XCTN_LLM_TRNL_PRDT_REF
            SET
                TRNG_DAT_VRSN_NUM = 'FT-PROGRESS',
            WHERE IVCE_XCTN_LLM_TRNL_PRDT_REF_UID IN (?, ?, ?)

-- Display training data version and corresponding descriptions count
SELECT TRNG_DAT_VRSN_NUM, COUNT(*)
    FROM SDPDWH.AIML.IVCE_XCTN_LLM_TRNL_PRDT_REF_06162025
    GROUP BY TRNG_DAT_VRSN_NUM
