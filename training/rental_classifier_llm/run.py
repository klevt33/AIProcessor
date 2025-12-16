import asyncio
import json
import os
import sys
from dataclasses import dataclass
from enum import StrEnum

import pandas as pd
from langchain.prompts import PromptTemplate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from ai_engine import AIEngine
from config import Config
from lemmatizer import get_lemmatizer
from sdp import SDP
from utils import clean_text_for_llm


@dataclass
class Row:
    description: str
    is_rental: bool
    llm_pred: str | None
    uid: str


class Label(StrEnum):
    RENTAL = "RENTAL"
    NON_RENTAL = "NON RENTAL"


excel_path = r"C:\Users\BuchholzAnthony\Downloads"
input_file_name = "Rentals_PROD_cleaned.xlsx"
prompt_to_use = "prompt1"
max_rows = 1000

config = Config()
sdp = SDP(config=config)
ai_engine = AIEngine(config=config, sdp=sdp)

non_rental_examples = [
    "CURRENT LIGHTING SOLUTIONS LLC F32T8/SPX50/ECO2 T8 32W SPX50 ECOLUX",
    (
        '70U-HANGERW/HOOK_UBEND 70U-HANGERW/HOOK 167 SMOOTH ROD "U" SHAPED EG 0.167" X 36" NON THREADED ZP BLANK ROD HANGER WITH'
        " HOOK ENDS (PLEASE ORDER IN 10PC./BUNDLES)"
    ),
    (
        "88W NFC Linear Constant Current LED Driver, 240 2400 mA Programmable, Dim 0 10V w/ 1 100% range, 12V AUX, NFC= Near"
        " Field Communication. Field Programmable. 75858 NEWSTOCK MAR 2023"
    ),
    """A&A Transfer to provide the following services: Subcontract #: 1038 - Change Order #1
    Trip 1:
    Equipment: Push# 468404 sections of SWGR and misc. crates
    Special Equipment: (2) 15-Ton hydro gantries with 12' beam, generator, pallet of cribbing
    Scope Of Work: A&A Transfer to provide labor and equipment needed to deliver to site SWGR sections and associated crates. Equipment will be set on pads to be connected by others.
    Trip 2:
    Equipment: Push# 46564, 46563
    Special Equipment: (2) 15-Ton Hydro Gantries with 12' Beam, Generator, Pallet of Cribbing
    Scope Of Work: A&A Transfer to provide labor and equipment needed to deliver to site SWGR sections and associated Crates. Equipment will be set on pads to be connected by others. A&A will set in place Transformers as directed by the Customer.
    Trip 3:
    Equipment: Push# 468404 Sections of SWGR and Misc Crates
    Special Equipment: (2) 15-Ton Hydro Gantries with 12' Beam, Generator, Pallet of Cribbing
    Scope Of Work: A&A Transfer to provide labor and equipment needed to deliver to site SWGR sections and associated crates. Equipment will be set on pads to be connected by others.
    Invoice Total: $35,065.00
    Please contact Aron Kinney with any questions at 703-475-1551.""",
    """A&A Transfer to provide the following services:
    Purchase Order# 1141 IAD 95
    Trip 1:
    Equipment: Picking up from Crane Service
    Special Equipment: 4 Chains & Binders
    Scope Of Work: A&A Transfer to provide labor and equipment needed to pick up (2) 20K transformers. Equipment will come back to A&A to go out to site at a later date.
    Trip 2: Equipment: Push# 48401 Picking up 2 from Crane Service
    Special Equipment: 4 Blue 20's
    Scope Of Work: A&A Transfer to provide labor and equipment needed to deliver to site (3) transformers approx. 19K each. Crane will set up in the roadway working with a 50' radius. Once set in place customer is responsible for all hook ups.
    Invoice Total: $16,128.00
    Please contact Aron Kinney with any questions at 703-424-8430.""",
    "LOT: CO2 LOT CURRENT 1 - TYPE F1: TYPE F1 WAS UNDERBILLED",
]


prompt1 = """Input description:
[BEGIN]
{description}
[END]

Only respond with either `RENTAL` or `NON RENTAL` based on whether you think the input description is. The input description is a line item description entered on an invoice.

If the description contains something such as Rent, Rental, Lease, Leased, 'Material still on rent', or 'Rent complete' then it is likely a RENTAL.

If the description has something with 'Meter In' or 'Meter Out' or 'Hour Out' or 'Hour In' or 'Invoice From' or 'Invoice To' or '__/__/____ Through__/__/____' (or slightly alternative spelling) with information it is likely a RENTAL.

If it has a day, week or month or 4 weeks amount then it is RENTAL.

If it is missing any of the above or other indications that it is a RENTAL then assume it is a NON RENTAL.

Is the given input description a RENTAL or NON RENTAL?

"""

prompt2 = """Input description:
[BEGIN]
{description}
[END]

Only respond with either `RENTAL` or `NON RENTAL` based on whether you think the input description is. The input description is a line item description entered on an invoice.

If the description contains something such as Rent, Rental, Lease, Leased, 'Material still on rent', or 'Rent complete' then it is likely a RENTAL. For example:
'AT41M Rental of the Following Mobile Utility Equipment:
Unit #: 037-71147349 Mounted On : 2020 FORD F550 1FDUF5HT5LDA01830
Rental Period: 01/23/25 - 02/19/25'
is a RENTAL

If the description has something with 'Meter In:' or 'Meter Out:' or 'Hr Out:' or 'Hr In:' or 'Invoice From:' or 'Invoice To:' or '__/__/____ Through__/__/____' with information it is likely a RENTAL. For example:
'50-8FGU25 LP_TOYOTA 50-8FGU25 LP 11731340 FORKLIFT WHSE 5000# LP ONLY
Serial: 508FGU25-19021 Meter out: 150.14 Meter in: . 00 Substituted for: FORKLIFT WHSE 5000# LOW PROFILE GAS/LP'
is a RENTAL

If it has a day, week or month or 4 weeks amount then it is RENTAL for example:
'5IN DIAM CORE BIT From 04/05/24 01:49PM To 04/17/24 02:39PM Rent Complete
DAY: 155.00 WK: 0.00 MO: 61.00 MIC OUT MIC IN RIDGE 12/14/16 GAL WET/DRY VAC From 04/05/24 01:49PM To 04/17/24 02:39PM Rent Complete
DAY: 13.00 WK: 37.00 MO: 73.00 UNIT IN ST. PAUL INCLUDES: (1)
(1) UTILITY NOZZLE, (1) WET NOZZLE, (1) CREVICE TOOL, (1) CAR NOZZLE,'
is a RENTAL

If it is missing any of the above or other indications that it is a RENTAL then assume it is a NON RENTAL. Here are example NON RENTAL items.

{non_rental_examples}

Is the given input description a RENTAL or NON RENTAL?

"""


def clean(row):
    return clean_text_for_llm(get_lemmatizer(), row["ITM_LDSC"])


async def call_llm(prompt, description, non_rental_examples):
    prompt_template = PromptTemplate(input_variables=["description"], template=prompt)
    chain = prompt_template | ai_engine.ai_stages.llms.aoai_gpt4o
    return await ai_engine.ai_stages.llms.get_llm_response(
        chain=chain, params={"description": description, "non_rental_examples": non_rental_examples}
    )


async def get_llm_guess(description, prompt) -> str:
    original_string = (await call_llm(prompt, description, non_rental_examples)).content
    ret_val = "".join(char for char in original_string if char.isupper() or char == " ")
    if ret_val not in (Label.RENTAL, Label.NON_RENTAL):
        raise Exception("UNKNOWN GUESS")
    return ret_val


ids_to_test = [
    798817,
    769529,
    958380,
    600847,
    578391,
    546517,
    1010268,
    787492,
    722340,
    712582,
    1081929,
    32429,
    11964,
    780476,
    199745,
    931,
    68309,
    927720,
    736741,
    949930,
    636311,
    79382,
    29684,
    111836,
    954359,
    417542,
    1051997,
    482715,
    944603,
    25092,
    725603,
    936183,
    1054736,
    1054735,
    1087576,
    459103,
    580463,
    580464,
    388248,
    388252,
    436660,
    388250,
    436659,
    382393,
    402654,
    998104,
    952948,
    109116,
    85210,
    351255,
    477552,
    252764,
    85211,
    118386,
    730327,
    518605,
    82278,
    82285,
    82284,
    82287,
    82286,
    82288,
    82283,
    82277,
    82280,
    82282,
    14233,
    699614,
    1098846,
    755393,
    1067706,
    532220,
    517344,
    1043656,
    85208,
    910177,
    741502,
    795357,
    1091107,
    29267,
    152574,
    51973,
    154760,
    90228,
    36527,
    51972,
    10698,
    540804,
    666743,
    192054,
    40100,
    1017681,
    950912,
    1054617,
    295421,
    371327,
    287631,
    589543,
    499291,
    589090,
    82281,
    606557,
    767016,
    643870,
    609442,
    439910,
    533095,
    119729,
    28921,
    716562,
    716561,
    716560,
    361659,
    249708,
    1067092,
    1009037,
    82279,
    156810,
    116271,
    123576,
    115890,
    4654,
    109331,
    659949,
    666757,
    693103,
    540832,
    601700,
    62565,
    253182,
    249859,
    292708,
    271609,
    341972,
    51408,
    62101,
    933257,
    202710,
    156169,
    15072,
    671926,
    1077900,
    458367,
    296527,
    600730,
    767015,
    364295,
    1108974,
    640651,
    640655,
    683377,
]


async def main():
    df = pd.read_excel(os.path.join(excel_path, input_file_name))
    df["Rental?"] = df["Rental?"].replace({"fee": "y", "freight": "y", "tax": "y"})
    # df["CLEAN_DESCRIPTION"] = df.apply(clean, axis=1)
    df = df[df["IVCE_DTL_UID"].isin(ids_to_test)]
    # df.to_excel("fn_cases.xlsx")
    # df = df[df["Rental?"].isin(("y", "n"))]
    # df = df.drop_duplicates(subset=["CLEAN_DESCRIPTION"], keep="first")

    # tasks = [get_llm_guess(row["CLEAN_DESCRIPTION"]) for _, row in df.iterrows()]
    # results = await asyncio.gather(*tasks)
    # df["PROMPT1_GUESS"] = results
    # df.to_excel(r"C:\Users\BuchholzAnthony\Downloads\Rentals_Prod_cleaned.xlsx", index=False)
    rows = [
        Row(description=row["CLEAN_DESCRIPTION"], is_rental=row["Rental?"] == "y", llm_pred=None, uid=row["IVCE_DTL_UID"])
        for row in df.to_dict(orient="records")
    ]
    # random.shuffle(rows)
    # rows = rows[:max_rows]
    tp, tn, fp, fn = 0, 0, 0, 0
    fp_ids, fn_ids = [], []
    for row in rows:
        try:
            row.llm_pred = await get_llm_guess(row.description, globals()[prompt_to_use])
        except Exception:
            print(f"Failed on id: {row.uid}")
            continue
        if row.llm_pred == Label.RENTAL and row.is_rental:
            tp += 1
        elif row.llm_pred == Label.RENTAL and not row.is_rental:
            fp += 1
            fp_ids.append(row.uid)
        elif row.llm_pred == Label.NON_RENTAL and row.is_rental:
            fn += 1
            fn_ids.append(row.uid)
        elif row.llm_pred == Label.NON_RENTAL and not row.is_rental:
            tn += 1

    def flag_to_label(is_rental: bool) -> str:
        if is_rental:
            return Label.RENTAL.value
        if not is_rental:
            return Label.NON_RENTAL.value

    labels = [flag_to_label(row.is_rental) for row in rows]
    preds = [row.llm_pred for row in rows]
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, pos_label=Label.RENTAL.value)
    recall = recall_score(labels, preds, pos_label=Label.RENTAL.value)
    f1 = f1_score(labels, preds, pos_label=Label.RENTAL.value)

    results = {
        "prompt": globals()[prompt_to_use],
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        },
        "fp_ids": fp_ids,
        "fn_ids": fn_ids,
    }

    with open(f"evaluation_results_{prompt_to_use}-fn.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")


asyncio.run(main())
