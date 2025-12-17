import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from cdb import CDB
from config import Config

path = r"C:\\Users\\VamsiMalneedi(Aspire\\M_Vamsi\\Projects\\Spend_Report\\ai_logs.xlsx"

config = Config()
cdb = CDB(config=config)

columns = " c.id, c.request_details.start_time, c.process_output.end_time "
where = """ WHERE c._ts >= DateTimeDiff(
            "second",
            "1970-01-01T00:00:00Z",
            DateTimeAdd("day", -3, GetCurrentDateTime())
          ) """

order = """ ORDER BY c.request_details.start_time ASC """

docs = cdb.get_documents(
    container=cdb.ai_logs_container,
    column_filter=columns,
    where_condition=where,
    # top=2,
    order_condition=order,
)

df = pd.DataFrame(docs)
df.to_excel(path, index=False)
