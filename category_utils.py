import pandas as pd

from constants import DescriptionCategories
from logger import logger
from sql_utils import get_classifier_categories_data


class CateogryUtils:

    def __init__(self, sdp):
        self.sdp = sdp

    async def async_init(self):
        try:
            self.mapping_df = await get_classifier_categories_data(sdp=self.sdp)
        except Exception as e:
            logger.error(f"Error occurred in async_init(). {str(e)}", exc_info=True)

    def read_categories_data(self) -> pd.DataFrame:
        """
        Return a DataFrame with columns: CTGY_ID, CTGY_NM, PRNT_CTGY_ID.
        """
        return self.mapping_df[["CTGY_ID", "CTGY_NM", "PRNT_CTGY_ID"]]

    def get_root_categories(self) -> pd.DataFrame:
        """
        Return categories that have no parent (PRNT_CTGY_ID is null or not in CTGY_ID list).
        """
        roots = self.mapping_df[
            self.mapping_df["PRNT_CTGY_ID"].isna() | ~self.mapping_df["PRNT_CTGY_ID"].isin(self.mapping_df["CTGY_ID"])
        ]
        return roots

    def get_children(self, *, parent_id: str) -> pd.DataFrame:
        """
        Return direct children of a given category.
        """
        return self.mapping_df[self.mapping_df["PRNT_CTGY_ID"] == parent_id][["CTGY_ID", "CTGY_NM", "PRNT_CTGY_ID"]]

    def get_all_descendants(self, *, parent_id) -> list:
        """
        Recursively get all descendant category IDs of a given PRNT_CTGY_ID.
        """
        try:
            descendants = []
            children = self.mapping_df[self.mapping_df["PRNT_CTGY_ID"] == parent_id]["CTGY_ID"].tolist()
            for child in children:
                descendants.append(child)
                descendants.extend(self.get_all_descendants(parent_id=child))
        except Exception as e:
            logger.error(f"Error occurred in get_all_descendants(). {str(e)}", exc_info=True)
        return descendants

    def get_category_path(self, *, category_id) -> list:
        """
        Return the path from root to the given category ID as a list of category names.
        """
        try:
            row = self.mapping_df[self.mapping_df["CTGY_ID"] == category_id]
            if row.empty:
                return []

            path = [row.iloc[0]["CTGY_NM"]]
            parent_id = row.iloc[0]["PRNT_CTGY_ID"]

            while pd.notna(parent_id):
                parent_row = self.mapping_df[self.mapping_df["CTGY_ID"] == parent_id]
                if parent_row.empty:
                    break
                path.insert(0, parent_row.iloc[0]["CTGY_NM"])
                parent_id = parent_row.iloc[0]["PRNT_CTGY_ID"]

        except Exception as e:
            logger.error(f"Error occurred in get_category_path(). {str(e)}", exc_info=True)

        return path

    def get_category_id(
        self, *, category_name: str, parent_category_name: str | None = None, parent_category_id: str | None = None
    ) -> str | None:
        """
        Return category_id for a category with a given name and optional parent category name.

        Args:
            category_name (str): Name of the category.
            parent_category_name (str | None): Name of the parent category. If None, match root-level.

        Returns:
            str | None: category_id if found, else None.
        """
        try:
            category_name = category_name.upper()

            if parent_category_name is None and parent_category_id is None:
                row = self.mapping_df[(self.mapping_df["CTGY_NM"] == category_name) & (self.mapping_df["PRNT_CTGY_ID"].isna())]

            elif parent_category_id is not None:
                row = self.mapping_df[
                    (self.mapping_df["CTGY_NM"] == category_name) & (self.mapping_df["PRNT_CTGY_ID"] == parent_category_id)
                ]

            else:
                parent_category_name = parent_category_name.upper()
                parent_row = self.mapping_df[self.mapping_df["CTGY_NM"] == parent_category_name]
                if parent_row.empty:
                    return None
                parent_id = parent_row.iloc[0]["CTGY_ID"]
                row = self.mapping_df[
                    (self.mapping_df["CTGY_NM"] == category_name) & (self.mapping_df["PRNT_CTGY_ID"] == parent_id)
                ]

            if row.empty:
                return None

            category_id = row.iloc[0]["CTGY_ID"]

        except Exception as e:
            logger.error(f"Error occurred in get_category_id(). {str(e)}", exc_info=True)

        return category_id

    def convert_bad_parent_to_compatible_parent(self, additional_parent_id: str) -> str:
        """
        Converts a BAD category parent ID into a compatible hierarchical parent ID
        expected by the SDP table.

        The BAD category always has a base ID of "6". Its sub-categories (e.g.,
        6.1.1 for LOT, 6.1.2 for GENERIC) never appear directly, and BAD is not
        further classified as MATERIAL (so 6.1 is never used). Instead, certain
        BAD subcategories require an intermediate parent (e.g., LOT must have
        parent 6.1 instead of just 6).

        This function appends the required intermediate parent layer to ensure
        compatibility with the SDP table structure.

        Args:
            additional_parent_id (str): The additional hierarchical parent ID
                segment to append (e.g., "1" to form "6.1").

        Returns:
            str: A compatible parent ID in the format "<BAD_ID>.<additional_parent_id>".
        """
        bad_cat_id = self.get_category_id(category_name=DescriptionCategories.BAD)
        compatible_parent_id = ".".join([bad_cat_id, additional_parent_id])
        return compatible_parent_id
