"""
## Overview
This module defines the `SDP` class, which provides methods for interacting with a database using both PyODBC and SQLAlchemy.
The class supports asynchronous query execution, data fetching, and batch processing.
It is designed to handle both read and write operations with robust error handling and logging.
"""

import asyncio
import functools
from typing import Iterator

import pandas as pd
import pyodbc
import sqlalchemy.exc
from pandas import DataFrame
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError

from constants import DatabaseObjects, Environments
from logger import logger


def retry_on_deadlock(max_retries: int = 3, backoff_factor: float = 1.0):
    """
    Decorator to retry a function if a SQL Server deadlock (error 1205) occurs.

    Args:
        max_retries (int): Maximum number of retry attempts.
        backoff_factor (float): Delay multiplier for exponential backoff.

    Usage:
        @retry_on_deadlock(max_retries=5, backoff_factor=2.0)
        async def update_data(...):
            ...
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except (sqlalchemy.exc.OperationalError, pyodbc.Error) as e:
                    err_msg = str(e).lower()
                    if "deadlock" in err_msg or "1205" in err_msg:
                        if attempt < max_retries - 1:
                            delay = backoff_factor * (2**attempt)
                            # Replace with logger.warning if you want structured logs
                            logger.debug(
                                f"[{func.__name__}()] Deadlock detected. Retrying in {delay:.1f}s... (attempt"
                                f" {attempt + 1}/{max_retries})"
                            )
                            await asyncio.sleep(delay)
                            continue
                    # Not a deadlock or retries exhausted → re-raise
                    raise

        return wrapper

    return decorator


class SDP:
    """
    SDP class for interacting with databases using PyODBC and SQLAlchemy.

    This class provides methods for executing queries, fetching data, and processing data in batches.
    It supports both synchronous and asynchronous operations, making it suitable for various use cases.

    Attributes:
        config: Configuration object containing database connection details.
        db_mapping: Dictionary mapping database tables to their respective database names.
    """

    db_mapping = {
        DatabaseObjects.TBL_INVC_DTL: DatabaseObjects.SDP_DB_SDPDWH,  # table
        DatabaseObjects.TBL_IVCE_HDR: DatabaseObjects.SDP_DB_SDPDWH,  # table
        DatabaseObjects.TBL_INVC_DTL: DatabaseObjects.SDP_DB_SDPDWH,  # table
        DatabaseObjects.TBL_IVCE_XCTN_LLM_TRNL_MFR_REF: DatabaseObjects.SDP_DB_SDPDWH,  # table
        DatabaseObjects.TBL_IVCE_XCTN_LLM_TRNL_PRDT_REF: DatabaseObjects.SDP_DB_SDPDWH,  # table
        DatabaseObjects.TBL_IVCE_TRKG_MSTR: DatabaseObjects.SDP_DB_SDPDWH,  # table
    }

    def __init__(self, config):
        """
        Initializes the SDP class with the provided configuration.

        Args:
            config: Configuration object containing database connection details.
        """
        self.config = config
        self.create_sqlalchemy_engine()

    async def get_pyodbc_connection(self, database=DatabaseObjects.SDP_DB_SDPDWH):
        """
        Creates and returns a PyODBC connection string for the specified database.

        Args:
            database (str): Name of the database. Defaults to "SDP_DB_SDPDWH".

        Returns:
            str: PyODBC connection string.
        """
        self.pyodbc_connection_string = (
            f"DRIVER={self.config.SDP_DRIVER};SERVER={self.config.SDP_SERVER};DATABASE={database};"
            f"UID={self.config.SDP_UID};PWD={self.config.SDP_PWD};"
        )
        return pyodbc.connect(self.pyodbc_connection_string)
        # conn_string = (f'DRIVER={self.config.SDP_DRIVER};SERVER={self.config.SDP_SERVER};DATABASE={database};'
        #                f'UID={self.config.SDP_UID};PWD={self.config.SDP_PWD};')
        # return await aioodbc.create_pool(dsn=conn_string)

    def create_sqlalchemy_engine(self, database=DatabaseObjects.SDP_DB_SDPDWH):
        """
        Creates a SQLAlchemy engine for the specified database.

        Args:
            database (str): Name of the database. Defaults to "SDP_DB_SDPDWH".

        Returns:
            Engine: SQLAlchemy engine object.
        """
        uid = self.config.SDP_UID
        pwd = self.config.SDP_PWD
        server = self.config.SDP_SERVER
        driver_version = "17"
        linux_lib_name = "libmsodbcsql-17.10.so.6.1"

        self.sqlalchemy_connection_string = (
            f"mssql+pyodbc://{uid}:{pwd}@{server}/{database}?driver=ODBC+Driver+{driver_version}+for+SQL+Server"
            if self.config.environment == Environments.LOCAL
            else (
                f"mssql+pyodbc://{uid}:{pwd}@{server}/{database}"
                f"?driver=/opt/microsoft/msodbcsql{driver_version}/lib64/{linux_lib_name}"
            )
        )

        self.sqlalchemy_engine = create_engine(self.sqlalchemy_connection_string, fast_executemany=True)
        # self.Session = sessionmaker(bind=self.sqlalchemy_engine)

        # self.sqlalchemy_connection_string = (f'mssql+aioodbc://{self.config.SDP_UID}:{self.config.SDP_PWD}@'
        #                                      f'{self.config.SDP_SERVER}/{database}?driver=ODBC+Driver+17+for+SQL+Server')

        # # Async SQLAlchemy engine creation
        # self.sqlalchemy_engine = create_async_engine(self.sqlalchemy_connection_string, echo=True, future=True)

        # # Async session setup
        # self.Session = sessionmaker(self.sqlalchemy_engine, expire_on_commit=False, class_=AsyncSession)

    async def execute_query(self, query):
        """
        Executes a read query asynchronously using PyODBC.

        Args:
            query (str): SQL query to execute.

        Returns:
            pd.DataFrame: DataFrame containing the query results.
        """
        try:
            pool = await self.get_pyodbc_connection()
            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(query)
                    rows = await cursor.fetchall()
                    [logger.info(row) for row in rows]
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise

    @retry_on_deadlock(max_retries=3, backoff_factor=1.0)
    async def fetch_data(self, query) -> DataFrame | Iterator[DataFrame]:
        """
        Fetches data from the database using SQLAlchemy and returns it as a Pandas DataFrame.

        Args:
            query (str): SQL query to execute.

        Returns:
            pd.DataFrame: DataFrame containing the query results.
        """
        try:
            # logger.debug(f"Env: {self.config.environment}, Connection string: {self.sqlalchemy_connection_string}")
            with self.sqlalchemy_engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
                return pd.read_sql(query, conn)

        except OperationalError as e:
            logger.error("Database connection failed. Possible causes:")
            logger.error("- SQL Server is unreachable or not accepting connections.")
            logger.error("- Network issue or wrong server/instance name.")
            logger.error("- Login timeout or SQL Server not allowing remote connections.")
            logger.error(f"Original error: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Error executing read query: {str(e)}")
            raise
        # """
        # Fetches data from the database asynchronously.
        # """
        # try:
        #     async with self.sqlalchemy_engine.connect() as conn:
        #         result = await conn.execute(text(query))
        #         rows = result.fetchall()
        #         if not rows:
        #             return pd.DataFrame()

        #         df = pd.DataFrame(rows, columns=result.keys())
        #         return df

        # except Exception as e:
        #     logger.error(f"Error executing read query: {str(e)}")
        #     raise

    async def update_data(self, query: str, values: tuple, conn=None, retry=True):
        """
        Calls function that executes a parameterized write query (INSERT/UPDATE/MERGE) with optional transactional control.

        Args:
            query (str): SQL query to execute.
            values (Tuple): Values to bind to the query parameters.
            conn (Optional[Engine]): Optional SQLAlchemy connection object for transactional control.
            retry (boolean): Flag to decide whether to retry if any error occurs during write operation at this function level

        Returns:
            None
        """
        if retry:
            # Use retriable update function with decorator
            return await self._update_data_with_retries(query=query, values=values, conn=conn)

        else:
            # Use this if retry is handled at a higher level of transactions
            return await self._update_data(query=query, values=values, conn=conn)

    @retry_on_deadlock(max_retries=3, backoff_factor=1.0)
    async def _update_data_with_retries(self, query: str, values: tuple, conn=None):
        """
        Calls function that executes a parameterized write query (INSERT/UPDATE/MERGE).
        Retries automatically if a deadlock (1205) occurs.

        Args:
            query (str): SQL query to execute.
            values (Tuple): Values to bind to the query parameters.
            conn (Optional[Engine]): Optional SQLAlchemy connection object for transactional control.

        Returns:
            None
        """
        return await self._update_data(query=query, values=values, conn=conn)

    async def _update_data(self, query: str, values: tuple, conn=None):
        """
        Executes a parameterized write query (INSERT/UPDATE/MERGE).

        Args:
            query (str): SQL query to execute.
            values (Tuple): Values to bind to the query parameters.
            conn (Optional[Engine]): Optional SQLAlchemy connection object for transactional control.

        Returns:
            None

        Raises:
            Exception: Reraises any DB exceptions encountered.
        """
        try:

            if conn is None:
                with self.sqlalchemy_engine.begin() as conn:
                    return await self._update_to_db(query=query, params=values, connection=conn)
            else:
                return await self._update_to_db(query=query, params=values, connection=conn)

        except Exception as e:
            logger.error(f"Error executing write query: {str(e)}")
            raise

    async def update_or_insert_data(self, update_q: str, update_params: tuple, insert_q: str, insert_params: tuple, conn=None):
        """
        Executes a parameterized write query (INSERT/UPDATE/MERGE) with optional transactional control.
        Retries automatically if a deadlock (1205) occurs.

        Args:
            query (str): SQL query to execute.
            values (Tuple): Values to bind to the query parameters.
            conn (Optional[Engine]): Optional SQLAlchemy connection object for transactional control.

        Returns:
            None

        Raises:
            Exception: Reraises any DB exceptions encountered.
        """

        try:
            if conn is None:
                with conn as conn:
                    await self._update_or_insert_to_db(
                        uquery=update_q, uparams=update_params, iquery=insert_q, iparams=insert_params, connection=conn
                    )

            # Handle transaction automatically (autocommit behavior)
            else:
                await self._update_or_insert_to_db(
                    uquery=update_q, uparams=update_params, iquery=insert_q, iparams=insert_params, connection=conn
                )

        except Exception as e:
            logger.error(f"Error executing write query: {str(e)}")
            raise

    async def _update_or_insert_to_db(self, iquery, iparams, uquery, uparams, connection):
        """
        Checks whether to update or insert to DB with given connection, query and params.
        It tries to update first, if not success, then inserts.
        """
        result = await self._update_to_db(query=uquery, params=uparams, connection=connection)
        if result.rowcount == 0:
            result = await self._update_to_db(query=iquery, params=iparams, connection=connection)
        return result

    async def _update_to_db(self, query, params, connection):
        """
        Last step to write to DB with given connection, query and params
        """
        result = connection.exec_driver_sql(query, params)
        return result

    async def fetch_data_in_batches(self, query_template, batch_size=1000, params=None):
        """
        Execute a query in batches and return results as a generator

        Args:
            query_template: SQL query template with {offset} and {batch_size} placeholders
            batch_size: Number of records to fetch per batch
            params: Additional parameters to format the query template

        Returns:
            Generator yielding DataFrames for each batch
        """
        if params is None:
            params = {}

        offset = 0
        while True:
            # Format the query with the current offset and batch size
            query = query_template.format(offset=offset, batch_size=batch_size, **params)

            # Execute the query
            df = await self.fetch_data(query)

            # If no more data, stop iteration
            if df.empty:
                break

            yield df

            # If fewer rows than batch_size were returned, we've reached the end
            if len(df) < batch_size:
                break

            offset += batch_size
