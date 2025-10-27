import sqlite3
from func_timeout import func_timeout, FunctionTimedOut



class SpiderDatabaseSearcher:
    def __init__(self):
        self.timeout = 30

    def search(self, sql_query: str, db_path: str, db_id: str=None):
        try:
            results = func_timeout(self.timeout, self._search, args=(sql_query, db_path, db_id))

        except FunctionTimedOut:
            # print("Function timed out!")
            results = []

        return results

    def _search(self, sql_query: str, db_path: str, db_id: str=None):
        conn = sqlite3.connect(db_path)
        # Connect to the database
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        return results
