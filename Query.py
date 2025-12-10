from curses import COLOR_WHITE
import sqlite3
from re import L
from sqlite3 import Connection, Cursor, OperationalError
from typing import Any, Literal


class SqliteDatabase:
    def __init__(self, database: str):
        self.database = database
        self.conn: Connection | None = None
        self.cursor: Cursor | None = None

    def __enter__(self):
        try:
            self.conn = sqlite3.connect(self.database)
            self.cursor = self.conn.cursor()
        except Exception as e:
            print(f"Error occured in connecting to the database {self.database}. Error Details: {e}")

        return self.query

    def __exit__(self, exc_type, exc, tb):
        assert self.conn is not None
        assert self.cursor is not None

        self.cursor.close()
        self.conn.close()

        return False  # dont suppress the error

    def query(
        self,
        query: str,
        *,
        is_updation=False,  # is the current query contains some kind of updation ?? Doesnt return anything if true
        return_rows: None | Literal["str"] | Literal["tuple"] = None,
        table_heading: str | None = None,  # Title printed before printing output
    ) -> None | tuple[tuple[str, ...], ...]:
        assert self.conn is not None
        assert self.cursor is not None

        try:
            results = self.cursor.execute(query)
            self.conn.commit()
        except Exception as err:
            print("** Row / Column names with spaces should be enlcosed within quotes **")
            raise err

        if is_updation:
            return

        rows: list[Any] = results.fetchall()
        columns_headers: tuple[str, ...] = tuple(str(col[0]) for col in results.description)

        lines: tuple[tuple[str, ...], ...] = tuple((columns_headers, *rows))
        
        if return_rows == "tuple":
            return lines
        elif return_rows is None:
            # printing table header if provided
            if table_heading is not None:
                print(table_heading)

            # Finding max column width
            column_widths: list[int] = []

            for col_idx in range(len(lines[0])):
                widths = []
                for row_idx in range(len(lines)):
                    widths.append(len(str(lines[row_idx][col_idx])))
                column_widths.append(max(widths))

            # Printing column headers
            border_top_bottom = "+" + "-" * (sum(column_widths) + 3 * len(column_widths) - 1)  + "+"
            print(border_top_bottom)
            print("| ", end="")
            for idx, col_label in enumerate(lines[0]):
                print(str(col_label).ljust(column_widths[idx]), end=" | ")
            print()
            print(border_top_bottom)

            for row in rows[1:]:
                print("| ", end="")
                for idx, col_val in enumerate(row):
                    print(str(col_val).ljust(column_widths[idx]), end=" | ")
                print()

            print(border_top_bottom)
            return None


def main():
    with SqliteDatabase("example.db") as query:
        query("select count(*) as 'wwwwwwweadaeeeeeeeeeeeeeeeeeeeeeeee', date() from item group by item_id")
        query("select count(*) as 'eeeeee', date(), time(date()) from item group by item_id")


if __name__ == "__main__":
    main()
