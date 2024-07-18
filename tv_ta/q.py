import inspect
import sqlite3
from contextlib import contextmanager
from typing import Type, Callable, Any, Generic, TypeVar, overload, Self

T = TypeVar("T")
AsT = TypeVar("AsT")
As = Type[T] | Callable[[sqlite3.Row], T]
SqliteConnOrCursor = sqlite3.Connection | sqlite3.Cursor


class QProto(Generic[T]):
    def select_all(self, sql, params=(), *, as_: As | None = None) -> list[T]: ...
    def select_one(self, sql, params=(), *, as_: As | None = None) -> Any: ...
    def select_val(self, sql, params=()) -> Any: ...
    def execute(self, sql, params=(), *, commit=False) -> None: ...
    def commit(self) -> None: ...

    @contextmanager
    def commit_after(self) -> None: ...


class Q(QProto):
    """
    Обертка над sqlite3.Cursor, для удобного использования в коде.

    q = Q(
      sqlite_conn_or_cursor=...,
      select_all_as=..., # type: cls | func
    )
    BookQ = Q.factory(
      select_all_as=..., # type: cls | func
    )
    q = BookQ(sqlite_cur)

    q.select_all(sql)
    q.select_all(sql, as=...)
    q.select_one(sql)
    q.select_val(sql)

    q.execute(sql)
    q.execute(sql, commit=True)
    q.commit(sql)
    with q.commit_after():
      q.execute(...)
      q.execute(...)

    class BookQ:
      def __init__(sqlite_cur):
        self.q = Q(sqlite_cur, select _all_as=Book)

      def select_wip():
        return self.q.select_all('select * from books where status = "wip"')

    book_q.q.select_all('select * from books')
    book_q.q.select_val('select count(*) from books')
    """

    def __init__(
        self,
        sqlite_conn_or_cursor: SqliteConnOrCursor,
        *,
        select_as: As | None = None,
    ):
        if isinstance(sqlite_conn_or_cursor, sqlite3.Connection):
            self.sqlite_cur = sqlite_conn_or_cursor.cursor()
        elif isinstance(sqlite_conn_or_cursor, sqlite3.Cursor):
            self.sqlite_cur = sqlite_conn_or_cursor

        self.sqlite_cur.row_factory = sqlite3.Row

        self._select_as = select_as

    @classmethod
    def factory(
        cls,
        select_as: As[T] | None = None,
    ) -> Callable[[SqliteConnOrCursor], QProto[T]]:
        def new(
            sqlite_conn_or_cursor: sqlite3.Connection | sqlite3.Cursor,
        ):
            return cls(
                sqlite_conn_or_cursor,
                select_as=select_as,
            )

        return new

    def select_all(self, sql, params=(), *, as_: As | None = None) -> list:
        if not isinstance(params, (list, tuple)):
            params = (params,)

        rows = self.sqlite_cur.execute(sql, params).fetchall()

        rows = [self._apply_as(row, as_) for row in rows]

        return rows

    def select_one(self, sql, params=(), *, as_: As | None = None) -> Any:
        if not isinstance(params, (list, tuple)):
            params = (params,)

        row = self.sqlite_cur.execute(sql, params).fetchone()

        row = self._apply_as(row, as_)

        return row

    def select_val(self, sql, params=()) -> Any:
        if not isinstance(params, (list, tuple)):
            params = (params,)

        return self.sqlite_cur.execute(sql, params).fetchone()[0]

    def execute(self, sql, params=(), *, commit=False):
        if not isinstance(params, (list, tuple)):
            params = (params,)

        self.sqlite_cur.execute(sql, params)

        if commit:
            self.commit()

    def commit(self):
        self.sqlite_cur.connection.commit()

    @contextmanager
    def commit_after(self):
        yield
        self.commit()

    def _apply_as(self, row_or_rows, as_: As | None = None):
        as_ = as_ or self._select_as
        if not as_:
            return row_or_rows

        is_func = inspect.isfunction(as_) or inspect.ismethod(as_)

        if is_func:
            return as_(row_or_rows)
        else:
            return as_(**row_or_rows)


class QQProto(Generic[T]):

    @overload
    def list_all(self, *, as_: None = None, order_by: str | None = None) -> list[T]: ...
    @overload
    def list_all(self, *, as_: Type[AsT], order_by: str | None = None) -> list[AsT]: ...
    @overload
    def list_all(self, *, as_: Callable[[sqlite3.Row], AsT], order_by: str | None = None) -> list[AsT]: ...
    def list_all(self, *, as_=None, order_by: str | None = None): ...


class QQ(QQProto[T], QProto[T]):
    def __init__(self, table: str, q: QProto[T]):
        self._table = table
        self.q = q

    @classmethod
    def factory(
        cls,
        table: str,
        cursor_factory: Callable[[SqliteConnOrCursor], QProto[T]],
    ) -> Callable[[SqliteConnOrCursor], Self]:
        def new(cursor: sqlite3.Cursor):
            return cls(table, cursor_factory(cursor))

        return new

    def __getattr__(self, item):
        return getattr(self.q, item)

    def list_all(self, *, as_=None, order_by: str | None = None):
        q = f"select * from {self._table}"
        if order_by:
            q += f" order by {order_by}"

        return self.q.select_all(q, as_=as_)
