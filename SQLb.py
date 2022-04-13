import pymysql as sql
from setting import database_set
connect = sql.connect(host='localhost', user='root', password='123456',)


def con():
    t = sql.connect(host=database_set[0],
                    user=database_set[1],
                    password=database_set[2],
                    database=database_set[3],
                    autocommit=True)
    return t.cursor()


def createone():
    cur = connect.cursor()
    cur.execute(
        "CREATE DATABASE IF NOT EXISTS battry DEFAULT CHARACTER SET utf8 COLLATE utf8_general_ci;"
    )
    cur = con()


def insert(title, author, content, status):
    cur = con()
    val = "'" + title + "', '" + author + "', '" + content + "', " + str(status)
    ins = """INSERT INTO `result`(
        ret,
        author,
        url,
        kind
    ) VALUE(""" + val + ")"""
    cur.execute(ins)


def find(kind, a):
    if kind == 1:
        f = "SELECT content FROM `result` WHERE status = " + str(a)
    elif kind == 2:
        f = "SELECT content FROM `result` WHERE author = " + a
    else:
        f = "SELECT content FROM `result` WHERE title = " + a
    c = con()
    c.execute(f)
    ret = c.fetchmany()
    print(ret)


if __name__ == '__main__':
    createone()
