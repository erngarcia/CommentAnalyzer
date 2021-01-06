#Connect to database.

import mysql.connector
from mysql.connector import Error

try:
    connection = db = mysql.connector.connect(
        host=,
        user=,
        passwd=,
        database=,
    )

    sql_select = "SELECT message FROM posts"
    cursor = connection.cursor()
    cursor.execute(sql_select)
    record = cursor.fetchall()
    print(record)

except Error as e:
    print("Error reading data from database")
