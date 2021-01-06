import mysql.connector
from mysql.connector import Error

try:
    connection = db = mysql.connector.connect(
        host="commentanalysdb.cibkkuhlvze9.us-east-1.rds.amazonaws.com",
        user="admin",
        passwd="admin001",
        database="db01"
    )

    sql_select = "SELECT message FROM posts"
    cursor = connection.cursor()
    cursor.execute(sql_select)
    record = cursor.fetchall()
    print(record)

except Error as e:
    print("Error reading data from database")
