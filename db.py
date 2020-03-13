import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="",
  database="pharma_m"
)
mycursor = mydb.cursor()

sql = "SELECT m_id, probability FROM prob ORDER BY id DESC LIMIT 2"
# val = ("1", "0.0045")
mycursor.execute(sql)

# mydb.commit()

# print(mycursor.rowcount, "record inserted.")
myresult = mycursor.fetchall()

for x in myresult:
  print(x[0], ' ', x[1])


