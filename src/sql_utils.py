import sqlite3


def create_table(cursor, name, values):
    txt = ''
    for value in values:
        txt += str(value[0]) + ' ' + str(value[1]) + ', '
    # Remove last space and last comma
    txt = txt[:len(txt)-2]
    toex = ('''CREATE TABLE {name} ({values})''').format(name=name, values=txt)
    cursor.execute(toex)
    return 1


def list_table(cursor, table):
    print('*' * 49)
    print("Listing Table: " + str(table))
    for row in cursor.execute('SELECT * FROM ' + str(table)):
        print("  -->  " + str(row))
    print('*' * 49)


def insert_data_table(cursor, table, values):
    # Remember to properly use delimiters for string arguments
    txt = ''
    for value in values:
        txt += str(value) + ','
    txt = txt[:len(txt)]
    cursor.execute(f"INSERT INTO {table} VALUES ({txt})")
    return 1


def get_data_table(cursor, table, attribute, attr_name):
    # Careful, String arguments need to be put in extra string delimiters: ''
    toex= '''SELECT * FROM {table} WHERE {attribute} = {attr_name}'''.format(table=table, attribute=attribute, attr_name=attr_name)
    ret = cursor.execute(toex)
    return ret


def update_data_table(cursor, table, attribute, value, idattribute, idvalue):
    cursor.execute(f"UPDATE {table} SET {attribute} = {value} WHERE {idattribute} = {idvalue}")
    return 1


def delta_update_data_table(cursor, table, attribute, attr_name, delta):
    val = int(get_data_table(cursor, table, attribute, attr_name))
    val += delta
    update_data_table(cursor, table, attribute, val, attr_name)


def remove_table(cursor, table):
    toex = f"DROP TABLE {table}"
    cursor.execute(toex)
    return 1


def connect_database(name):
    con = sqlite3.connect(name)
    cursor = con.cursor()
    return cursor
