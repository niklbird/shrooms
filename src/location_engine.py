import sql_utils

# Engine to use real finding location in processing of probabilties
def new_finding(id,location, mushroom, time, temperature_14, rain_14):
    # Add a new finding at specific location of mushroom
    # Also store data of find, temperature of last 14 days and rains of last 14 days
    # May be interesting for later evaluation of data
    cursor, con = sql_utils.connect_database("../data/locations.db")
    rows = "id,location_x,location_y,mushroom,temperature,rain"
    #values = ['1','50.0','10.0','Steinpilz','15.0','0.5']
    values = [id, location,  mushroom, time, temperature_14, rain_14]
    sql_utils.insert_data_table(cursor, "findings", rows, values)
    con.commit()
    pass


def get_findings_location(location):
    pass


def create_table():
    cursor,con = sql_utils.connect_database("../data/locations.db")
    values = [["id", "integer"], ["location_x", "float"], ["location_y", "float"],
              ["mushroom", "text"], ["temperature", "float"],
              ["rain", "float"]]
    sql_utils.create_table(cursor, "findings", values)
    con.commit()


cursor,con = sql_utils.connect_database("../data/locations.db")
#sql_utils.remove_table(cursor, "findings")
#con.commit()
#sql_utils.remove_table(cursor, "findings")
#create_table()
new_finding(str(sql_utils.size_table(cursor, "findings")),'50.0','10.0','Steinpilz','15.0','0.5')
#new_finding(0,0,0,0,0)
sql_utils.list_table(cursor, "findings")
#con.commit()
#print(sql_utils.size_table(cursor, "findings"))
