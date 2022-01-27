import sql_utils

# Engine to use real finding location in processing of probabilties
def new_finding(location, mushroom, time, temperature_14, rain_14):
    # Add a new finding at specific location of mushroom
    # Also store data of find, temperature of last 14 days and rains of last 14 days
    # May be interesting for later evaluation of data
    cursor = sql_utils.connect_database("../data/locations.db")
    values = "50.0,10.0,Steinpilz,15.0,0.5"
    sql_utils.insert_data_table(cursor, "findings",values)
    pass


def get_findings_location(location):
    pass


def create_table():
    cursor = sql_utils.connect_database("../data/locations.db")
    values = [["location_x", "float"], ["location_y", "float"],
              ["mushroom", "text"], ["temperature", "float"],
              ["rain", "float"]]
    sql_utils.create_table(cursor, "findings", values)


new_finding(0,0,0,0,0)

