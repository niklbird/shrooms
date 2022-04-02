import sql_utils
import reparse_utils
import io_utils
import constants

'''
This is still mostly a TODO
'''

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


def fit_findings_to_patches(patches):
    cursor, con = sql_utils.connect_database("../data/locations.db")
    findings = sql_utils.get_table(cursor, "findings")
    patch_middles = [patches[i].middle for i in range(len(patches))]
    for i in range(len(findings)):
        cord = [findings[0], findings[1]]
        dist = []
        closest_point = reparse_utils.find_n_closest_points(patch_middles, cord, 1)
        a = 0
    pass



def create_table():
    cursor,con = sql_utils.connect_database("../data/locations.db")
    values = [["id", "integer"], ["location_x", "float"], ["location_y", "float"],
              ["mushroom", "text"], ["temperature", "float"],
              ["rain", "float"]]
    sql_utils.create_table(cursor, "findings", values)
    con.commit()


def create_mapping_table():
    cursor, con = sql_utils.connect_database("../data/locations.db")
    values = [["patch_number", "integer"], ["id", "integer"]]
    sql_utils.create_table(cursor, "mappings", values)
    con.commit()

cursor, con = sql_utils.connect_database("../data/locations.db")

patches_split = io_utils.read_patches_from_folder(constants.pwd + "/data/dumps/patches/")

patches = io_utils.flatten_patches(patches_split)


fit_findings_to_patches(patches)

print(sql_utils.get_table(cursor, "findings")[0][1])

#sql_utils.remove_table(cursor, "findings")
#con.commit()
#sql_utils.remove_table(cursor, "findings")
#create_table()
#new_finding(str(sql_utils.size_table(cursor, "findings")),'50.0','10.0','Steinpilz','15.0','0.5')
#new_finding(0,0,0,0,0)
#sql_utils.list_table(cursor, "findings")
#con.commit()
#print(sql_utils.size_table(cursor, "findings"))
