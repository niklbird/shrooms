import io_utils
import constants
import utils
import reparse_utils
import factor_calculations
import time


'''
Main file of the mushroom app. Here is where all the magic happens.
'''

Reparse = False

Recover = False

def main():
    start = time.time()
    if Reparse:
        start_cord = [50.327834, 7.578448]
        end_cord = [49.598154, 9.553240]
        corners = [start_cord[0], start_cord[1], end_cord[0], end_cord[1]]

        patches = reparse_utils.create_points(start_cord[0], start_cord[1], end_cord[0], end_cord[1],
                                              constants.point_dist, constants.points_per_patch_sqrt)

        patches_split = utils.split_patches(patches, 2000)

        file_names = io_utils.generate_file_names(len(patches_split))

        if not Recover:
            io_utils.clear_directory(constants.pwd + "/data/dumps/patches/")

        start_point = 0

        if Recover:
            start_point = io_utils.get_dumpamount_in_folder(constants.pwd + "/data/dumps/patches/")
        first_reparse = True
        for i in range(start_point, len(patches_split)):
            parsed = reparse_utils.reparse(patches_split[i], corners, first_reparse)
            io_utils.dump_to_file(parsed, constants.pwd + file_names[i])
            first_reparse = False

    patches_split = io_utils.read_patches_from_folder(constants.pwd + "/data/dumps/patches/")

    for patch in patches_split:
        factor_calculations.calc_static_values(patch)

    patches = io_utils.flatten_patches(patches_split)

    # Read in pre-processed data points with tree-data
    # patches = io_utils.read_dump_from_file(constants.pwd + "/data/dumps/patches_weather.dump")

    # Query weather-data from DWD
    utils.add_weather(patches)

    # Dump file with current weather data
    io_utils.dump_to_file(patches, constants.pwd + "/data/dumps/patches_weather.dump")

    # Calculate the actual mushroom probabilities
    factor_calculations.calc_dynamic_value(patches)

    # Dump final result to a file for usage in JS
    # io_utils.dump_to_file(patches, constants.pwd + "/data/dumps/patches_probabilities.dump")
    io_utils.write_to_GEOJSON(patches)
    end = time.time()
    print("Total Time for this run: " + str(end - start))


if __name__ == "__main__":
    main()
