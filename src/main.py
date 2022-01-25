import io_utils
import constants
import utils
import reparse_utils

'''
Main file of the mushroom app. Here is where all the magic happens.
'''

Reparse = True

def main():
    if Reparse:
        patches = reparse_utils.create_points(49.959518, 8.729952, 49.821941, 9.067095,
                                              constants.point_dist, constants.points_per_patch_sqrt)

        patches_split = utils.split_patches(patches, 300)

        parsed_patches = []
        for patch in patches_split:
            parsed_patches.append(reparse_utils.reparse(patch))

        io_utils.patches_to_folder(parsed_patches)

        # io_utils.dump_to_file(patches, constants.pwd + "/data/dumps/patches_weather.dump")

    patches_split = io_utils.read_patches_from_folder(constants.pwd + "/data/dumps/patches/")

    patches = io_utils.flatten_patches(patches_split)

    # Read in pre-processed data points with tree-data
    # patches = io_utils.read_dump_from_file(constants.pwd + "/data/dumps/patches_weather.dump")

    # Query weather-data from DWD
    utils.add_weather(patches)

    # Dump file with current weather data
    io_utils.dump_to_file(patches, constants.pwd + "/data/dumps/patches_weather.dump")

    # Calculate the actual mushroom probabilities
    utils.calc_dynamic_value(patches)

    # Dump final result to a file for usage in JS
    # io_utils.dump_to_file(patches, constants.pwd + "/data/dumps/patches_probabilities.dump")
    io_utils.write_to_GEOJSON(patches)


if __name__ == "__main__":
    main()
