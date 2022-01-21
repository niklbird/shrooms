import io_utils
import constants
import utils

'''
Main file of the mushroom app. Here is where all the magic happens.
'''

def main():
    # If desired -> Re-Parse everything
    # This required downloading all auxiliary data
    # utils.reparse()

    # Read in pre-processed data points with tree-data
    patches = io_utils.read_dump_from_file(constants.pwd + "/data/dumps/patches_weather.dump")

    # Query weather-data from DWD
    utils.add_weather(patches)

    # Dump file with current weather data
    io_utils.dump_to_file(patches, constants.pwd + "/data/dumps/patches_weather.dump")

    # Calculate the actual mushroom probabilities
    utils.calc_dynamic_value(patches)

    # Dump final result to a file for usage in JS
    io_utils.dump_to_file(patches, constants.pwd + "/data/dumps/patches_probabilities.dump")
    io_utils.write_to_GEOJSON(patches)


if __name__ == "__main__":
    main()
