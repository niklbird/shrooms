import io_utils
import constants
import utils


def main():
    # Read in pre-processed data points with tree-data
    utils.reparse2()
    patches = io_utils.read_dump_from_file(constants.pwd + "/data/patches_weather2.dump")
    # Query weather-data from DWD
    utils.add_weather(patches)
    # Dump file with current weather data
    io_utils.dump_to_file(patches, constants.pwd + "/data/patches_weather2.dump")
    # Calculate the actual mushroom probabilities
    utils.calc_dynamic_value(patches)
    # Dump final result to a file for usage in JS?
    io_utils.dump_to_file(patches, constants.pwd + "/data/patches_probabilities.dump")
    io_utils.write_to_GEOJSON(patches)


if __name__ == "__main__":
    main()
