import io_utils
import utils
import environment_utils


def main():
    patches = io_utils.read_dump_from_file("C:/Users/Niklas/Desktop/GIT/shrooms/data/patches_weather.dump")
    io_utils.add_weather(patches)
    io_utils.dump_to_file(patches, "C:/Users/Niklas/Desktop/GIT/shrooms/data/patches_weather.dump")
    io_utils.calc_dynamic_value(patches)


if __name__ == "__main__":
    main()
