# Installation Guide

This document contains information on how to setup the mushroom app locally and start developing. I did not yet test this guide, so if you follow it and encouter a problem, please hit me up or create a pull-request with the missing information.

The engine of the app is written in Python3 so you will need it installed to run the app. 

It generates a data.txt file which contains shapes with the probabilities in GEOJSON format.

The frontend runs on Javascript, to run a test-server I recommend using NodeJS, but you may use any server you like (the live-server runs on nginx, so this also works).

# Cloning the Repo
If you want to contribute code to this project, you can fork this project and clone the repo. You need git lfs installed to clone this project as it includes some bigger files.
Once your feature is finished, please open a Pull request, I will look at it as soon as possible. Please also include a short description on what you did, to help me understand your code.


## Python requirements
The Python code has a number of dependencies. To install them, you require an installation of pip3, preferably the newest pip3 version as some dependencies may not resolve correctly otherwise.
I listed (hopefully) all dependencies in the requirements.txt file. To bulk install them, you can run 
    pip3 install requirements.txt
If you encounter a problem with this, please open an issue as some dependencies may be missing / not resolve correctly anymore.

## Web Server
The mushroom app uses a webserver to render the map. For development, a vile dev server can be used. 
To install this server and all dependencies, install NodeJS. Then open a console inside the web directory of the repo and run
    npm install
It will install all node dependencies. To start the web-sever, run
    npm start
Now you should be able to open the map inside a web browser, by going to the address displayed by vile (Default is localhost:3000).

# Web Engine
The map used to display the data is from the openstreetmap-project. It includes a library to deal with shapes in Javascript, called openlayers (ol). This is what I used to display the probabilites on the map. If you want to work on the Javascript part of this app, you may start by reading into openlayers and openstreetmap. You will find that the current code is not very complex as I'm not an expert in web-dev, so feel free to add features and improvements.
If you want to pack the code to run it on a web-server, you may use webpack. 

# Strucutre of the Python Engine
The app currently consists of a number of different Python files. The main function is located in main.py. It contains the discrete steps executed to generate the final data.txt file.

### Reparse
The first (optional) step in the process is reparsing. which recreates the point grid that is used to calculate the discrete probabilities. Reparsing is by default deactivated as it is not necessary to recreate the grid for every run of the program (Which takes forever). The grid is saved in a pickle-dump inside the data directory and will be loaded over the function read_dump_from_file().
Every function related to reparsing is located inside reparse_utils.py. 
During the reparse, the following steps happen:
1. Tree Data is read from the file. This tree data contains information about the vegitation at every point in Germany, so the data-set is quite large.
2. A grid of points is created. The points are combined into patches, each patch containing a fixed amount of points (Default: 100). This speeds up the later calculations significantly.
3. A second auxilary grid for the trees is created. This is also used later to speed up some calculations.
4. Tree information is pre-processed to speed up later calculations. The function does a form of clustering of the tree shapes. This is necessary as we can not iterate every tree-shape for every grid point to find out in which tree shape this point lies. Or maybe we could if we would want to wait for 3 months.
5. Fit trees to patches. This is the continuation of the previous step. The tree shapes are fitted to the grid patches. This takes by far the most processing power as this function finds the correct tree shape for each point in the grid. Furthermore the algorithm I use for this is not 100% determenistic, it fails in edge cases to identify the correct tree shape for each point. However this is not a problem in practice. If you have an idea how to fix this without sacrifizing a large quantity of processing time, feel free to open a pull-request.
6. Calc static values. This calculates a static probability value for each point in the grid, to speed up mushroom probabiltiy calculation later. E.g. if a point lies inside a city, it will always have a probabilty of 0 to spawn any mushrooms. 

### Add weather
This step pulls weather information from the DWD and adds it to each point. It is approximated that each point inside a patch will have the same weather, as the default patch has a size of 1km x 1km.
The API of DWD is only querried for the stations we need. Each point stores information about the weather of the last 30 days, as this is the period most relevant to mushroom growth. It will query the API for data on every day of the last 30 days for which it does not yet have data. 
It is important to note that the API gives more information than we need, so the data is also filtered to only store the relevant values.

### Calculating dynamic Values
This function calculates the probabilty at each point of the grid for a number of different mushrooms, using the local vegitation and weather from the last 30 days. This calculation is currently quite basic and will certainly need a more scientific approach later. 

### Write to GEOJSON
The last step creates the shapes that can later be displayed on the map. This includes major data reduction steps, which is necessary to prevent the application from lagging hard. It turns out that displaying a few hundret million seperate squares is quite invovled, so we wont do that. The optimations are as follows:
1. The data is removed from the patches and combined into one giant patch (basically a single continous rectangular grid).
2. The amount of shapes is reduced by:
    - Merging points with equal probability values inside each row into a single larger shape. So xxxooooxx would be reduced to XOX with corresponding new rectangle borders.
    - Merging points between rows. This looks at each row and the next row and again searches for points with equal probabilites. If the probabilities are equal, it will again combine the points to a larger shape.
      This is, however, not as trivial as it may sound. It has to be ensured that the borders of the newly created (non rectangular) shapes have the exact correct order. They will be rendered on the map according to the order they are stored, so if you do not think alot about the correct pattern to order these points, you will get an unrecognizable result.
3. Shapes that have a zero probabilty will be removed from the data set. They will not be displayed anyway so why keep them?
Then at last, the data will be written into a file according to GEOJSON format. The color of each shape will be taken from the probability that mushrooms may be found inside this shapes. 

### General thoughts
The code contents quite a large amount of comments, explaining the ideas of most complex steps in the code. If you contribute to the projects, please also document what you do inside the code, espacially if it is not clear at first sight.
If you have trouble understanding anything inside the project, feel free to open an Issue in GitHub!
Happy Coding :)
