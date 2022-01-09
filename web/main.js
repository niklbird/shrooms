import Map from './node_modules/ol/Map.js';
import View from './node_modules/ol/View.js';

import TileLayer from './node_modules/ol/layer/Tile.js';
import OSM from './node_modules/ol/source/OSM';
import GeoJSON from './node_modules/ol/format/GeoJSON';
import VectorLayer from './node_modules/ol/layer/Vector';
import VectorSource from './node_modules/ol/source/Vector';
import {Fill, Stroke, Style, Text, Circle as CircleStyle} from './node_modules/ol/style';
const styles2 = [
  /* We are using two different styles for the polygons:
   *  - The first style is for the polygons themselves.
   *  - The second style is to draw the vertices of the polygons.
   *    In a custom `geometry` function the vertices of a polygon are
   *    returned as `MultiPoint` geometry, which will be used to render
   *    the style.
   */
  new Style({
    fill: new Fill({
      color: 'rgba(255, 0, 255, 0.1)',
    })
  }),
];

const geojsonObject2 = {
  "type": "FeatureCollection", 
  "crs": {
    "type": "name", 
    "properties": 
    {"name": "EPSG:4326"}}, 
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon", 
        "coordinates": [[
        [6, 50],
        [7, 50],
        [7, 51],
        [6, 51]
        ]]}, 
      "properties": {"color": "rgba(0, 0, 255, 0.5)"}}, 
       ]}

const geojsonObject = {
  'type': 'FeatureCollection',
  'crs': {
    'type': 'name',
    'properties': {
      'name': 'EPSG:3857',
    },
  },
  'features': [
    {
      'type': 'Feature',
      'geometry': {
        'type': 'Polygon',
        'coordinates': [
          [
            [-5e6, 6e6],
            [-5e6, 8e6],
            [-3e6, 8e6],
            [-3e6, 6e6],
            [-5e6, 6e6],
          ],
        ],
      },
      "properties": {
        "color": 'rgba(255, 255, 255, 0.1)'
      },
    },
    {
      'type': 'Feature',
      'geometry': {
        'type': 'Polygon',
        'coordinates': [
          [
            [-2e6, 6e6],
            [-2e6, 8e6],
            [0, 8e6],
            [0, 6e6],
            [-2e6, 6e6],
          ],
        ],
      },
      "properties": {
        "color": 'rgba(0, 0, 255, 0.5)'
      },
    },
    {
      'type': 'Feature',
      'geometry': {
        'type': 'Polygon',
        'coordinates': [
          [
            [1e6, 6e6],
            [1e6, 8e6],
            [3e6, 8e6],
            [3e6, 6e6],
            [1e6, 6e6],
          ],
        ],
      },
      "properties": {
        "color": 'rgba(0, 0, 255, 0.9)'
      },
    },
    {
      'type': 'Feature',
      'geometry': {
        'type': 'Polygon',
        'coordinates': [
          [
            [-2e6, -1e6],
            [-1e6, 1e6],
            [0, -1e6],
            [-2e6, -1e6],
          ],
        ],
      },
      "properties": {
        "color": 'rgba(255, 0, 255, 0.5)'
      },
    },
  ],
};


const source4 = new VectorSource({
  title: 'added Layer',
  url: 'data.txt',
  format: new GeoJSON(),
});

const source2 = new VectorSource({
  features: new GeoJSON({defaultDataProjection:'EPSG:3857'}).readFeatures(geojsonObject2, {
      dataProjection: 'EPSG:4326',
      featureProjection: 'EPSG:3857'
    }),
});
  


const layer2 = new VectorLayer({
  source: source4,
  style: function (feature) {
    return new Style({
      fill: new Fill({
        color: feature.getProperties().color,
      }),
    });
  },
});

source2.forEachFeature(function (feature){
  var styles3 = [
    /* We are using two different styles for the polygons:
     *  - The first style is for the polygons themselves.
     *  - The second style is to draw the vertices of the polygons.
     *    In a custom `geometry` function the vertices of a polygon are
     *    returned as `MultiPoint` geometry, which will be used to render
     *    the style.
     */
    new Style({
      fill: new Fill({
        color: 'rgba(0, 222, 233, 0.5)',
      })
    }),
  ];
feature.setStyle(styles3);
}
)


const source = new VectorSource({wrapX: false});

const vector = new VectorLayer({
  source: source,
});

const map = new Map({
  target: 'map',
  layers: [
    new TileLayer({
      source: new OSM()
    }),
    layer2,
  ],
  view: new View({
    center: [0, 0],
    zoom: 2
  })
});


const style = new Style({
  fill: new Fill({
    color: 'rgba(255, 255, 255, 0.6)',
  }),
  stroke: new Stroke({
    color: '#319FD3',
    width: 1,
  }),
  text: new Text({
    font: '12px Calibri,sans-serif',
    fill: new Fill({
      color: '#000',
    }),
    stroke: new Stroke({
      color: '#fff',
      width: 3,
    }),
  }),
});


//map.on('pointermove', function (evt) {
//  if (evt.dragging) {
//    return;
//  }
//  const pixel = map.getEventPixel(evt.originalEvent);
//  displayFeatureInfo(pixel);
//});

//map.on('click', function (evt) {
//  displayFeatureInfo(evt.pixel);
//});