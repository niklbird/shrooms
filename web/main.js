import Map from './node_modules/ol/Map.js';
import View from './node_modules/ol/View.js';

import TileLayer from './node_modules/ol/layer/Tile.js';
import OSM from './node_modules/ol/source/OSM';
import GeoJSON from './node_modules/ol/format/GeoJSON';
import VectorLayer from './node_modules/ol/layer/Vector';
import VectorSource from './node_modules/ol/source/Vector';
import {Fill, Style} from './node_modules/ol/style';
var today = new Date().getDate();

const source = new VectorSource({
  title: 'added Layer',
  url: './data' + today + '.json',
  format: new GeoJSON(),
});

const layer = new VectorLayer({
  source: source,
  style: function (feature) {
    return new Style({
      fill: new Fill({
        color: feature.getProperties().color,
      }),
    });
  },
});


const source_grainy = new VectorSource({
  title: 'added Layer',
  url: './data_grainy' + today + '.json',
  format: new GeoJSON(),
});
 
const layer_grainy = new VectorLayer({
  source: source_grainy,
  style: function (feature) {
    return new Style({
      fill: new Fill({
        color: feature.getProperties().color,
      }),
    });
  },
});

layer.setMinZoom(10.5);

layer_grainy.setMaxZoom(10.5);


const map = new Map({
  target: 'map',
  layers: [
    new TileLayer({
      source: new OSM()
    }),
    layer_grainy,layer
  ],
  view: new View({
    center: [1118760.88, 6636047.68],
    zoom: 7
  })
});