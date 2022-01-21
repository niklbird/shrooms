import Map from './node_modules/ol/Map.js';
import View from './node_modules/ol/View.js';

import TileLayer from './node_modules/ol/layer/Tile.js';
import OSM from './node_modules/ol/source/OSM';
import GeoJSON from './node_modules/ol/format/GeoJSON';
import VectorLayer from './node_modules/ol/layer/Vector';
import VectorSource from './node_modules/ol/source/Vector';
import {Fill, Style} from './node_modules/ol/style';


const source4 = new VectorSource({
  title: 'added Layer',
  url: './data.txt',
  format: new GeoJSON(),
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

const map = new Map({
  target: 'map',
  layers: [
    new TileLayer({
      source: new OSM()
    }),
    layer2,
  ],
  view: new View({
    center: [1118760.88, 6636047.68],
    zoom: 7
  })
});