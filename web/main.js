import './style.css';
import {Map, View} from 'ol';
import TileLayer from 'ol/layer/Tile';
import OSM from 'ol/source/OSM';
import GeoJSON from 'ol/format/GeoJSON';
import 'ol/ol.css';
import Draw, {
  createBox,
  createRegularPolygon,
} from 'ol/interaction/Draw';
import VectorLayer from 'ol/layer/Vector';
import VectorSource from 'ol/source/Vector';
import Vector from 'ol/source/Vector';
import {Fill, Stroke, Style, Text, Circle as CircleStyle} from 'ol/style';
import MultiPoint from 'ol/geom/MultiPoint';
import Projection from 'ol/proj/Projection';

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

const typeSelect = document.getElementById('type');

let draw; // global so we can remove it later
function addInteraction() {
  let value = typeSelect.value;
  if (value !== 'None') {
    let geometryFunction;
    if (value === 'Square') {
      value = 'Circle';
      geometryFunction = createRegularPolygon(4);
    } else if (value === 'Box') {
      value = 'Circle';
      geometryFunction = createBox();
    } else if (value === 'Star') {
      value = 'Circle';
      geometryFunction = function (coordinates, geometry) {
        const center = coordinates[0];
        const last = coordinates[coordinates.length - 1];
        const dx = center[0] - last[0];
        const dy = center[1] - last[1];
        const radius = Math.sqrt(dx * dx + dy * dy);
        const rotation = Math.atan2(dy, dx);
        const newCoordinates = [];
        const numPoints = 12;
        for (let i = 0; i < numPoints; ++i) {
          const angle = rotation + (i * 2 * Math.PI) / numPoints;
          const fraction = i % 2 === 0 ? 1 : 0.5;
          const offsetX = radius * fraction * Math.cos(angle);
          const offsetY = radius * fraction * Math.sin(angle);
          newCoordinates.push([center[0] + offsetX, center[1] + offsetY]);
        }
        newCoordinates.push(newCoordinates[0].slice());
        if (!geometry) {
          geometry = new Polygon([newCoordinates]);
        } else {
          geometry.setCoordinates([newCoordinates]);
        }
        return geometry;
      };
    }
    draw = new Draw({
      source: source,
      type: value,
      geometryFunction: geometryFunction,
    });
    map.addInteraction(draw);
  }
}

typeSelect.onchange = function () {
  map.removeInteraction(draw);
  addInteraction();
};

document.getElementById('undo').addEventListener('click', function () {
  draw.removeLastPoint();
});

addInteraction();

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

const highlightStyle = new Style({
  stroke: new Stroke({
    color: '#f00',
    width: 1,
  }),
  fill: new Fill({
    color: 'rgba(255,0,0,0.1)',
  }),
  text: new Text({
    font: '12px Calibri,sans-serif',
    fill: new Fill({
      color: '#000',
    }),
    stroke: new Stroke({
      color: '#f00',
      width: 3,
    }),
  }),
});

const featureOverlay = new VectorLayer({
  source: new VectorSource(),
  map: map,
  style: function (feature) {
    highlightStyle.getText().setText(feature.get('name'));
    return highlightStyle;
  },
});

let highlight;

const displayFeatureInfo = function (pixel) {
  const feature = map.forEachFeatureAtPixel(pixel, function (feature) {
    return feature;
  });
  const info = document.getElementById('info');
  if (feature) {
    info.innerHTML = feature.getId() + ': ' + feature.get('name');
  } else {
    info.innerHTML = '&nbsp;';
  }
  if (feature !== highlight) {
    if (highlight) {
      featureOverlay.getSource().removeFeature(highlight);
    }
    if (feature) {
      featureOverlay.getSource().addFeature(feature);
    }
    highlight = feature;
  }
};

map.on('pointermove', function (evt) {
  if (evt.dragging) {
    return;
  }
  const pixel = map.getEventPixel(evt.originalEvent);
  displayFeatureInfo(pixel);
});

map.on('click', function (evt) {
  displayFeatureInfo(evt.pixel);
});