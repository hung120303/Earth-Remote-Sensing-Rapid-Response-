import Map from 'ol/Map.js';
import View from 'ol/View.js';
import KML from 'ol/format/KML.js';
import HeatmapLayer from 'ol/layer/Heatmap.js';
import TileLayer from 'ol/layer/Tile.js';
import { fromLonLat } from 'ol/proj';
import StadiaMaps from 'ol/source/StadiaMaps.js';
import VectorSource from 'ol/source/Vector.js';

const US_Center = [-98.583333, 39.833333];
const US_WebMercator = fromLonLat(US_Center, 'EPSG:4326');

const vector = new HeatmapLayer({
  source: new VectorSource({
    url: 'data/2012_Earthquakes_Mag5.kml',
    format: new KML({
      extractStyles: false,
    }),
  }),
  blur: 5,
  radius: 10,
  weight: function (feature) {
    const name = feature.get('name');
    const magnitude = parseFloat(name.substr(2));
    return magnitude - 5;
  },
});

const raster = new TileLayer({
  source: new StadiaMaps({
    layer: 'outdoors',
  }),
});

new Map({
  layers: [raster, vector],
  target: 'map',
  view: new View({
    center: US_WebMercator,
    projection: 'EPSG:4326',
    zoom: 4,
  }),
});

