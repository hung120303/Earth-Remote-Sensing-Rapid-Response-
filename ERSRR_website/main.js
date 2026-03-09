import Map from 'ol/Map.js';
import View from 'ol/View.js';
import KML from 'ol/format/KML.js';
import { fromExtent } from 'ol/geom/Polygon';
import HeatmapLayer from 'ol/layer/Heatmap.js';
import TileLayer from 'ol/layer/Tile.js';
import { fromLonLat } from 'ol/proj';
import StadiaMaps from 'ol/source/StadiaMaps.js';
import VectorSource from 'ol/source/Vector.js';
import VectorLayer from 'ol/layer/Vector.js';
import Feature from 'ol/Feature.js';
import { getBottomLeft, getTopRight } from 'ol/extent';
import XYZ from 'ol/source/XYZ.js';
import { transformExtent } from 'ol/proj';

const US_Center = [-98.583333, 39.833333];
const US_WebMercator = fromLonLat(US_Center, 'EPSG:4326');

const vectorEarthquake = new HeatmapLayer({
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
  zIndex: 10
});

const boxSource = new VectorSource();
const boxLayer = new VectorLayer({
  source: boxSource,
  zIndex: 5 
});

let s2Layer = new TileLayer({
  source: new XYZ({ url: '' }),
  zIndex: 0
});

const raster = new TileLayer({
  source: new StadiaMaps({
    layer: 'outdoors',
  }),
  zIndex: -1
});

const map = new Map({
  layers: [raster, s2Layer, boxLayer, vectorEarthquake],
  target: 'map',
  view: new View({
    center: fromLonLat(US_Center),
    zoom: 4,
  }),
});

map.on('moveend', async function(){
  const extent = map.getView().calculateExtent(map.getSize());  
  const extent4326 = transformExtent(extent,'EPSG:3857','EPSG:4326');
  const [minx,miny,maxx,maxy] = extent4326;
  
  try {
    const response = await fetch(
      `http://localhost:5000/sentinel?minx=${minx}&miny=${miny}&maxx=${maxx}&maxy=${maxy}`
    );
    const data = await response.json();
    s2Layer.getSource().setUrl(data.tile_url);
  }
  catch (e)
  {
    console.error('Failed to fetch S2 tile:', e);
  }
  
  boxSource.clear();
  const boxFeature = new Feature(fromExtent(extent));
  boxSource.addFeature(boxFeature);
  
})

