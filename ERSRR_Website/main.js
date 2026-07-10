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
import GeoTIFF from 'ol/source/GeoTIFF.js';


const US_Center = [-98.583333, 39.833333];
const US_WebMercator = fromLonLat(US_Center, 'EPSG:4326');
const API_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000').replace(/\/$/, '');
const MIN_INFERENCE_ZOOM = 14;
const statusElement = document.getElementById('status');

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

const predictionLayer = new TileLayer({
  source: new XYZ({
    url: ''
  }),
  opacity: 1.0,
  zIndex: 2
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
  layers: [raster, s2Layer, predictionLayer, boxLayer, vectorEarthquake],
  target: 'map',
  view: new View({
    projection: 'EPSG:3857',
    center: fromLonLat(US_Center),
    zoom: 4,
  }),
});

let timeout;
let activeRequest;

map.on('moveend', async function(){
  clearTimeout(timeout);
  activeRequest?.abort();

  if (map.getView().getZoom() < MIN_INFERENCE_ZOOM) {
    statusElement.textContent = `Zoom to level ${MIN_INFERENCE_ZOOM} or closer to run research inference.`;
    return;
  }

  timeout = setTimeout(async () => {
    activeRequest = new AbortController();
    const extent = map.getView().calculateExtent(map.getSize());  
    const extent4326 = transformExtent(extent,'EPSG:3857','EPSG:4326');
    const [minx,miny,maxx,maxy] = extent4326;
    statusElement.textContent = 'Selecting imagery and running inference…';

    try {
      const response = await fetch(
        `${API_BASE}/sentinel`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ minx, miny, maxx, maxy }),
          signal: activeRequest.signal,
        },
      );
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || `Inference failed with HTTP ${response.status}`);
      }
      s2Layer.getSource().setUrl(data.tile_url);
      predictionLayer.getSource().setUrl(`${API_BASE}${data.prediction_tile_url}`);
      predictionLayer.getSource().refresh();
      statusElement.textContent = `${data.scene_id} · ${data.product} · research-only probability mask`;
    }
    catch (e) {
      if (e.name === 'AbortError') return;
      console.error('Failed to fetch S2 tile:', e);
      statusElement.textContent = e.message;
    }
    
    boxSource.clear();
    const boxFeature = new Feature(fromExtent(extent));
    boxSource.addFeature(boxFeature);
  }, 500);
})

const lonInput = document.getElementById('lon');
const latInput = document.getElementById('lat');
const goToLocationButton = document.getElementById('goToLocationButton');

goToLocationButton.addEventListener('click', () => {
  const lon = parseFloat(lonInput.value);
  const lat = parseFloat(latInput.value);
  if (isNaN(lon) || isNaN(lat)) {
    alert("Invalid coordinates");
    return;
  }

  const newCenter = fromLonLat([lon, lat]);

  map.getView().animate({
    center: newCenter,
    zoom: 15,        // adjust zoom level
    duration: 1000   // smooth animation
  });
});
