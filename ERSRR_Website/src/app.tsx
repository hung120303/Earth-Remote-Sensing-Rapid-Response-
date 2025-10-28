/**
 * Copyright 2024 Google LLC
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *    https://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

import React from 'react';
import { createRoot } from "react-dom/client";
import { APIProvider, Map, AdvancedMarker, MapCameraChangedEvent, Pin } from '@vis.gl/react-google-maps';


type Poi ={ key: string, location: google.maps.LatLngLiteral }
const locations: Poi[] = [
  {key: 'pioneerCourthouse', location: { lat: 45.5180, lng: -122.6780  }},
  {key: 'modaCenter', location: { lat: 45.531609, lng: -122.667236 }},
  {key: 'internationalRoseTestGarden', location: { lat: 45.519091, lng: -122.705657 }},
  {key: 'pittockMansion', location: { lat: 45.5252, lng: -122.71629 }},
  {key: 'powellsCityOfBooks', location: { lat: 45.5207, lng: -122.6756 }},
];

const App = () => (
    <APIProvider apiKey={'AIzaSyBP7-ymEMYiD4yw46oFgJ_EQI69d7VsC0k'} onLoad={() => console.log('Google Maps API loaded')}>
        <Map
            defaultZoom={13}
            defaultCenter={{ lat: 45.5152, lng: -122.6784 }}
            onCameraChanged={(ev: MapCameraChangedEvent) =>
                console.log('Map camera changed:', ev.detail.center, 'zoom:', ev.detail.zoom)
            }>

        </Map>
    </APIProvider>
);

const PoiMarkers = (props: {pois: Poi[]}) => {
  return (
    <>
      {props.pois.map( (poi: Poi) => (
        <AdvancedMarker
          key={poi.key}
          position={poi.location}>
        <Pin background={'#FBBC04'} glyphColor={'#000'} borderColor={'#000'} />
        </AdvancedMarker>
      ))}
    </>
  );
};

const container = document.getElementById('app');
if (!container) {
  throw new Error("Root container '#app' not found");
}
const root = createRoot(container);
root.render(<App />);

export default App;