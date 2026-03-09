import ee
from flask import Flask, request, jsonify
from flask_cors import CORS

print("Server starting...")

ee.Authenticate() 
ee.Initialize(project='ersrr-475700') # Ensure this project ID is correct and you have access

app = Flask(__name__)
CORS(app)  # allow cross-origin requests

@app.route("/sentinel")
def sentinel():

    minx = float(request.args.get("minx"))
    miny = float(request.args.get("miny"))
    maxx = float(request.args.get("maxx"))
    maxy = float(request.args.get("maxy"))

    roi = ee.Geometry.Rectangle([minx, miny, maxx, maxy])

    image = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filterBounds(roi)
        .filterMetadata('CLOUD_COVERAGE_ASSESSMENT', 'LESS_THAN', 5)
        .filterDate("2022-01-01","2022-12-31")
        .select(["B4","B3","B2"])
    )

    vis = {"min":0,"max":3000,"bands":["B4","B3","B2"]}

    map_id = image.getMapId(vis)

    return jsonify({
        "tile_url": map_id["tile_fetcher"].url_format
    })

if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True)