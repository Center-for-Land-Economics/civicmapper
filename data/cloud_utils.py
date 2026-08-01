import requests
import pandas as pd
from io import BytesIO
from datetime import datetime
import json
import time
from shapely.geometry import Polygon
import geopandas as gpd
from azure.storage.blob import BlobServiceClient

# ArcGIS public servers commonly drop connections from the default Python UA.
# Use a realistic browser User-Agent so the server responds normally.
_ARCGIS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
}
_DEFAULT_TIMEOUT = 60  # seconds


def get_layer_crs(base_url, dataset_name, layer_id=0, verbose=True, service_type="FeatureServer"):
    """
    Fetch the layer spatial reference WKID from ArcGIS layer metadata.
    Returns an integer WKID or None if it cannot be detected.

    Args:
        base_url: Service root URL, e.g. "https://server/rest/services/Folder"
        dataset_name: Dataset/service name, e.g. "Property_and_Land"
        layer_id: Integer layer ID within the service
        service_type: "FeatureServer" (default) or "MapServer"
    """
    meta_url = f"{base_url}/{dataset_name}/{service_type}/{layer_id}"
    try:
        response = requests.get(
            meta_url, params={"f": "json"},
            headers=_ARCGIS_HEADERS, timeout=_DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        meta = response.json()
    except requests.exceptions.RequestException as e:
        if verbose:
            print(f"⚠️ Failed to fetch layer metadata: {e}")
        return None
    except ValueError:
        if verbose:
            print("⚠️ Layer metadata did not return valid JSON.")
        return None

    sr = meta.get("extent", {}).get("spatialReference", {}) or meta.get("spatialReference", {})
    wkid = sr.get("latestWkid") or sr.get("wkid")
    if verbose:
        print(f"Layer metadata spatialReference: {sr}")
    try:
        return int(wkid) if wkid is not None else None
    except (TypeError, ValueError):
        return None



def get_feature_data_with_geometry(
    dataset_name, base_url, layer_id=0, paginate=False,
    out_epsg=4326, verbose=True, service_type="FeatureServer"
):
    """
    ArcGIS FeatureServer or MapServer downloader with geometry + CRS awareness.

    - Supports both FeatureServer (default) and MapServer via service_type param
    - Detects layer CRS from metadata
    - Optionally requests output in a known CRS (outSR)
    - Validates CRS assumptions

    URL construction:
        query URL  = {base_url}/{dataset_name}/{service_type}/{layer_id}/query
        meta URL   = {base_url}/{dataset_name}/{service_type}/{layer_id}

    So base_url should be the service root EXCLUDING the dataset name and service type.
    Example:
        base_url     = "https://maps2.dcgis.dc.gov/DCGIS/rest/services/DCGIS_DATA"
        dataset_name = "Property_and_Land"
        service_type = "MapServer"
        layer_id     = 53
    → query URL: .../DCGIS_DATA/Property_and_Land/MapServer/53/query
    """

    url = f"{base_url}/{dataset_name}/{service_type}/{layer_id}/query"

    # Detect original CRS from layer metadata
    layer_wkid = get_layer_crs(base_url, dataset_name, layer_id, verbose=verbose, service_type=service_type)

    if layer_wkid is None:
        if verbose:
            print("⚠️ Could not detect layer CRS from metadata. Defaulting to EPSG:3857 (risky).")
        layer_wkid = 3857

    # Count params for pagination
    count_params = {'f': 'json', 'where': '1=1', 'returnCountOnly': 'true'}

    try:
        if paginate:
            total_records = requests.get(
                url, params=count_params,
                headers=_ARCGIS_HEADERS, timeout=_DEFAULT_TIMEOUT,
            ).json().get("count", 0)
            if verbose:
                print(f"Total records in {dataset_name}: {total_records}")
        else:
            total_records = 1000

        all_features = []
        offset = 0
        chunk_size = 1000

        while offset < total_records:
            params = {
                'f': 'json',
                'where': '1=1',
                'outFields': '*',
                'returnGeometry': 'true',
                'resultOffset': offset,
                'resultRecordCount': chunk_size,
                'outSR': out_epsg,               # ✅ request output CRS
                'geometryPrecision': 6
            }

            response = requests.get(
                url, params=params,
                headers=_ARCGIS_HEADERS, timeout=_DEFAULT_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()

            # Print spatialReference info from response
            if verbose and offset == 0:
                sr = data.get("spatialReference", {})
                print("Query response spatialReference:", sr)

            if "features" not in data:
                if verbose:
                    print(f"No features found in response for {dataset_name}")
                break

            num_features = len(data["features"])
            if verbose:
                print(f"Fetched records {offset} to {offset + num_features}")

            # Parse features
            for feature in data["features"]:
                attrs = feature["attributes"]
                geom = feature.get("geometry", None)

                if geom and "rings" in geom:
                    # NOTE: This assumes the first ring is outer boundary.
                    # If holes/multi-polygons exist, this is a simplification.
                    ring = geom["rings"][0]
                    attrs["geometry"] = Polygon(ring)
                    all_features.append(attrs)

            # Stop logic
            if not paginate or num_features < chunk_size:
                break
            offset += num_features

        if not all_features:
            return None

        # ✅ If we requested outSR=4326, then that's what we got back.
        gdf = gpd.GeoDataFrame(all_features, geometry="geometry", crs=f"EPSG:{out_epsg}")

        # A quick sanity check on first geometry bounds
        if verbose:
            b = gdf.total_bounds
            print("Total bounds:", b)
            if abs(b[0]) > 180 or abs(b[1]) > 90:
                print("⚠️ Bounds look non-degree-like even though CRS says degrees. Something may be wrong.")

        return gdf

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from {dataset_name}: {e}")
        return None


def get_feature_data(dataset_name, base_url, layer_id=0):
    """
    Get all features from a service using pagination
    
    Args:
        dataset_name (str): Name of the dataset to fetch
        base_url (str): Base URL for the feature service
        layer_id (int, optional): Layer ID to query. Defaults to 0.
    """
    url = f"{base_url}/{dataset_name}/FeatureServer/{layer_id}/query"
    
    # First, get the count of all features
    count_params = {
        'f': 'json',
        'where': '1=1',
        'returnCountOnly': 'true'
    }
    
    try:
        count_response = requests.get(url, params=count_params)
        count_response.raise_for_status()
        total_records = count_response.json().get('count', 0)
        
        print(f"Total records in {dataset_name}: {total_records}")
        
        # Now fetch the actual data in chunks
        all_features = []
        offset = 0
        chunk_size = 2000  # ArcGIS typically limits to 2000 records per request
        
        while offset < total_records:
            params = {
                'f': 'json',
                'where': '1=1',
                'outFields': '*',
                'returnGeometry': 'false',
                'resultOffset': offset,
                'resultRecordCount': chunk_size
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'features' in data:
                features = [feature['attributes'] for feature in data['features']]
                all_features.extend(features)
                
                print(f"Fetched records {offset} to {offset + len(features)} for {dataset_name}")
                
                if len(features) < chunk_size:
                    break
                    
                offset += chunk_size
            else:
                print(f"No features found in response for {dataset_name}")
                break
                
        return pd.DataFrame(all_features)
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from {dataset_name}: {e}")
        return None

def save_to_azure(df, dataset_name):
    """
    Save DataFrame to Azure Blob Storage
    """
    if df is None or df.empty:
        print(f"No data to save for {dataset_name}")
        return
        
    # Create CSV in memory
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    # Generate blob name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d")
    blob_name = f"{folder_name}/{dataset_name}_{timestamp}.csv"
    
    # Upload to Azure
    try:
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(csv_buffer, overwrite=True)
        print(f"Successfully uploaded {dataset_name} to Azure")
        
        # Save data dictionary separately
        data_dict = {
            'columns': list(df.columns),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'record_count': len(df),
            'timestamp': timestamp
        }
        
        dict_blob_name = f"{folder_name}/{dataset_name}_{timestamp}_dictionary.json"
        dict_blob_client = container_client.get_blob_client(dict_blob_name)
        dict_blob_client.upload_blob(json.dumps(data_dict, indent=2), overwrite=True)
        
    except Exception as e:
        print(f"Error saving {dataset_name} to Azure: {e}")

def get_mapserver_data_with_geometry(dataset_name, base_url, layer_id=0):
    """Get all features from a MapServer service including geometry with pagination"""
    url = f"{base_url}/{dataset_name}/MapServer/{layer_id}/query"
    
    # First, get the count of all features
    count_params = {
        'f': 'json',
        'where': '1=1',
        'returnCountOnly': 'true'
    }
    
    try:
        count_response = requests.get(url, params=count_params)
        count_response.raise_for_status()
        count_data = count_response.json()
        total_records = count_data.get('count', 0)
        
        print(f"Total records in {dataset_name}: {total_records}")
        
        if total_records == 0:
            print("No records found, trying without count check...")
            # Some MapServers don't support count, so try direct query
            params = {
                'f': 'json',
                'where': '1=1',
                'outFields': '*',
                'returnGeometry': 'true',
                'geometryPrecision': 6,
                'resultRecordCount': 2000
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'features' in data and data['features']:
                total_records = len(data['features'])
                print(f"Found {total_records} records in direct query")
            else:
                print("No features found in direct query either")
                return None
        
        # Now fetch the actual data in chunks
        all_features = []
        offset = 0
        chunk_size = 2000  # ArcGIS typically limits to 2000 records per request
        
        while offset < total_records:
            params = {
                'f': 'json',
                'where': '1=1',
                'outFields': '*',
                'returnGeometry': 'true',
                'geometryPrecision': 6,
                'resultOffset': offset,
                'resultRecordCount': chunk_size
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'features' in data and data['features']:
                # Convert ESRI features to GeoDataFrame format
                for feature in data['features']:
                    # Extract attributes
                    attributes = feature['attributes']
                    
                    # Handle geometry - check if it exists and has the expected structure
                    geometry = None
                    if 'geometry' in feature and feature['geometry']:
                        geom_data = feature['geometry']
                        if 'rings' in geom_data and geom_data['rings']:
                            try:
                                # Take the first ring (exterior ring)
                                ring = geom_data['rings'][0]
                                # Create Shapely polygon
                                geometry = Polygon(ring)
                            except Exception as e:
                                print(f"Warning: Could not process geometry for feature: {e}")
                                geometry = None
                        elif 'x' in geom_data and 'y' in geom_data:
                            # Point geometry
                            from shapely.geometry import Point
                            try:
                                geometry = Point(geom_data['x'], geom_data['y'])
                            except Exception as e:
                                print(f"Warning: Could not process point geometry: {e}")
                                geometry = None
                    
                    # Combine attributes and geometry
                    attributes['geometry'] = geometry
                    all_features.append(attributes)
                
                print(f"Fetched records {offset} to {offset + len(data['features'])} of {total_records}")
                
                if len(data['features']) < chunk_size:
                    break
                    
                offset += chunk_size
            else:
                print(f"No features found in response for {dataset_name}")
                break
        
        if all_features:
            # Create GeoDataFrame
            # Cook County uses EPSG:3435 (Illinois State Plane East, NAD83)
            try:
                gdf = gpd.GeoDataFrame(all_features, crs='EPSG:3435')
                
                # Filter out features without geometry if needed
                valid_geom_count = gdf['geometry'].notna().sum()
                print(f"Features with valid geometry: {valid_geom_count} out of {len(gdf)}")
                
                if valid_geom_count > 0:
                    # Convert to WGS84 for compatibility
                    gdf = gdf.to_crs('EPSG:4326')
                    print(f"Successfully loaded {len(gdf)} features")
                    return gdf
                else:
                    print("No features with valid geometry found")
                    # Return as regular DataFrame if no geometry
                    return pd.DataFrame(all_features)
                    
            except Exception as e:
                print(f"Error creating GeoDataFrame: {e}")
                print("Returning as regular DataFrame")
                return pd.DataFrame(all_features)
        
        return None
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from {dataset_name}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def debug_mapserver_structure(dataset_name, base_url, layer_id=0, max_features=5):
    """Debug function to examine the structure of MapServer data"""
    url = f"{base_url}/{dataset_name}/MapServer/{layer_id}/query"
    
    params = {
        'f': 'json',
        'where': '1=1',
        'outFields': '*',
        'returnGeometry': 'true',
        'geometryPrecision': 6,
        'resultRecordCount': max_features
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        print(f"Response keys: {list(data.keys())}")
        
        if 'features' in data and data['features']:
            print(f"Number of features returned: {len(data['features'])}")
            
            for i, feature in enumerate(data['features'][:3]):  # Look at first 3 features
                print(f"\n--- Feature {i+1} ---")
                print(f"Feature keys: {list(feature.keys())}")
                
                if 'attributes' in feature:
                    attrs = feature['attributes']
                    print(f"Attributes keys (first 10): {list(attrs.keys())[:10]}")
                
                if 'geometry' in feature:
                    geom = feature['geometry']
                    if geom:
                        print(f"Geometry keys: {list(geom.keys()) if geom else 'None'}")
                        if 'rings' in geom:
                            print(f"Number of rings: {len(geom['rings'])}")
                            if geom['rings']:
                                print(f"First ring has {len(geom['rings'][0])} points")
                    else:
                        print("Geometry is None")
                else:
                    print("No geometry key found")
        else:
            print("No features found in response")
            
        return data
        
    except Exception as e:
        print(f"Error in debug function: {e}")
        return None

def get_mapserver_data_with_geometry_filtered(dataset_name, base_url, layer_id=0, where_clause="1=1"):
    """Get filtered features from a MapServer service including geometry with pagination"""
    url = f"{base_url}/{dataset_name}/MapServer/{layer_id}/query"
    
    # First, get the count of all features
    count_params = {
        'f': 'json',
        'where': where_clause,
        'returnCountOnly': 'true'
    }
    
    try:
        count_response = requests.get(url, params=count_params)
        count_response.raise_for_status()
        count_data = count_response.json()
        total_records = count_data.get('count', 0)
        
        print(f"Total records matching filter in {dataset_name}: {total_records}")
        
        if total_records == 0:
            print("No records found, trying without count check...")
            # Some MapServers don't support count, so try direct query
            params = {
                'f': 'json',
                'where': where_clause,
                'outFields': '*',
                'returnGeometry': 'true',
                'geometryPrecision': 6,
                'resultRecordCount': 2000
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'features' in data and data['features']:
                total_records = len(data['features'])
                print(f"Found {total_records} records in direct query")
            else:
                print("No features found in direct query either")
                return None
        
        # Now fetch the actual data in chunks
        all_features = []
        offset = 0
        chunk_size = 2000  # ArcGIS typically limits to 2000 records per request
        
        while offset < total_records:
            params = {
                'f': 'json',
                'where': where_clause,
                'outFields': '*',
                'returnGeometry': 'true',
                'geometryPrecision': 6,
                'resultOffset': offset,
                'resultRecordCount': chunk_size
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'features' in data and data['features']:
                # Convert ESRI features to GeoDataFrame format
                for feature in data['features']:
                    # Extract attributes
                    attributes = feature['attributes']
                    
                    # Handle geometry - check if it exists and has the expected structure
                    geometry = None
                    if 'geometry' in feature and feature['geometry']:
                        geom_data = feature['geometry']
                        if 'rings' in geom_data and geom_data['rings']:
                            try:
                                # Take the first ring (exterior ring)
                                ring = geom_data['rings'][0]
                                # Create Shapely polygon
                                geometry = Polygon(ring)
                            except Exception as e:
                                print(f"Warning: Could not process geometry for feature: {e}")
                                geometry = None
                        elif 'x' in geom_data and 'y' in geom_data:
                            # Point geometry
                            from shapely.geometry import Point
                            try:
                                geometry = Point(geom_data['x'], geom_data['y'])
                            except Exception as e:
                                print(f"Warning: Could not process point geometry: {e}")
                                geometry = None
                    
                    # Combine attributes and geometry
                    attributes['geometry'] = geometry
                    all_features.append(attributes)
                
                print(f"Fetched records {offset} to {offset + len(data['features'])} of {total_records}")
                
                if len(data['features']) < chunk_size:
                    break
                    
                offset += chunk_size
            else:
                print(f"No features found in response for {dataset_name}")
                break
        
        if all_features:
            # Create GeoDataFrame
            # Cook County uses EPSG:3435 (Illinois State Plane East, NAD83)
            try:
                gdf = gpd.GeoDataFrame(all_features, crs='EPSG:3435')
                
                # Filter out features without geometry if needed
                valid_geom_count = gdf['geometry'].notna().sum()
                print(f"Features with valid geometry: {valid_geom_count} out of {len(gdf)}")
                
                if valid_geom_count > 0:
                    # Convert to WGS84 for compatibility
                    gdf = gdf.to_crs('EPSG:4326')
                    print(f"Successfully loaded {len(gdf)} features")
                    return gdf
                else:
                    print("No features with valid geometry found")
                    # Return as regular DataFrame if no geometry
                    return pd.DataFrame(all_features)
                    
            except Exception as e:
                print(f"Error creating GeoDataFrame: {e}")
                print("Returning as regular DataFrame")
                return pd.DataFrame(all_features)
        
        return None
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from {dataset_name}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def explore_cities_in_dataset(dataset_name, base_url, layer_id=0, max_features=1000):
    """Explore what cities are available in the dataset"""
    url = f"{base_url}/{dataset_name}/MapServer/{layer_id}/query"
    
    params = {
        'f': 'json',
        'where': '1=1',
        'outFields': 'CITYNAME,site_city',  # Try common city field names
        'returnGeometry': 'false',
        'resultRecordCount': max_features
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'features' in data and data['features']:
            cities = []
            for feature in data['features']:
                attrs = feature['attributes']
                # Check different possible city field names
                city_fields = ['CITYNAME', 'site_city', 'CITY', 'City']
                for field in city_fields:
                    if field in attrs and attrs[field]:
                        cities.append(attrs[field])
                        break
            
            if cities:
                unique_cities = list(set(cities))
                unique_cities.sort()
                print(f"Found {len(unique_cities)} unique cities in sample of {len(cities)} records:")
                for city in unique_cities[:20]:  # Show first 20
                    print(f"  - {city}")
                if len(unique_cities) > 20:
                    print(f"  ... and {len(unique_cities) - 20} more")
                
                # Look for Chicago variations
                chicago_variations = [city for city in unique_cities if 'CHICAGO' in str(city).upper()]
                if chicago_variations:
                    print(f"\nChicago variations found: {chicago_variations}")
                
                return unique_cities
            else:
                print("No city data found in the sample")
                return []
        else:
            print("No features returned")
            return []
            
    except Exception as e:
        print(f"Error exploring cities: {e}")
        return []

def get_mapserver_data_with_resume(dataset_name, base_url, layer_id=0, where_clause="1=1", 
                                  max_records=None, start_offset=0, chunk_delay=1):
    """
    Get filtered features from a MapServer service with resume capability and connection handling
    
    Args:
        dataset_name: Name of the dataset
        base_url: Base URL for the service
        layer_id: Layer ID (default 0)
        where_clause: SQL WHERE clause for filtering
        max_records: Maximum number of records to download (None for all)
        start_offset: Offset to start from (for resuming)
        chunk_delay: Delay in seconds between chunks to avoid overwhelming server
    """
    url = f"{base_url}/{dataset_name}/MapServer/{layer_id}/query"
    
    # First, get the count of all features
    count_params = {
        'f': 'json',
        'where': where_clause,
        'returnCountOnly': 'true'
    }
    
    try:
        print(f"Getting count for filter: {where_clause}")
        count_response = requests.get(url, params=count_params, timeout=30)
        count_response.raise_for_status()
        count_data = count_response.json()
        total_records = count_data.get('count', 0)
        
        print(f"Total records available: {total_records}")
        
        if max_records:
            total_records = min(total_records, max_records)
            print(f"Limited to: {total_records} records")
        
        if total_records == 0:
            print("No records found")
            return None
        
        # Now fetch the actual data in chunks
        all_features = []
        offset = start_offset
        chunk_size = 2000  # Smaller chunks to avoid timeouts
        failed_attempts = 0
        max_failed_attempts = 3
        
        print(f"Starting download from offset {offset}")
        
        while offset < total_records and failed_attempts < max_failed_attempts:
            try:
                params = {
                    'f': 'json',
                    'where': where_clause,
                    'outFields': '*',
                    'returnGeometry': 'true',
                    'geometryPrecision': 6,
                    'resultOffset': offset,
                    'resultRecordCount': chunk_size
                }
                
                print(f"Fetching records {offset} to {offset + chunk_size}...")
                
                response = requests.get(url, params=params, timeout=60)
                response.raise_for_status()
                data = response.json()
                
                if 'features' in data and data['features']:
                    # Convert ESRI features to GeoDataFrame format
                    chunk_features = []
                    for feature in data['features']:
                        # Extract attributes
                        attributes = feature['attributes']
                        
                        # Handle geometry - check if it exists and has the expected structure
                        geometry = None
                        if 'geometry' in feature and feature['geometry']:
                            geom_data = feature['geometry']
                            if 'rings' in geom_data and geom_data['rings']:
                                try:
                                    # Take the first ring (exterior ring)
                                    ring = geom_data['rings'][0]
                                    # Create Shapely polygon
                                    geometry = Polygon(ring)
                                except Exception as e:
                                    geometry = None
                        
                        # Combine attributes and geometry
                        attributes['geometry'] = geometry
                        chunk_features.append(attributes)
                    
                    all_features.extend(chunk_features)
                    print(f"✅ Successfully fetched {len(chunk_features)} records. Total so far: {len(all_features)}")
                    
                    if len(data['features']) < chunk_size:
                        print("Reached end of data")
                        break
                        
                    offset += chunk_size
                    failed_attempts = 0  # Reset failed attempts on success
                    
                    # Add delay between chunks to be nice to the server
                    if chunk_delay > 0:
                        time.sleep(chunk_delay)
                        
                else:
                    print(f"No features found in response")
                    break
                    
            except requests.exceptions.RequestException as e:
                failed_attempts += 1
                print(f"❌ Request failed (attempt {failed_attempts}/{max_failed_attempts}): {e}")
                
                if failed_attempts < max_failed_attempts:
                    wait_time = failed_attempts * 10  # Exponential backoff
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"Max failed attempts reached. Saving what we have so far: {len(all_features)} records")
                    print(f"To resume, use start_offset={offset}")
                    break
        
        if all_features:
            print(f"Creating GeoDataFrame from {len(all_features)} features...")
            
            # Create GeoDataFrame
            try:
                gdf = gpd.GeoDataFrame(all_features, crs='EPSG:3435')
                
                # Filter out features without geometry if needed
                valid_geom_count = gdf['geometry'].notna().sum()
                print(f"Features with valid geometry: {valid_geom_count} out of {len(gdf)}")
                
                if valid_geom_count > 0:
                    # Convert to WGS84 for compatibility
                    gdf = gdf.to_crs('EPSG:4326')
                    print(f"✅ Successfully loaded {len(gdf)} features")
                    return gdf
                else:
                    print("No features with valid geometry found")
                    # Return as regular DataFrame if no geometry
                    return pd.DataFrame(all_features)
                    
            except Exception as e:
                print(f"Error creating GeoDataFrame: {e}")
                print("Returning as regular DataFrame")
                return pd.DataFrame(all_features)
        
        return None
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def ensure_geodataframe(df, geometry_col='geometry'):
    """
    Ensures that a DataFrame with geometry column is converted to a GeoDataFrame.
    If conversion fails, tries to robustly decode geometry values before failing.
    
    Args:
        df: DataFrame or GeoDataFrame
        geometry_col: Name of the geometry column
    
    Returns:
        GeoDataFrame with proper CRS set
    """
    import shapely
    import binascii

    def try_decode_geometry(val):
        """
        Try to decode a geometry value that may be:
        - Already a shapely geometry
        - A WKT string
        - A WKB hex string (bytes or str)
        - A WKB bytes object
        If it cannot be decoded, returns None.
        """
        if isinstance(val, shapely.geometry.base.BaseGeometry):
            return val
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        # Try WKT
        if isinstance(val, str):
            try:
                # Try WKT first
                return shapely.wkt.loads(val)
            except Exception:
                pass
            try:
                # Try WKB hex string
                return shapely.wkb.loads(binascii.unhexlify(val))
            except Exception:
                pass
        # Try WKB bytes
        if isinstance(val, (bytes, bytearray)):
            try:
                return shapely.wkb.loads(val)
            except Exception:
                pass
        return None

    # If already a GeoDataFrame, ensure CRS is set
    if isinstance(df, gpd.GeoDataFrame):
        if df.crs is None:
            df = df.set_crs(epsg=4326)  # WGS84
        return df

    # If regular DataFrame with geometry column, convert to GeoDataFrame
    if geometry_col in df.columns:
        # Convert geometry column from WKT strings to geometry objects if needed
        if df[geometry_col].dtype == 'object':
            try:
                df[geometry_col] = df[geometry_col].apply(wkt.loads)
            except Exception:
                # If wkt.loads fails, try to robustly decode geometry values
                try:
                    df[geometry_col] = df[geometry_col].apply(try_decode_geometry)
                    # Remove rows where geometry could not be decoded
                    n_invalid = df[geometry_col].isna().sum()
                    if n_invalid > 0:
                        print(f"⚠️ {n_invalid} rows had invalid geometry and will be dropped.")
                        df = df[df[geometry_col].notna()]
                except Exception as e:
                    print(f"❌ Could not decode geometry: {e}")
                    raise

        # Convert to GeoDataFrame
        try:
            df = gpd.GeoDataFrame(df, geometry=geometry_col)
        except Exception as e:
            # Try to robustly decode geometry and try again
            try:
                df[geometry_col] = df[geometry_col].apply(try_decode_geometry)
                n_invalid = df[geometry_col].isna().sum()
                if n_invalid > 0:
                    print(f"⚠️ {n_invalid} rows had invalid geometry and will be dropped.")
                    df = df[df[geometry_col].notna()]
                df = gpd.GeoDataFrame(df, geometry=geometry_col)
            except Exception as e2:
                print(f"❌ Could not convert to GeoDataFrame after robust decode: {e2}")
                raise

        # Set CRS if not already set
        if df.crs is None:
            df = df.set_crs(epsg=4326)  # WGS84

        return df

    # Return as-is if no geometry column found
    return df
