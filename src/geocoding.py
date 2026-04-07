import ssl
from geopy.geocoders import Nominatim

try:
    import certifi
    ssl_context = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    ssl_context = ssl.create_default_context()

geolocator = Nominatim(user_agent="emergency_system", ssl_context=ssl_context)


def get_coordinates(location):

    if location is None:
        return None, None

    try:
        # Capitalize properly
        location = location.title()

        # Add India bias
        query = f"{location}, India"

        geo = geolocator.geocode(query, timeout=10)

        if geo:
            return geo.latitude, geo.longitude

    except Exception as e:
        print("Geocoding error:", e)

    return None, None