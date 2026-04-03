from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="emergency_system")


def get_coordinates(location):

    if location is None:
        return None, None

    try:
        # Bias search to India
        geo = geolocator.geocode(location + ", India")

        if geo:
            return geo.latitude, geo.longitude

    except:
        pass

    return None, None