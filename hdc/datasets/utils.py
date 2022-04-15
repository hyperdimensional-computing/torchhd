import zipfile
import requests

CHUNK_SIZE = 32768


def download_file(url, destination):
    response = requests.get(url, allow_redirects=True, stream=True)
    write_response_to_disk(response, destination)


def download_file_from_google_drive(file_id, destination):
    # Based on https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url

    URL = "https://docs.google.com/uc"
    params = dict(id=file_id, export="download")

    with requests.Session() as session:

        response = session.get(URL, params=params, stream=True)
        token = get_google_drive_confirm_token(response)

        if token:
            params = dict(id=id, confirm=token)
            response = session.get(URL, params=params, stream=True)

        write_response_to_disk(response, destination)


def get_google_drive_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def write_response_to_disk(response, destination):
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def unzip_file(file, destination):
    with zipfile.ZipFile(file, "r") as zip_file:
        zip_file.extractall(destination)
