import zipfile
import requests
import re
import tqdm

# Code adapted from:
# https://github.com/wkentaro/gdown/blob/941200a9a1f4fd7ab903fb595baa5cad34a30a45/gdown/download.py
# https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url


def download_file(url, destination):
    response = requests.get(url, allow_redirects=True, stream=True)
    write_response_to_disk(response, destination)


def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc"
    params = dict(id=file_id, export="download")

    with requests.Session() as session:

        response = session.get(URL, params=params, stream=True)

        # downloads right away
        if "Content-Disposition" in response.headers:
            write_response_to_disk(response, destination)
            return

        # try to find a confirmation token
        token = get_google_drive_confirm_token(response)

        if token:
            params = dict(id=id, confirm=token)
            response = session.get(URL, params=params, stream=True)

        # download if confirmation token worked
        if "Content-Disposition" in response.headers:
            write_response_to_disk(response, destination)
            return

        # extract download url from confirmation page
        url = get_url_from_gdrive_confirmation(response.text)
        response = session.get(url, stream=True)

        write_response_to_disk(response, destination)


def get_google_drive_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def get_url_from_gdrive_confirmation(contents):
    url = ""
    for line in contents.splitlines():
        m = re.search(r'href="(\/uc\?export=download[^"]+)', line)
        if m:
            url = "https://docs.google.com" + m.groups()[0]
            url = url.replace("&amp;", "&")
            break
        m = re.search('id="downloadForm" action="(.+?)"', line)
        if m:
            url = m.groups()[0]
            url = url.replace("&amp;", "&")
            break
        m = re.search('id="download-form" action="(.+?)"', line)
        if m:
            url = m.groups()[0]
            url = url.replace("&amp;", "&")
            break
        m = re.search('"downloadUrl":"([^"]+)', line)
        if m:
            url = m.groups()[0]
            url = url.replace("\\u003d", "=")
            url = url.replace("\\u0026", "&")
            break
        m = re.search('<p class="uc-error-subcaption">(.*)</p>', line)
        if m:
            error = m.groups()[0]
            raise RuntimeError(error)
    if not url:
        raise RuntimeError(
            "Cannot retrieve the public link of the file. "
            "You may need to change the permission to "
            "'Anyone with the link', or have had many accesses."
        )
    return url


def get_download_progress_bar(response):
    total = response.headers.get("Content-Length")
    if total is not None:
        total = int(total)

    if total is not None:
        pbar = tqdm.tqdm(total=total, unit="B", unit_scale=True)

    def update(progress):
        if total is not None:
            pbar.update(progress)

    return update


def write_response_to_disk(response, destination):
    CHUNK_SIZE = 32768

    update_progress_bar = get_download_progress_bar(response)

    with open(destination, "wb") as file:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                file.write(chunk)
                update_progress_bar(len(chunk))


def unzip_file(file, destination):
    with zipfile.ZipFile(file, "r") as zip_file:
        zip_file.extractall(destination)
