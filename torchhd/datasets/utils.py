#
# MIT License
#
# Copyright (c) 2023 Mike Heddes, Igor Nunes, Pere Vergés, Denis Kleyko, and Danny Abraham
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import zipfile
import requests
import tqdm


def download_file(url, destination):
    response = requests.get(url, allow_redirects=True, stream=True)
    write_response_to_disk(response, destination)


def download_file_from_google_drive(file_id, destination):
    try:
        import gdown
    except ImportError:
        raise ImportError(
            "Downloading files from Google drive requires gdown to be installed, see: https://github.com/wkentaro/gdown"
        )

    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, destination)


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
