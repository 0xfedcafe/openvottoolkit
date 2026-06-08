"""Network utilities for the toolkit."""

import os
import re
import shutil
import tempfile
import time
import typing
from urllib.parse import urlparse, urljoin

import requests

from vot import ToolkitException, get_logger

class NetworkException(ToolkitException):
    """Exception raised when a network error occurs."""  
    pass

def get_base_url(url: str) -> str:
    """Returns the base url of a given url.

    :param url: The url to parse.
    :type url: str

    :returns: The base url.
    :rtype: str"""
    return url.rsplit('/', 1)[0]
    
def is_absolute_url(url: str) -> bool:
    """Returns True if the given url is absolute.

    :param url: The url to parse.
    :type url: str

    :returns: True if the url is absolute, False otherwise.
    :rtype: bool"""
    
    return bool(urlparse(url).netloc)

def join_url(url_base: str, url_path: str) -> str:
    """Joins a base url with a path.

    :param url_base: The base url.
    :type url_base: str
    :param url_path: The path to join.
    :type url_path: str

    :returns: The joined url.
    :rtype: str"""
    if is_absolute_url(url_path):
        return url_path
    return urljoin(url_base, url_path)

def get_url_from_gdrive_confirmation(contents: str) -> str:
    """Returns the url of a google drive file from the confirmation page.

    :param contents: The contents of the confirmation page.
    :type contents: str

    :returns: The url of the file.
    :rtype: str"""
    url = ''
    for line in contents.splitlines():
        m = re.search(r'href="(\/uc\?export=download[^"]+)', line)
        if m:
            url = 'https://docs.google.com' + m.groups()[0]
            url = url.replace('&amp;', '&')
            return url
        m = re.search('confirm=([^;&]+)', line)
        if m:
            confirm = m.groups()[0]
            url = re.sub(r'confirm=([^;&]+)', r'confirm='+confirm, url)
            return url
        m = re.search(r'"downloadUrl":"([^"]+)', line)
        if m:
            url = m.groups()[0]
            url = url.replace('\\u003d', '=')
            url = url.replace('\\u0026', '&')
            return url
    raise NetworkException("Unable to retrieve google drive confirmation url")


def is_google_drive_url(url: str) -> bool:
    """Returns True if the given url is a google drive url.

    :param url: The url to parse.
    :type url: str

    :returns: True if the url is a google drive url, False otherwise.
    :rtype: bool"""
    m = re.match(r'^https?://drive.google.com/uc\?id=.*$', url)
    return m is not None

def download_json(url: str) -> dict:
    """Downloads a JSON file from the given url.

    :param url: The url to parse.
    :type url: str

    :returns: The JSON content.
    :rtype: dict"""
    try:
        return requests.get(url).json()
    except requests.exceptions.RequestException as e:
        raise NetworkException("Unable to read JSON file {}".format(e))


class _DownloadSink:
    """Write destination for :func:`download`.

    Encapsulates the two output kinds so the download loop never has to branch on the
    output type: a path is buffered through a temporary file (so an interrupted download
    never corrupts the destination, and is only copied over on success), while an already
    open binary handle is written to directly.
    """

    def __init__(self, output: str | typing.IO[bytes]) -> None:
        """Open the underlying write handle for the given path or file handle."""
        if isinstance(output, str):
            self._destination: str | None = output
            self._handle: typing.IO[bytes] = tempfile.NamedTemporaryFile(delete=False)
            self._tempfile: str | None = self._handle.name
        else:
            self._destination = None
            self._handle = output
            self._tempfile = None

    @property
    def handle(self) -> typing.IO[bytes]:
        """The binary handle that download chunks are written to."""
        return self._handle

    def reset(self) -> None:
        """Rewind to the start before a fresh (non-resumable) retry."""
        self._handle.seek(0)

    def commit(self) -> None:
        """Finalize a successful download, moving the temp file onto the destination path."""
        if self._destination is not None and self._tempfile is not None:
            self._handle.close()
            shutil.copy(self._tempfile, self._destination)

    def cleanup(self) -> None:
        """Remove the temporary file, if one was used."""
        if self._tempfile is not None and os.path.exists(self._tempfile):
            try:
                os.remove(self._tempfile)
            except OSError:
                pass


def download(url: str, output: str | typing.IO[bytes], callback: typing.Callable | None = None, chunk_size: int = 1024*32, retry: int = 10) -> str | typing.IO[bytes]:
    """Downloads a file from the given url. Supports google drive urls. callback for
    progress report, automatically resumes download if connection is closed.

    :param url: The url to parse.
    :type url: str
    :param output: The output file path or file handle.
    :type output: str
    :param callback: The callback function for progress report.
    :type callback: function
    :param chunk_size: The chunk size for download.
    :type chunk_size: int
    :param retry: The number of retries.
    :type retry: int

    :raises NetworkException: If the file is not available."""
    
    logger = get_logger()

    with requests.session() as sess:

        is_gdrive = is_google_drive_url(url)

        # HTTP statuses that indicate a transient condition (rate limiting or a
        # temporary server-side error) and are therefore worth retrying. A bulk
        # dataset download fires many requests in quick succession, so the server
        # commonly throttles with 429/503 even though the file exists.
        transient_statuses = {429, 500, 502, 503, 504}
        status_attempts = 0

        while True:
            res = sess.get(url, stream=True)

            if not res.status_code == 200:
                if res.status_code in transient_statuses and status_attempts < retry:
                    status_attempts += 1
                    # Honor Retry-After when the server provides it, otherwise
                    # back off exponentially (capped at 30s).
                    retry_after = res.headers.get("Retry-After", "")
                    delay = float(retry_after) if retry_after.isdigit() else min(2 ** status_attempts, 30)
                    logger.warning("HTTP %d for %s, retrying in %.0fs (%d/%d)",
                                   res.status_code, url, delay, status_attempts, retry)
                    res.close()
                    time.sleep(delay)
                    continue
                raise NetworkException("File not available (HTTP {}) for {}".format(res.status_code, url))

            if 'Content-Disposition' in res.headers:
                # This is the file
                break
            if not is_gdrive:
                break

            # Need to redirect with confirmation. ``get_url_from_gdrive_confirmation``
            # returns a URL or raises ``NetworkException`` if none could be found.
            url = get_url_from_gdrive_confirmation(res.text)

        total_str = res.headers.get('Content-Length')
        if total_str is None:
            raise NetworkException("Content-Length header missing for {}".format(url))
        total = int(total_str)

        sink = _DownloadSink(output)
        filehandle = sink.handle

        position = 0
        progress = False

        try:
            while True:
                try:
                    for chunk in res.iter_content(chunk_size=chunk_size):
                        filehandle.write(chunk)
                        position += len(chunk)
                        progress = True
                        if callback:
                            callback(position, total)

                    if position < total:
                        raise requests.exceptions.RequestException("Connection closed")

                    sink.commit()
                    break

                except requests.exceptions.RequestException as e:
                    if not progress:
                        logger.warning("Error when downloading file, retrying")
                        retry-=1
                        if retry < 1:
                            raise NetworkException("Unable to download file {}".format(e))
                        res = sess.get(url, stream=True)
                        sink.reset()
                        position = 0
                    else:
                        logger.warning("Error when downloading file, trying to resume download")
                        res = sess.get(url, stream=True, headers=({'Range': f'bytes={position}-'} if position > 0 else None))
                    progress = False

            if position < total:
                raise NetworkException("Unable to download file")

        except IOError as e:
            raise NetworkException("Local I/O Error when downloading file: %s" % e)
        finally:
            sink.cleanup()

        return output


def download_uncompress(url: str, path: str) -> None:
    """Downloads a file from the given url and uncompress it to the given path.

    :param url: The url to parse.
    :type url: str
    :param path: The path to uncompress the file.
    :type path: str

    :raises NetworkException: If the file is not available."""
    from vot.utilities import extract_files
    _, ext = os.path.splitext(urlparse(url).path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
        tmp_file = f.name
    try:
        download(url, tmp_file)
        extract_files(tmp_file, path)
    finally:
        if os.path.exists(tmp_file):
            os.unlink(tmp_file)
