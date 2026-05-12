import json
import os
import re
import shutil
import sys
from datetime import datetime
from io import StringIO

import cv2
import humanize
import numpy as np
import pandas as pd
import requests
from ccapi.ccapi_ipynb import gui as notebook_gui
from cloudmesh.common.Shell import Shell
# from cloudmesh.common.Printer import Printer
from cloudmesh.common.Tabulate import Printer
from cloudmesh.common.systeminfo import os_is_linux, os_is_mac


def df_from_csv(content):
    df = pd.read_csv(StringIO(content))
    return df


class Computer:

    def beep(self):
        sys.stdout.write('\a')


class CCAPI:
    """
    A Python class to interface with a Cannon camera via the Canon CCAPI using
    REST.
    """

    # A LIST OF FEATUURES THAT MAY BE USEFUL TO BE ADDED
    #
    # http://<IP>:8080/ccapi
    # /ver100/functions/beep
    # /ver100/functions/displayoff
    # /ver100/functions/sensorcleaning  # button under lens
    # metering
    # drive

    def __init__(self, ip=None, port=8080, debug=True, settings="canon-settings.json"):
        """
        Connects to the camera given the ip and port.
        Prioritizes a wired USB connection (e.g., over RNDIS/NCM interface)
        before falling back to Wi-Fi. It will check CANON_USB_IP, then
        common USB IPs, and finally CANON_IP or the provided ip.

        :param ip: The ip number in the form xxx.xxx.xxx.xxx
        :type ip: string
        :param port: The port number. Default is 8080
        :type port: integer
        """
        self.debug = debug
        self.settings_file = settings
        self.port = port

        # Build priority list of IPs to check
        candidate_ips = []
        if os.environ.get("CANON_USB_IP"):
            candidate_ips.append(os.environ["CANON_USB_IP"])

        # Common IPs assigned to USB network interfaces (Smartphone connection)
        candidate_ips.extend(["192.168.1.1", "192.168.1.2", "192.168.42.1", "192.168.42.129", "192.168.225.1"])

        if ip is not None:
            candidate_ips.append(ip)
        elif os.environ.get("CANON_IP"):
            candidate_ips.append(os.environ.get("CANON_IP"))

        self.ip = None
        for test_ip in candidate_ips:
            if self.debug:
                print(f"Testing connection to IP: {test_ip}:{self.port} ...")
            try:
                # Use a short timeout to quickly skip unavailable IPs
                resp = requests.get(f'http://{test_ip}:{self.port}/ccapi/ver100/deviceinformation', timeout=1.0)
                if resp.status_code == 200:
                    if self.debug:
                        print(f"Successfully connected to CCAPI at {test_ip}.")
                    self.ip = test_ip
                    break
            except requests.RequestException:
                continue

        if self.ip is None:
            print("Warning: Could not establish CCAPI connection to any known IPs.")
            # Default to the originally provided IP or CANON_IP if nothing worked
            self.ip = ip if ip else os.environ.get("CANON_IP")

        if self.ip:
            self._init_endpoints()
            self.settings = self.get_settings()
        else:
            self.endpoints = {}
            self.settings = {}

    def _init_endpoints(self):
        """
        Discovers all available CCAPI versions from the camera's root
        /ccapi endpoint and builds a lookup table mapping each base path
        (e.g. '/deviceinformation') to the highest available versioned
        path (e.g. '/ccapi/ver140/deviceinformation').
        """
        self.endpoints = {}  # base_path -> highest versioned full path
        self._api_versions = []  # sorted list of discovered versions
        try:
            r = requests.get(
                f'http://{self.ip}:{self.port}/ccapi', timeout=3.0
            )
            data = r.json()
            # data looks like: {"ver100": [{"path": "/ccapi/ver100/...", ...}, ...], "ver110": [...]}
            # Sort version keys so we process lowest first; later (higher)
            # versions overwrite earlier ones, leaving us with the highest.
            versions = sorted(data.keys())
            self._api_versions = versions
            if self.debug:
                print(f"Camera supports CCAPI versions: {versions}")
            for ver in versions:
                for entry in data[ver]:
                    full_path = entry["path"]  # e.g. "/ccapi/ver110/devicestatus/storage"
                    # Strip the /ccapi/verXXX prefix to get the base path
                    base = re.sub(r'^/ccapi/ver\d+', '', full_path)
                    self.endpoints[base] = full_path
        except Exception as e:
            if self.debug:
                print(f"Warning: Could not discover CCAPI endpoints: {e}")
            # Fallback: empty dict means _resolve_path will use ver100

    def _resolve_path(self, path):
        """
        Resolves a path to the highest available CCAPI version.

        Accepts paths in any of these forms:
          - Already-versioned:  '/ccapi/ver100/deviceinformation'
          - Base path only:     '/deviceinformation'
          - Special:            '/ccapi'  (root discovery endpoint)

        Returns the full versioned path, or falls back to ver100 if
        the endpoint was not found during discovery.

        :param path: The API path to resolve
        :type path: string
        :return: The resolved full path
        :rtype: string
        """
        if path is None:
            return path

        # Don't touch the root discovery endpoint
        if path == '/ccapi':
            return path

        # Strip an existing /ccapi/verXXX prefix to get the base path
        base = re.sub(r'^/ccapi/ver\d+', '', path)

        # Look up the highest version from our discovery table
        if base in self.endpoints:
            return self.endpoints[base]

        # Fallback: if not found in discovery, use ver100
        return f'/ccapi/ver100{base}'

    def gui(self, key=None):
        w = notebook_gui(self, key=key)
        return w

    def create_cache(self, directory):
        """
        Not yet implemented.
        This provides a location where the files from the camera
        are to be stored locally on the computer. If the file is already
        downloaded the download will be skipped

        For now this function just creates the directory for the cache

        :param directory: The name of the directory
        :type directory: string
        :return: None
        :rtype: None
        """
        Shell.mkdir(directory)

    def get_thumbnail(self, url, prefix="thumb-", directory="./cache"):
        """
        Downlods the thumbnil correspondig to the image specified by the url.
        The prefix is put before the original filename so that a thumbnail can
        easily be used in the cache.

        This function is not yet used

        :param url: The url of the image
        :type url: string
        :param prefix: the prefix of the filename to be added. default is "thumb-"
        :type prefix: string
        :param directory: The directory where to store the thumbnail
        :type directory: string
        :return: None
        :rtype: None
        """
        self.create_cache(directory)
        url = url.split("?kind=thumbnail")[0]
        name = os.path.basename(url)
        thumb = url + "?kind=thumbnail"

        if self.debug:
            print("FETCH:", thumb)
        response = requests.get(thumb, stream=True)
        filename = f"{directory}/{prefix}{name}"

        with open(filename, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        return response

    def get_deviceinformation(self):
        """
        Returns the device information as a json object

        :return: device information
        :rtype: dict
        """
        r = self._get(path="/deviceinformation").json()
        return r

    def get_temperature(self):
        """
        Returns the temperature information as a json object

        :return: temperature information
        :rtype: string
        """
        r = self._get(path="/devicestatus/temperature").json()
        return r

    @property
    def temperature(self):
        return self.get_temperature()["status"]

    def get_datetime(self):
        """
        Returns the datetime information as a json object

        :return: datetime information
        :rtype: string
        """
        r = self._get(path="/functions/datetime").json()
        return r

    @property
    def datetime(self):
        d = self.get_datetime()["datetime"][0:-6]
        datetime_object = datetime.strptime(d, "%a, %d %b %Y %H:%M:%S")
        return datetime_object

    @property
    def copyright(self):
        """
        Gets the copyright information

        :return: the copyright information
        :rtype: string
        """
        r = self._get(path="/functions/registeredname/copyright").json()['copyright']
        return r

    @copyright.setter
    def copyright(self, _copyright):
        """
        Sets the copyright information

        :param _copyright: The copyright information
        :type _copyright: string
        :return: None
        :rtype: None
        """
        """
        Sets the copyright information

        :return: copyright information
        :rtype: string
        """
        r = self._put(path="/functions/registeredname/copyright",
                      json={"copyright": _copyright})
        return r

    @property
    def author(self):
        """
        Gets the authors information

        :return: Author name as set in the camera
        :rtype: string
        """
        r = self._get(path="/functions/registeredname/author").json()['author']
        return r

    @author.setter
    def author(self, _author):
        """
         Sets the authors information on the acamera

        :param _author: The author information
        :type _author: string
        :return: None
        :rtype: None
        """
        r = self._put(path="/functions/registeredname/author",
                      json={"author": _author})
        return r

    @property
    def owner(self):
        """
        Gets the owners information

        :return: The owner information as set on the camera
        :rtype: string
        """
        r = self._get(path="/functions/registeredname/ownername").json()['ownername']
        return r

    @owner.setter
    def owner(self, _owner):
        """
        Gets the owners information from the camera

        :param _owner: Owners information as set on the camera
        :type _owner: string
        :return: None
        :rtype: None
        """
        r = self._put(path="/functions/registeredname/ownername",
                      json={"ownername": _owner})
        return r

    @property
    def nickname(self):
        """
        The nickname used for Wifi connections

        :return: The nickname
        :rtype: string
        """
        r = self._get(path="/functions/registeredname/nickname").json()['nickname']
        return r

    @nickname.setter
    def nickname(self, _nickname):
        """
        Sets the nickname of the camera as used in wifi settings

        :param _nickname: The nickname
        :type _nickname: string
        :return: None
        :rtype: None
        """
        r = self._put(path="/functions/registeredname/nickname",
                      json={"nickname": _nickname})
        return r

    def _get(self, path=None):
        """
        Executes a GET on the path. The path is automatically resolved
        to the highest available CCAPI version.

        :param path: The path
        :type path: string
        :return: response from the server
        :rtype:
        """
        resolved = self._resolve_path(path)
        url = f'http://{self.ip}:{self.port}{resolved}'
        if self.debug:
            print(f"GET: {url}")
        r = requests.get(url)
        return r

    def _put(self, path=None, json=None):
        """
        Executes a PUT on the path. The path is automatically resolved
        to the highest available CCAPI version.

        :param path: The path
        :type path: string

        :param json:
        :type json:
        :return: response from the server
        :rtype:
        """
        resolved = self._resolve_path(path)
        url = f'http://{self.ip}:{self.port}{resolved}'
        if self.debug:
            print(f"PUT: {url} <- {json}")
        r = requests.put(url, json=json)
        return r

    def _post(self, path=None, json=None):
        """
        Executes a POST on the path. The path is automatically resolved
        to the highest available CCAPI version.

        :param path:
        :type path:
        :param json:
        :type json:
        :return:
        :rtype:
        """
        resolved = self._resolve_path(path)
        url = f'http://{self.ip}:{self.port}{resolved}'
        if self.debug:
            print(f"POST: {url} <- {json}")
        r = requests.post(url, json=json)
        return r

    def ccapi(self, output="table"):
        """

        :param output:
        :type output:
        :return:
        :rtype:
        """
        if output == "df":
            r = self.ccapi(output="csv")
            r = df_from_csv(r)
        else:
            result = []
            r = self._get(path="/ccapi")
            d = r.json()
            if output == "native":
                r = d
            else:
                for version in d.keys():
                    api = d[version]
                    for entry in api:
                        entry["version"] = version
                        result.append(entry)
                    r = Printer.write(result,
                                      max_width=128,
                                      order=["version",
                                             "path",
                                             "get",
                                             "put",
                                             "post",
                                             "delete"],
                                      output=output)
        return r

    def get_lens(self):
        """
        Gets the lens information

        :return:
        :rtype:
        """
        r = self._get(path="/devicestatus/lens").json()
        return r

    def get_storage(self):
        """
        Gets information about the storage

        :return:
        :rtype:
        """
        r = self._get(path="/devicestatus/storage").json()["storagelist"]
        for entry in r:
            entry["maxsize"] = humanize.naturalsize(entry["maxsize"])
            entry["spacesize"] = humanize.naturalsize(entry["spacesize"])
            entry["name"] = entry["name"].replace("card", "Card ")
            # entry["path"] = os.path.basename(entry["path"])
            #    .replace("card", "Card ")
        return r

    @property
    def charge(self):
        """
        The chare level as integer (0-100)
        :return:
        :rtype:
        """
        b = self.get_battery(output="json")["level"]
        return int(b)

    def get_battery(self, output="table"):
        """
        Gets information about the battery

        :param output:
        :type output:
        :return:
        :rtype:
        """
        # The battery call needs to be done first to get accurate level information
        no_percent = self._get(path="/devicestatus/battery")
        result = self._get(path="/devicestatus/batterylist")
        r = result.json()["batterylist"][0]
        if output in ["json", None]:
            return r
        else:
            r = Printer.attribute(r, output=output)
        return r

    def get_settings(self, refresh=True):
        """
        Gets the current settings from the camera.
        Dynamically queries all discovered CCAPI versions for their
        shooting settings.

        :return:
        :rtype:
        """
        if not refresh and os.path.exists(self.settings_file):
            with open(self.settings_file, "r") as file:
                c = json.load(file)
        else:
            # Dynamically query settings from all discovered versions
            self.settings = {}
            versions_to_query = self._api_versions if self._api_versions else ["ver100", "ver110"]
            for ver in versions_to_query:
                try:
                    settings_data = self._get(path=f"/ccapi/{ver}/shooting/settings").json()
                    if settings_data:  # only add if the camera returned data
                        self.settings[ver] = settings_data
                except Exception:
                    pass  # version may not have shooting/settings
            # Ensure at least empty dicts so downstream code doesn't break
            if not self.settings:
                self.settings = {"ver100": {}}

            for version in self.settings:
                for key in self.settings[version]:
                    entry = self.settings[version][key]
                    #
                    # the kind settings does not yet work for all as some have
                    # multiple choices and sliders
                    #
                    # an example is still imagequality
                    # which can have a value for raw and jpeg
                    # and not just a single value. Also all picture
                    # quality apis need to be handeled differentl
                    try:
                        # if len(self.settings[version][key]["value"]) > 1:
                        #    self.settings[version][key]["kind"] = "list"
                        if key in ["stillimagequality", "wbshift"] or "picturestyle" in key:
                            self.settings[version][key]["kind"] = "unkown"
                        elif "min" in entry["ability"]:
                            self.settings[version][key]["kind"] = "slider"
                        else:
                            self.settings[version][key]["kind"] = "choice"
                    except:
                        pass
                    api = "/shooting/settings/" + key.replace("_", "/")
                    self.settings[version][key]["api"] = api
            with open(self.settings_file, "w") as file:
                json.dump(self.settings, indent=4, fp=file)
        return self.settings

    def contents(self):
        """
        List the contents of all cards

        :return:
        :rtype:
        """

        cards = self._get(path="/contents").json()["path"]
        images = []
        for path in cards:
            directories = self._get(path=path).json()["path"]
            for d in directories:
                pages = int(self._get(path=d + "?kind=number").json()["pagenumber"])
                for page in range(1, pages + 1):
                    files = self._get(path=d + f"?page={page}").json()["path"]
                    for f in files:
                        images.append(f)
        return images

        # r = result.json()["batterylist"][0]
        # r = Printer.attribute(r, output=output)
        # return r.json()

    @property
    def exposuresmoothing(self):
        """

        :return:
        :rtype:
        """
        r = self._get(path="/shooting/settings/focusbracketing/exposuresmoothing").json()
        return r

    @exposuresmoothing.setter
    def exposuresmoothing(self, on):
        value = self._get_enable(on)
        r = self._put(path="/shooting/settings/focusbracketing/exposuresmoothing",
                      json={"value": value})
        return r

    def get_numberofshots(self):
        r = self._get(path="/shooting/settings/focusbracketing/numberofshots").json()
        return r

    @property
    def numberofshots(self):
        r = self.get_numberofshots()
        return r

    @numberofshots.setter
    def numberofshots(self, n):
        ability = self.get_numberofshots()["ability"]
        if ability["min"] <= n <= ability["max"]:
            r = self._put(path="/shooting/settings/focusbracketing/numberofshots",
                          json={"value": n})
        return r

    def get_focusincrement(self):
        r = self._get(path="/shooting/settings/focusbracketing/focusincrement").json()
        return r

    @property
    def focusincrement(self):
        r = self.get_focusincrement()
        return r

    @focusincrement.setter
    def focusincrement(self, n):
        ability = self.get_focusincrement()["ability"]
        if ability["min"] <= n <= ability["max"]:
            r = self._put(path="/shooting/settings/focusbracketing/focusincrement",
                          json={"value": n})
        return r

    def _get_enable(self, on):
        value = False
        if on in ["1", "on", True, "enable"]:
            value = "enable"
        else:
            value = "disable"
        return value

    def _get_bool(self, on):
        value = False
        if on in ["1", "on", True, "true", "True"]:
            value = True
        else:
            value = False
        return value

    @property
    def focusbracketing(self):
        r = self._get(path="/shooting/settings/focusbracketing").json()
        return r

    @focusbracketing.setter
    def focusbracketing(self, on):
        value = self._get_enable(on)
        r = self._put(path="/shooting/settings/focusbracketing",
                      json={"value": value})
        return r

    def liveview(self, display="on", size="medium"):
        if display not in ["on", "off", "keep"]:
            print("error setting display")
            return None
        if size not in ["small", "off", "medium"]:
            print("error setting size")
            return None

        r = self._post(path="/shooting/liveview",
                       json={"cameradisplay": display,
                             "liveviewsize": size
                             })

        return r

    cam = "/shooting/liveview/rtp"

    def cam_start(self):
        r = self._post(path=self.cam,
                       json={"action": "start",
                             "ipaddress": "rtsp://192.168.50.10:8554/camera"
                             })

    def cam_stop(self):
        r = self._post(path=self.cam,
                       json={"action": "stop",
                             "ipaddress": "rtsp://192.168.50.10:8554/camera"
                             })

    def cam_view(self, location=None):

        frame_window = location.image([])
        take_picture_button = location.button('Take Picture from Video')

        self.cam_start()
        url = f"http://192.168.50.210:8080{self.cam}"
        # while True:
        #     # Request the image from the server
        #     response = requests.get(url=url)
        #     imgNp = np.array(bytearray(response.content), dtype=np.uint8)
        #     frame = cv2.imdecode(imgNp, cv2.IMREAD_UNCHANGED)
        #     # As OpenCV decodes images in BGR format, we'd convert it to the RGB format
        #     # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #     try:
        #         frame_window.image(frame)
        #     except Exception as e:
        #         print (e)
        #         pass

        # vid = cv2.VideoCapture(url)  # For streaming links
        # while True:
        #     _, frame = vid.read()
        #     print(frame)
        #     cv2.imshow('Video Live IP cam', frame)
        #     key = cv2.waitKey(1) & 0xFF
        #     if key == ord('q'):
        #         break
        #
        # vid.release()
        # cv2.destroyAllWindows()
        # vid = cv2.VideoCapture("rtp://192.168.50.210:8081")
        # while True:
        #     rdy, frame = vid.read()
        #     print(rdy)
        #     try:
        #         cv2.imshow('Video Live IP cam', frame)
        #         key = cv2.waitKey(1) & 0xFF
        #         if key == ord('q'):
        #             break
        #     except:
        #         pass
        #
        # vid.release()
        # cv2.destroyAllWindows()
        #
        # self.cam_stop()

        pass

    def load(self, image, download=False, refresh=False):
        from IPython.display import Image
        if download:
            self.download(image, refresh=refresh)
        name = os.path.basename(image)
        return Image(filename=name)

    def display(self, image, download=False, refresh=False):
        i = self.load(image, download=download, refresh=refresh)
        return i

    def download(self, image, refresh=False):
        """
        Downloads an image, if itis not already downloaded

        :param image:
        :type image:
        :return:
        :rtype:
        """
        url = f"http://{self.ip}:{self.port}{image}"

        name = os.path.basename(url)
        if self.debug:
            print("Downloading:", url)

        response = requests.get(url, stream=True)

        if not os.path.isfile(name) or refresh:
            with open(name, 'wb') as out_file:
                shutil.copyfileobj(response.raw, out_file)
        return response

    def get_liveview_image(self, name):
        """
        Downloads he liveview image and puts it in the file with the given names
        :param name:
        :type name:
        :return:
        :rtype:
        """

        url = f"http://{self.ip}:{self.port}{self._resolve_path('/shooting/liveview/flipdetail')}?kind=image"
        url = f"http://{self.ip}:{self.port}{self._resolve_path('/shooting/liveview/flip')}"

        response = requests.get(url, stream=True)
        with open(name, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        return response

    def preview(self, name):
        if os_is_mac():
            os.system(f"open /Applications/Google\ Chrome.app {name}")
        elif os_is_linux():
            os.system(f"open {name}")
        else:
            print("previe for this OS not yet supported")

    # def get_cardformat(self):
    #     # only supported for PowerShot cameras
    #     r = self._get(path="/ccapi/ver100/functions/cardformat")
    #     return r

    def get_zoom(self):
        # only supported for PowerShot cameras
        r = self._get(path="/shooting/control/zoom")
        return r

    def get_shootingmodedial(self):
        # only supported for PowerShot cameras
        r = self._get(path="/shooting/settings/shootingmodedial").json()
        return r

    @property
    def shootingmodedial(self):
        r = self.get_settings_value(key="shootingmodedial")["value"]
        return r

    @shootingmodedial.setter
    def shootingmodedial(self, value):
        r = self.set_settings_value(key="shootingmodedial", value=value)
        return r

    def set_value(self, path=None, value=None):
        # old = self.settings[version][key]["value"]
        r = self._get(path=path).json()
        ability = r["ability"]
        if value in ability:
            result = value
            r = self._put(path=path,
                          json={"value": value})
        else:
            print("failed")
            result = old

        return result

    def get_autopoweroff(self):
        """
        The chare level as integer (0-100)
        :return:
        :rtype:
        """
        r = self._get(path="/functions/autopoweroff").json()
        return r

    @property
    def autopoweroff(self):
        r = self.get_autopoweroff()["value"]
        return r

    @autopoweroff.setter
    def autopoweroff(self, value):
        r = self.set_value(path="/functions/autopoweroff",
                           value=value)
        return r

    def autofocus(self, on):
        action = self._get_bool(on)
        if action:
            action = "start"
        else:
            action = "stop"
        r = self._post(path="/shooting/control/af",
                       json={
                           "action": action
                       })
        return r

    def flickerdetection(self, on):
        action = self._get_bool(on)
        if action:
            action = "start"
        else:
            action = "stop"
        r = self._post(path="/shooting/control/flickerdetection",
                       json={
                           "action": action
                       })
        return r

    def shoot(self, af=True):
        af = self._get_bool(af)
        r = self._post(path="/shooting/control/shutterbutton",
                       json={
                           "af": af
                       })
        return r

    def shoot_control(self, af=True, action="full_press"):
        af = self._get_bool(af)
        if action not in ["full_press", "half_press", "release"]:
            return
        r = self._post(path="/shooting/control/shutterbutton/manual",
                       json={
                           "af": af,
                           "action": action
                       })
        return r

    def focus(self):
        r = camera.shoot_control(af=True, action="half_press")
        return r

    def release(self):
        r = camera.shoot_control(af=False, action="release")
        return r

    def set_settings_value(self, key=None, value=None):
        version = self.get_settings_version(key=key)
        old = self.settings[version][key]["value"]
        ability = self.settings[version][key]["ability"]
        if value in ability:
            result = value
            r = self._put(path=f"/ccapi/{version}/shooting/settings/{key}",
                          json={"value": value})
        else:
            print("failed")
            result = old

        return result

    def get_settings_version(self, key=None):
        """
        Determines which CCAPI version contains a given settings key.
        Searches all discovered versions, preferring the highest version
        that has the key (to leverage the latest API).
        """
        # Search from highest version to lowest; return first match
        for ver in sorted(self.settings.keys(), reverse=True):
            if key in self.settings[ver]:
                return ver
        # Fallback to first available version
        return sorted(self.settings.keys())[0] if self.settings else "ver100"

    def get_settings_value(self, key=None):
        version = self.get_settings_version(key=key)
        value = self._get(path=f"/ccapi/{version}/shooting/settings/{key}").json()
        return value

    @property
    def beep(self):
        key = "beep"
        r = self._get(path=f"/functions/{key}").json()["value"]
        return r

    # @beep.setter
    # def beep(self, value):
    #     r = self.set_settings_value(key="beep", value=value)
    #     return r

    @property
    def av(self):
        r = self.get_settings_value(key="av")["value"]
        return r

    @av.setter
    def av(self, value):
        r = self.set_settings_value(key="av", value=value)
        return r

    @property
    def tv(self):
        r = self.get_settings_value(key="tv")["value"]
        return r

    @tv.setter
    def tv(self, value):
        r = self.set_settings_value(key="tv", value=value)
        return r

    @property
    def iso(self):
        r = self.get_settings_value(key="iso")["value"]
        return r

    @iso.setter
    def iso(self, value):
        r = self.set_settings_value(key="iso", value=value)
        return r

    @property
    def wb(self):
        r = self.get_settings_value(key="wb")["value"]
        return r

    @wb.setter
    def wb(self, value):
        r = self.set_settings_value(key="wb", value=value)
        return r

    @property
    def afmethod(self):
        r = self.get_settings_value(key="afmethod")["value"]
        return r

    @afmethod.setter
    def afmethod(self, value):
        r = self.set_settings_value(key="afmethod", value=value)
        return r

    @property
    def drive(self):
        r = self.get_settings_value(key="drive")["value"]
        return r

    @drive.setter
    def drive(self, value):
        r = self.set_settings_value(key="drive", value=value)
        return r

    @property
    def aeb(self):
        r = self.get_settings_value(key="aeb")["value"]
        return r

    @aeb.setter
    def aeb(self, value):
        r = self.set_settings_value(key="aeb", value=value)
        return r

    @property
    def colortemperature(self):
        r = self.get_settings_value(key="colortemperature")["value"]
        return r

    @colortemperature.setter
    def colortemperature(self, value):
        r = self.set_settings_value(key="colortemperature", value=value)
        return r

    @property
    def colorspace(self):
        r = self.get_settings_value(key="colorspace")["value"]
        return r

    @colorspace.setter
    def colorspace(self, value):
        r = self.set_settings_value(key="colorspace", value=value)
        return r

    @property
    def metering(self):
        r = self.get_settings_value(key="metering")["value"]
        return r

    @metering.setter
    def metering(self, value):
        r = self.set_settings_value(key="metering", value=value)
        return r

    @property
    def stillimageaspectratio(self):
        r = self.get_settings_value(key="stillimageaspectratio")["value"]
        return r

    @stillimageaspectratio.setter
    def stillimageaspectratio(self, value):
        r = self.set_settings_value(key="stillimageaspectratio", value=value)
        return r

    @property
    def afoperation(self):
        r = self.get_settings_value(key="afoperation")["value"]
        return r

    @afoperation.setter
    def afoperation(self, value):
        r = self.set_settings_value(key="afoperation", value=value)
        return r

    @property
    def shuttermode(self):
        r = self.get_settings_value(key="shuttermode")["value"]
        return r

    @shuttermode.setter
    def shuttermode(self, value):
        r = self.set_settings_value(key="shuttermode", value=value)
        return r


def beep():
    sys.stdout.write('\a')


if __name__ == '__main__':

    camera = CCAPI()
    r = camera.ccapi(output="github")
    print(r)
    # r = camera.ccapi(output="json")
    # print(r)

    # camera.author = "Gregor von Laszewski"

    # author = camera.author
    # owner = camera.owner
    # copyright = camera.copyright
    # nickname = camera.nickname
    #
    # print(f"Author: {author}")
    # print(f"Owner: {owner}")
    # print(f"Copyright: {copyright}")
    # print(f"Nickname: {nickname}")

    # r = camera.ccapi(output="table")
    # print (r)
    #
    # r = camera.get_battery(output="json")
    # print(r)
    # r = camera.get_battery()
    # print (r)
    #
    # r = camera.settings(output="table")
    # pprint (r)
    #
    # # r = camera.contents()
    # # pprint(r)
    #
    # # r = camera.contents()
    # # pprint(r)
    #
    # macro = Focus(camera)
    # r = macro.focusbracketing()
    # print (r)
    #
    # r = macro.focusbracketing("enable")
    # # print (r)
    #
    # r = macro.focusbracketing()
    # print (r)
    #
    # r = camera.get_battery()
    # print (r)
    #
    # #print (r)
    #
    beep()

    print("SUPER")

    # print(camera.author)
    # print(camera.nickname)
    # print(camera.owner)
    # print(camera.copyright)

    print(camera.charge)

    # print (camera.focusbracketing)
    # camera.focusbracketing = True
    # print (camera.focusbracketing)

    # camera.exposuresmoothing = True
    # print (camera.exposuresmoothing)
    #
    # print(camera.numberofshots)
    #
    # camera.numberofshots = 13
    #
    # print(camera.numberofshots)
    #
    # camera.focusincrement = 4
    # print (camera.focusincrement)

    # print(camera.get_storage())

    # pprint(camera.get_settings())

    camera.release()
    r = camera.liveview(display="on", size="medium")
    print(r.text)

    for i in range(3):
        name = f"img{i}.jpeg"
        r = camera.get_liveview_image(name)
        camera.preview(name)
        print(r)
        print(r.headers)

    # r = camera.shoot(af=1)
    # r = camera.shoot_control(af=True, action="full_press")
    # r = camera.shoot_control(af=True, action="release")
    # r = camera.shoot_control(af=True, action="half_press")
    # r = camera.focus()

    # url = "https://192.xxx.xxx.xxx:8080/ccapi/ver100/shooting/liveview/flip"

    # r = requests.get(url)

    # print(dir(r))
    # print(r)
    #
    # print(r.text)
    # print(r.content)
    # print(r.reason)
    # print(r.raw)
    # print (r._content)
