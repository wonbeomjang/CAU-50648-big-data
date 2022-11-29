from urllib import request


def download_image(urls, image_path):
    try:
        request.urlretrieve(urls, image_path)
    except:
        pass

