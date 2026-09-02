import urllib.request
import json
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

url = "https://commons.wikimedia.org/w/api.php?action=query&titles=File:Early_blight_on_tomato_leaves_(7871930010).jpg&prop=imageinfo&iiprop=url&format=json"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
response = urllib.request.urlopen(req)
data = json.loads(response.read())
pages = data['query']['pages']
page_id = list(pages.keys())[0]
image_url = pages[page_id]['imageinfo'][0]['url']
print(f"Downloading test image from Wikimedia Commons: {image_url}")
urllib.request.urlretrieve(image_url, "internet_test.jpg")
print("Saved as internet_test.jpg")
