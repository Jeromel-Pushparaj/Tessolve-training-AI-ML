#pip install bs4
import requests
from bs4 import BeautifulSoup
#response = requests.get("https://en.wikipedia.org/wiki/Web_scraping")
response = requests.get("https://en.wikipedia.org/wiki/Natural_language_processing")
bs = BeautifulSoup(response.text,"lxml")
print(bs.find("p").text)
# https://en.wikipedia.org/wiki
#/Natural_language_processing
