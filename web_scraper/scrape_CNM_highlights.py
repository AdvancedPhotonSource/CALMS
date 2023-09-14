import re
from pathlib import Path
from urllib.parse import urljoin
import time
import requests
import requests_cache
from bs4 import BeautifulSoup
import random

requests_cache.install_cache(
    cache_name='scraper_cache', backend='sqlite', expire_after=86400
)


url = 'https://www.anl.gov/cnm/cnm-research-highlights'
baseurl = urljoin(url, '..')
print('baseurl', baseurl)
savedir = 'DOC_STORE/CNM-Science-Highlight'
Path(f'./{savedir}').mkdir(exist_ok=True)


# -------- start scraping ----------


def download(link):
    print(f'scraping {link}')
    response = requests.get(link)
    page = BeautifulSoup(response.content, 'html.parser')
    content = page.find('main', class_='l-main--grid')
    return response, page, content


def strip_cnm_description(text, substring):
    index = text.find(substring)
    if index != -1:  # Only strip if the substring exists
        return text[:index]
    else:
        return text  # If substring is not found, return the original text


def strip_acknowledgement(text):
    for keep, s in [
        (0, '\nThis work was supported by '),
        (0, '\nThis research was supported by '),
        (0, '\nThis work was funded by '),
        (0, '\nWe are grateful for support from '),
        (0, '\nThis work is supported by '),
        (0, '\nFunding for this project was provided in part by '),
        (0, '\nThis work was financially supported by '),
        (1, '\nCorrespondence: '),
        (1, '\nAuthor affiliations: '),
        (0, '\nDownload this highlight '),  
        (0, '\nAbout Argonne’s Center for Nanoscale Materials '),  
        (1, '\nSee: '),
    ]:
        cut = text.find(s)
        if cut > 0:
            lastline = text[cut:].lstrip().split('\n', 1)[0]
            text = text[:cut].rstrip() + keep * f'\n\n{lastline}'
            # break
    return text


def save_html_text(link):
    fname = '_'.join(link.split('/')[2:])
    fpath = Path(f"{savedir}/{fname}.txt")
    if fpath.exists():
        return
    url = urljoin(baseurl, link)
    response, page, content = download(url)

    # --------
    title = page.find('h1', class_='basic-header__title')
    article = content
    # --------

    def to_text(html):
        text = html.text
        return re.sub(r'(\n\s?)+', r'\n\n', text).strip()

    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(to_text(title))
        f.write('\n\n')
        # f.write(content)
        
        # Using the function to strip everything after the specific substring
        substring = "Download this highlight"
        stripped_text =strip_cnm_description(to_text(article), substring)
        f.write(strip_acknowledgement(stripped_text))#strip_acknowledgement(to_text(article)))


def get_soup_from_request(url: str) -> BeautifulSoup:
        """Get a BeautifulSoup parse tree (lxml parser) from a url request
        Args:
            url: A requested url
        Returns:
            A BeautifulSoup parse tree.
        """
        headers = {
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;"
                "q=0.9,image/webp,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.5",
            "Upgrade-Insecure-Requests": "1",
            "User-Agent": (
                "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:82.0)"
                " Gecko/20100101 Firefox/82.0"
            ),
        }
        wait_time = float(random.randint(0, 50))
        time.sleep(wait_time / float(10))
        with requests.Session() as session:
            r = session.get(url, headers=headers)
        soup = BeautifulSoup(r.text, "html")
        return soup


response, one_page, _ = download(url)

page_menu = one_page.find('ul',class_='pager__items js-pager__items')
current_page = page_menu.select_one('li.pager__item.is-active')
current_page_num = current_page.find('a')['href'].split('=')[-1]

while True:

    news =  one_page.find_all("a")
    articles = news

    for article in articles:
        if article['href'].startswith("/cnm/article"):
            #  print( baseurl + article['href'])
             save_html_text(article['href'])

    # is there a next page?
    
    if page_menu is None:  # only one page
        break
    last_page = page_menu.find('li',class_='pager__item pager__item--last')
    # print('last_page', last_page.find('a')['href'])
    if last_page is None:  # on last page
        break

    new_url = url + "?page=" + str(current_page_num)
    print('new_url', new_url)

    response, one_page, _ = download(new_url)
    current_page_num = int(current_page_num) + 1