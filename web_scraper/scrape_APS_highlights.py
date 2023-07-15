import re
from pathlib import Path
from urllib.parse import urljoin

import requests
import requests_cache
from bs4 import BeautifulSoup

requests_cache.install_cache(
    cache_name='scraper_cache', backend='sqlite', expire_after=86400
)


url = 'https://www.aps.anl.gov/APS-Science-Highlight/'
baseurl = urljoin(url, '..')

savedir = 'DOC_STORE/APS-Science-Highlight'
Path(f'./{savedir}').mkdir(exist_ok=True)


# -------- start scraping ----------


def download(link):
    print(f'scraping {link}')
    response = requests.get(link)
    page = BeautifulSoup(response.content, 'lxml')
    content = page.select_one('div.region.region-content')
    return response, page, content


response, index_page, _ = download(url)
if response.from_cache:
    print('[ WARNING: page requested from cache ]\n')


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
        (1, '\nSee: '),
    ]:
        cut = text.find(s)
        if cut > 0:
            lastline = text[cut:].lstrip().split('\n', 1)[0]
            text = text[:cut].rstrip() + keep * f'\n\n{lastline}'
            break
    return text


def save_html_text(link):
    fname = '_'.join(link.split('/')[2:])
    fpath = Path(f"{savedir}/{fname}.txt")
    if fpath.exists():
        return
    url = urljoin(baseurl, link)
    response, page, content = download(url)
    # --------
    title = content.select_one('h1.page-header')
    article = content.select_one('article')
    # --------

    def to_text(html):
        text = html.get_text()
        return re.sub(r'(\n\s?)+', r'\n\n', text).strip()

    with open(fpath, 'w') as f:
        f.write(to_text(title))
        f.write('\n\n')
        f.write(strip_acknowledgement(to_text(article)))


# get years from the side menu
side_menu = index_page.select_one('ul.menu--menu-science-highlights')
links = side_menu.find_all('a', href=re.compile(r'[12][90]\d\d$'))

# go through all years
for link in links:

    year_url = link['href']
    url = urljoin(baseurl, year_url)

    response, one_year, _ = download(url)

    # go through all pages of a year
    while True:

        # save articles to text file
        news = one_year.select_one('div.view-news-feed > div.view-content')
        articles = news.find_all('a')
        for article in articles:
            save_html_text(article['href'])

        # is there a next page?
        page_menu = one_year.find('ul', class_='pagination')
        if page_menu is None:  # only one page
            break
        last_page = page_menu.find('a', rel='last')
        if last_page is None:  # on last page
            break

        # go to the next page
        current_page = page_menu.select_one('li.pager__item.is-active')
        next_page = current_page.find_next_sibling('li')
        page_url = next_page.find('a')['href']
        url = urljoin(baseurl, f'{year_url}{page_url}')
        response, one_year, _ = download(url)
