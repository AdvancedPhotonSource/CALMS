import re
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

url = 'https://www.aps.anl.gov/APS-Science-Highlight/'
domain = urlparse(url).netloc


def get_news_links(page):
    news = page.select_one('div.view-news-feed > div.view-content')
    return news.find_all('a')


def save_html_text(link):
    fname = '_'.join(link.split('/')[2:])
    fpath = Path(f"APS-Science-Highlight/{fname}.txt")
    if fpath.exists():
        return
    response = requests.get(f'http://{domain}/{link}')
    page = BeautifulSoup(response.content, 'lxml')
    content = page.select_one('div.region.region-content')
    # --------
    title = content.select_one('h1.page-header')
    article = content.select_one('article')
    # --------
    fpath.parent.mkdir(exist_ok=True)

    def to_text(html):
        text = html.get_text()
        return re.sub(r'(\n\s?)+', r'\n\n', text).strip()

    with open(fpath, 'w') as f:
        f.write(to_text(title))
        f.write('\n\n')
        f.write(to_text(article))


# -------- start scraping ----------

response = requests.get(url)
index_page = BeautifulSoup(response.content, 'lxml')

# get years from the side menu
side_menu = index_page.select_one('ul.menu--menu-science-highlights')
links = side_menu.find_all('a', href=re.compile(r'[12][90]\d\d$'))

# go through all years
for link in links:

    year_url = link['href']
    url = f"https://{domain}/{year_url}"
    print(url)

    response = requests.get(url)
    one_year = BeautifulSoup(response.content, 'lxml')

    # go through all pages of a year
    while True:

        # save articles to text file
        articles = get_news_links(one_year)
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
        url = f'https://{domain}/{year_url}{page_url}'
        print(f'  {page_url}')
        response = requests.get(url)
        one_year = BeautifulSoup(response.content, 'lxml')
