#! /usr/bin/env python

import sys
import urllib.request
from bs4 import BeautifulSoup
import spacy

nlp = spacy.load('en')

def parse_index(index):
    index = scrape_recipe_index(index)

    print(f"Number of recipe urls in index: {len(index)}")

    for recipe_url in index:
        parse_recipe(recipe_url)

def scrape_recipe_index(url):
    soup = url_to_soup(url)
    index = soup.find('div', class_="results-group")
    return soup_to_index(index)

def soup_to_index(soup):
    u = []

    urls = soup.select('.recipe-content-card a[itemprop="url"]')
    for url in urls:
        u.append(url.attrs['href'])

    return u

def parse_recipe(url):
    base_url = "https://www.epicurious.com"
    outfile = base_url.rsplit('.')[1]

    recipe = scrape_recipe(f"{base_url}{url}")

    desc_docs = split_into_sentences(recipe['description'])
    instruct_docs = split_into_sentences(recipe['instructions'])

    with open(f"{outfile}-pos.txt", 'a', encoding='utf-8') as f:
        for doc in instruct_docs:
            for sent in doc.sents:
                f.write(sent.text)
                f.write('\n')

    with open(f"{outfile}-neg.txt", 'a', encoding='utf-8') as f:
        for doc in desc_docs:
            for sent in doc.sents:
                f.write(sent.text)
                f.write('\n')

def scrape_recipe(url):
    print(f"Scraping {url}...")

    soup = url_to_soup(url)
    recipe = soup.find('div', class_="main")
    return soup_to_recipe(recipe)

def soup_to_recipe(soup):
    d = {}

    desc_select = soup.select('div[itemprop="description"]')
    inst_select = soup.select('div[itemprop="recipeInstructions"] .preparation-groups')

    d['description'] = desc_select[0].text if len(desc_select) > 0 else ''
    d['instructions'] = inst_select[0].text if len(inst_select) > 0 else ''

    return d

def url_to_soup(url):
    try:
        page = urllib.request.urlopen(url)
    except Exception as e:
        print(f"Failed to fetch {url}")
        if verbose:
            print(e)
        sys.exit()
    try:
        soup = BeautifulSoup(page, 'lxml')
    except Exception as e:
        print(f"Failed to parse {url}")
        if verbose:
            print(e)
        sys.exit()

    return soup

def split_into_sentences(text):
    """ Split by newlines, strip whitespace, and remove empty strings """
    lines = list(filter(None, [l.strip() for l in text.split('\n')]))

    return [nlp(line) for line in lines]

if __name__ == '__main__':
    index = "https://www.epicurious.com/search?content=recipe&page="
    num_pages = 10

    for p in range(1,num_pages+1):
        parse_index(f"{index}{p}")
