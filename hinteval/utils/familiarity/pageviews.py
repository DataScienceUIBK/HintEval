import requests
from requests.utils import quote
from datetime import date, datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

endpoints = {
    'article': 'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article',
    'project': 'https://wikimedia.org/api/rest_v1/metrics/pageviews/aggregate',
    'top': 'https://wikimedia.org/api/rest_v1/metrics/pageviews/top',
    'top-by-country': 'https://wikimedia.org/api/rest_v1/metrics/pageviews/top-by-country',
    'top-per-country': 'https://wikimedia.org/api/rest_v1/metrics/pageviews/top-per-country'
}


def parse_date(stringDate):
    return datetime.strptime(stringDate.ljust(10, '0'), '%Y%m%d%H')


def format_date(d):
    return datetime.strftime(d, '%Y%m%d%H')


def timestamps_between(start, end, increment):
    # convert both start and end to datetime just in case either are dates
    start = datetime(start.year, start.month, start.day, getattr(start, 'hour', 0))
    end = datetime(end.year, end.month, end.day, getattr(end, 'hour', 0))

    while start <= end:
        yield start
        start += increment


def month_from_day(dt):
    return datetime(dt.year, dt.month, 1)


class PageviewsClient:
    def __init__(self, user_agent, parallelism=10):
        self.headers = {"User-Agent": user_agent}
        self.parallelism = parallelism

    def article_views(
            self, project, articles,
            access='all-access', agent='all-agents', granularity='daily', start=None, end=None):

        endDate = end or date.today()
        if type(endDate) is not date:
            endDate = parse_date(end)

        startDate = start or endDate - timedelta(30)
        if type(startDate) is not date:
            startDate = parse_date(start)

        # If the user passes in a string as "articles", convert to a list
        if type(articles) is str:
            articles = [articles]

        articles = [a.replace(' ', '_') for a in articles]
        articlesSafe = [quote(a, safe='') for a in articles]

        urls = [
            '/'.join([
                endpoints['article'], project, access, agent, a, granularity,
                format_date(startDate), format_date(endDate),
            ])
            for a in articlesSafe
        ]

        outputDays = timestamps_between(startDate, endDate, timedelta(days=1))
        if granularity == 'monthly':
            outputDays = list(set([month_from_day(day) for day in outputDays]))
        output = defaultdict(dict, {
            day: {a: None for a in articles} for day in outputDays
        })

        try:
            results = self.get_concurrent(urls)
            some_data_returned = False
            for result in results:
                if 'items' in result:
                    some_data_returned = True
                else:
                    continue
                for item in result['items']:
                    output[parse_date(item['timestamp'])][item['article']] = item['views']
            if not some_data_returned:
                raise Exception()

            return output
        except:
            raise Exception()

    def get_concurrent(self, urls):
        with ThreadPoolExecutor(self.parallelism) as executor:
            f = lambda url: requests.get(url, headers=self.headers).json()
            return list(executor.map(f, urls))
