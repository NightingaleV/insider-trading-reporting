from abc import ABC, abstractmethod

class FinancialNewsFeed(ABC):
    @abstractmethod
    def __init__(self):
        self.news = []

    def add_news(self, title, content):
        self.news.append({"title": title, "content": content})

    @abstractmethod
    def get_news(self):
        return self.news
    
    
class ForbesNewsFeed(FinancialNewsFeed):
    def __init__(self):
        super().__init__()
        
    def _get_ticker_url(self, ticker):
        return f"https://www.forbes.com/search/?q={ticker}"
    
    
    def get_news(self):
        # Here you would implement the logic to fetch news from Forbes
        return super().get_news()