import asyncio
import threading
import traceback
import bittensor as bt
from typing import List, Tuple, Optional, Dict, Any
from common import constants
from common.data import DataEntity, DataLabel, DataSource
from common.date_range import DateRange
from scraping.scraper import ScrapeConfig, Scraper, ValidationResult, HFValidationResult
from scraping.x.model import XContent
# from scraping.x.twitter_api_scraper import ApiDojoTwitterScraper
from scraping.x.twitter_api_scraper import ApiDojoTwitterScraper
from scraping.x import utils
import datetime as dt
import json

# Import the EnhancedXContent class
from scraping.x.on_demand_model import EnhancedXContent


class EnhancedTwitterApiScraper(ApiDojoTwitterScraper):
    """
    An enhanced version of TwitterApiScraper that collects more detailed Twitter data
    using the EnhancedXContent model and Twitter API.
    """

    def __init__(self, bearer_token=None):
        # Initialize the parent class
        super().__init__(bearer_token=bearer_token)
        self.enhanced_contents = []

    def _best_effort_parse_dataset(self, tweets) -> Tuple[List[XContent], List[bool]]:
        """
        Enhanced version that parses the full dataset into both standard XContent (for backward compatibility)
        and EnhancedXContent objects.

        Returns:
            Tuple[List[XContent], List[bool]]: (standard_parsed_content, is_retweets)
        """
        # Call the parent class method to get standard parsed content
        standard_contents, is_retweets = super()._best_effort_parse_dataset(tweets)

        # Also parse into enhanced content and store it in a class attribute
        self.enhanced_contents = self._parse_enhanced_content(tweets)

        return standard_contents, is_retweets

    def _parse_enhanced_content(self, tweets: List[dict]) -> List[EnhancedXContent]:
        """
        Parse the tweets into EnhancedXContent objects with all available metadata.

        Args:
            tweets (List[dict]): The raw tweets from Twitter API.

        Returns:
            List[EnhancedXContent]: List of parsed EnhancedXContent objects.
        """
        if not tweets:
            return []

        results: List[EnhancedXContent] = []
        for tweet in tweets:
            try:
                # Extract user information
                user_id = tweet.get('author_id')
                username = tweet.get('username', '')
                display_name = tweet.get('name', '')
                verified = tweet.get('verified', False)
                followers_count = tweet.get('public_metrics', {}).get('followers_count', 0)
                following_count = tweet.get('public_metrics', {}).get('following_count', 0)

                # Extract tweet metadata
                tweet_id = tweet.get('id')
                like_count = tweet.get('public_metrics', {}).get('like_count', 0)
                retweet_count = tweet.get('public_metrics', {}).get('retweet_count', 0)
                reply_count = tweet.get('public_metrics', {}).get('reply_count', 0)
                quote_count = tweet.get('public_metrics', {}).get('quote_count', 0)
                view_count = tweet.get('public_metrics', {}).get('impression_count', 0)

                # Determine tweet type
                is_retweet = tweet.get('referenced_tweets', [])
                is_reply = any(ref.get('type') == 'replied_to' for ref in is_retweet)
                is_quote = any(ref.get('type') == 'quoted' for ref in is_retweet)
                is_retweet = any(ref.get('type') == 'retweeted' for ref in is_retweet)

                # Extract conversation and reply data
                conversation_id = tweet.get('conversation_id')
                in_reply_to_user_id = None
                if is_reply and tweet.get('referenced_tweets'):
                    for ref in tweet['referenced_tweets']:
                        if ref.get('type') == 'replied_to':
                            in_reply_to_user_id = ref.get('id')

                # Extract hashtags and cashtags
                hashtags = []
                cashtags = []
                if 'entities' in tweet and 'hashtags' in tweet['entities']:
                    hashtags = ["#" + item['tag'] for item in tweet['entities']['hashtags']]
                if 'entities' in tweet and 'cashtags' in tweet['entities']:
                    cashtags = ["$" + item['tag'] for item in tweet['entities']['cashtags']]

                # Sort hashtags and cashtags by index if available
                sorted_tags = []
                if 'entities' in tweet:
                    tag_items = []
                    if 'hashtags' in tweet['entities']:
                        for item in tweet['entities']['hashtags']:
                            tag_items.append({
                                'text': item['tag'],
                                'indices': item.get('indices', [0, 0]),
                                'type': 'hashtag'
                            })
                    if 'cashtags' in tweet['entities']:
                        for item in tweet['entities']['cashtags']:
                            tag_items.append({
                                'text': item['tag'],
                                'indices': item.get('indices', [0, 0]),
                                'type': 'cashtag'
                            })
                    
                    # Sort by first index
                    sorted_items = sorted(tag_items, key=lambda x: x['indices'][0])
                    sorted_tags = ["#" + item['text'] if item['type'] == 'hashtag' else "$" + item['text']
                                   for item in sorted_items]

                # Extract media content
                media_urls = []
                media_types = []
                if 'entities' in tweet and 'urls' in tweet['entities']:
                    for url_entity in tweet['entities']['urls']:
                        if 'images' in url_entity:
                            for image in url_entity['images']:
                                media_urls.append(image.get('url', ''))
                                media_types.append('photo')

                # Create timestamp
                timestamp = None
                if 'created_at' in tweet:
                    try:
                        timestamp = dt.datetime.strptime(
                            tweet["created_at"], "%Y-%m-%dT%H:%M:%S.%fZ"
                        ).replace(tzinfo=dt.timezone.utc)
                    except ValueError:
                        try:
                            timestamp = dt.datetime.strptime(
                                tweet["created_at"], "%Y-%m-%dT%H:%M:%SZ"
                            ).replace(tzinfo=dt.timezone.utc)
                        except ValueError:
                            timestamp = dt.datetime.now(dt.timezone.utc)

                # Create the enhanced content object
                enhanced_content = EnhancedXContent(
                    # Basic fields
                    username=f"@{username}" if username else "",
                    text=utils.sanitize_scraped_tweet(tweet.get('text', '')),
                    url=f"https://twitter.com/{username}/status/{tweet_id}" if username and tweet_id else "",
                    timestamp=timestamp,
                    tweet_hashtags=sorted_tags,

                    # Enhanced user fields
                    user_id=user_id,
                    user_display_name=display_name,
                    user_verified=verified,
                    user_followers_count=followers_count,
                    user_following_count=following_count,

                    # Enhanced tweet metadata
                    tweet_id=tweet_id,
                    like_count=like_count,
                    retweet_count=retweet_count,
                    reply_count=reply_count,
                    quote_count=quote_count,
                    is_retweet=is_retweet,
                    is_reply=is_reply,
                    is_quote=is_quote,

                    # Media content
                    media_urls=media_urls,
                    media_types=media_types,

                    # Additional metadata
                    conversation_id=conversation_id,
                    in_reply_to_user_id=in_reply_to_user_id,
                )
                results.append(enhanced_content)

            except Exception as e:
                bt.logging.warning(
                    f"Failed to decode EnhancedXContent from Twitter API response: {traceback.format_exc()}."
                )

        return results

    async def scrape(self, scrape_config: ScrapeConfig) -> List[DataEntity]:
        """
        Enhanced scrape method that uses Twitter API and returns enhanced content.
        """
        # Call parent scrape method to get standard data entities
        data_entities = await super().scrape(scrape_config)
        
        # Also populate enhanced content
        if hasattr(self, 'enhanced_contents'):
            # Enhanced content is already populated by _best_effort_parse_dataset
            pass
        else:
            # If enhanced content wasn't populated, create it from the data entities
            self.enhanced_contents = []
            for entity in data_entities:
                try:
                    # Parse the content back to create enhanced content
                    content_data = json.loads(entity.content.decode('utf-8'))
                    enhanced_content = EnhancedXContent.from_api_response(content_data)
                    self.enhanced_contents.append(enhanced_content)
                except Exception as e:
                    bt.logging.warning(f"Failed to create enhanced content from entity: {e}")
        
        return data_entities

    async def scrape_enhanced(self, scrape_config: ScrapeConfig) -> List[EnhancedXContent]:
        """
        Scrape and return enhanced content directly.
        """
        await self.scrape(scrape_config)
        return self.enhanced_contents

    async def get_enhanced_data_entities(self, scrape_config: ScrapeConfig) -> List[DataEntity]:
        """
        Scrape and return data entities with enhanced content stored in the content field.
        """
        # Get enhanced content
        enhanced_contents = await self.scrape_enhanced(scrape_config)
        
        # Convert to data entities
        enhanced_data_entities = []
        for content in enhanced_contents:
            # Convert to DataEntity but store full rich content in serialized form
            api_response = content.to_api_response()
            data_entity = DataEntity(
                uri=content.url,
                datetime=content.timestamp,
                source=DataSource.X,
                label=DataLabel(value=content.tweet_hashtags[0].lower()) if content.tweet_hashtags else None,
                # Store the full enhanced content as serialized JSON in the content field
                content=json.dumps(api_response).encode('utf-8'),
                content_size_bytes=len(json.dumps(api_response))
            )
            enhanced_data_entities.append(data_entity)
        
        return enhanced_data_entities

    def get_enhanced_content(self) -> List[EnhancedXContent]:
        """Returns the enhanced content from the last scrape."""
        return self.enhanced_contents


async def test_enhanced_scraper():
    """Test function for the enhanced Twitter API scraper."""
    bt.logging.info("Testing Enhanced Twitter API Scraper...")
    
    # Create scraper instance
    scraper = EnhancedTwitterApiScraper()
    
    # Test configuration
    config = ScrapeConfig(
        entity_limit=10,
        date_range=DateRange(
            start=dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1),
            end=dt.datetime.now(dt.timezone.utc)
        ),
        labels=[DataLabel(value="#bittensor")]
    )
    
    try:
        # Test enhanced scraping
        enhanced_data_entities = await scraper.get_enhanced_data_entities(config)
        bt.logging.info(f"Retrieved {len(enhanced_data_entities)} enhanced data entities")
        
        # Test getting enhanced content
        enhanced_content = scraper.get_enhanced_content()
        bt.logging.info(f"Retrieved {len(enhanced_content)} enhanced content items")
        
        # Print sample enhanced content
        if enhanced_content:
            print_enriched_content(enhanced_content[0])
        
        return enhanced_data_entities
        
    except Exception as e:
        bt.logging.error(f"Error testing enhanced scraper: {e}")
        bt.logging.debug(traceback.format_exc())
        return []


def print_enriched_content(content: EnhancedXContent):
    """Print detailed information about enhanced content."""
    print(f"\n=== Enhanced Twitter Content ===")
    print(f"Username: {content.username}")
    print(f"Display Name: {content.user_display_name}")
    print(f"Verified: {content.user_verified}")
    print(f"Followers: {content.user_followers_count}")
    print(f"Text: {content.text[:100]}...")
    print(f"URL: {content.url}")
    print(f"Timestamp: {content.timestamp}")
    print(f"Hashtags: {content.tweet_hashtags}")
    print(f"Like Count: {content.like_count}")
    print(f"Retweet Count: {content.retweet_count}")
    print(f"Reply Count: {content.reply_count}")
    print(f"Quote Count: {content.quote_count}")
    print(f"Is Retweet: {content.is_retweet}")
    print(f"Is Reply: {content.is_reply}")
    print(f"Is Quote: {content.is_quote}")
    print(f"Media URLs: {content.media_urls}")
    print(f"Media Types: {content.media_types}")
    print(f"Conversation ID: {content.conversation_id}")
    print(f"In Reply To User ID: {content.in_reply_to_user_id}")
    print("=" * 40)


if __name__ == "__main__":
    asyncio.run(test_enhanced_scraper()) 