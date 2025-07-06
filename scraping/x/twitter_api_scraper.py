import asyncio
import threading
import traceback
import bittensor as bt
from typing import List, Tuple, Optional
from common import constants
from common.data import DataEntity, DataLabel, DataSource
from common.date_range import DateRange
from scraping.scraper import ScrapeConfig, Scraper, ValidationResult, HFValidationResult
from scraping.x.model import XContent
from scraping.x import utils
import datetime as dt
try:
    import tweepy
except ImportError:
    bt.logging.warning("tweepy not installed. Twitter API scraper will not work.")
    tweepy = None

class ApiDojoTwitterScraper(Scraper):
    """
    Scrapes tweets using the Apidojo Twitter Scraper: https://console.apify.com/actors/61RPP7dywgiy0JPD0.
    """


    SCRAPE_TIMEOUT_SECS = 120

    def __init__(self, bearer_token=None):
        # Use the provided bearer_token or get from environment/config
        if bearer_token is None:
            import os
            bearer_token = os.environ.get("TWITTER_BEARER_TOKEN")
            bt.logging.info(f"Bearer token: {bearer_token}")
        self.client = tweepy.Client(bearer_token=bearer_token)

    async def validate(self, entities: List[DataEntity]) -> List[ValidationResult]:
        """Validate the correctness of a DataEntity by URI using Twitter API."""

        async def validate_entity(entity) -> ValidationResult:
            if not utils.is_valid_twitter_url(entity.uri):
                return ValidationResult(
                    is_valid=False,
                    reason="Invalid URI.",
                    content_size_bytes_validated=entity.content_size_bytes,
                )
            
            try:
                # Extract tweet ID from URL
                tweet_id = utils.extract_tweet_id_from_url(entity.uri)
                if not tweet_id:
                    return ValidationResult(
                        is_valid=False,
                        reason="Could not extract tweet ID from URL.",
                        content_size_bytes_validated=entity.content_size_bytes,
                    )
                
                # Fetch tweet using Twitter API
                response = self.client.get_tweet(
                    tweet_id,
                    tweet_fields=["created_at", "author_id", "text", "id", "entities"]
                )
                
                if not response.data:
                    return ValidationResult(
                        is_valid=False,
                        reason="Tweet not found via Twitter API.",
                        content_size_bytes_validated=entity.content_size_bytes,
                    )
                
                # Convert to XContent and validate
                x_content = self._xcontent_from_tweepy_tweet(response.data)
                return utils.validate_tweet_content(
                    actual_tweet=x_content,
                    entity=entity,
                    is_retweet=False  # We'll need to check this from the API response
                )
                
            except Exception as e:
                bt.logging.error(f"Failed to validate tweet {entity.uri}: {e}")
                return ValidationResult(
                    is_valid=False,
                    reason=f"Twitter API validation failed: {str(e)}",
                    content_size_bytes_validated=entity.content_size_bytes,
                )

        if not entities:
            return []

        results = await asyncio.gather(
            *[validate_entity(entity) for entity in entities]
        )

        return results

    async def validate_hf(self, entities) -> HFValidationResult:
        """Validate the correctness of a HFEntities by URL."""

        async def validate_hf_entity(entity) -> ValidationResult:
            if not utils.is_valid_twitter_url(entity.get('url')):
                return ValidationResult(
                    is_valid=False,
                    reason="Invalid URI.",
                    content_size_bytes_validated=0,
                )

            attempt = 0
            max_attempts = 2

            while attempt < max_attempts:
                # Increment attempt.
                attempt += 1

                # On attempt 1 we fetch the exact number of tweets. On retry we fetch more in case they are in replies.
                tweet_count = 1 if attempt == 1 else 5

                run_input = {
                    **ApiDojoTwitterScraper.BASE_RUN_INPUT,
                    "startUrls": [entity.get('url')],
                    "maxItems": tweet_count,
                }
                run_config = RunConfig(
                    actor_id=ApiDojoTwitterScraper.ACTOR_ID,
                    debug_info=f"Validate {entity.get('url')}",
                    max_data_entities=tweet_count,
                )

                # Retrieve the tweets from Apify.
                dataset: List[dict] = None
                try:
                    dataset: List[dict] = await self.runner.run(run_config, run_input)
                except (
                        Exception
                ) as e:  # Catch all exceptions here to ensure we do not exit validation early.
                    if attempt != max_attempts:
                        # Retrying.
                        continue
                    else:
                        bt.logging.error(
                            f"Failed to run actor: {traceback.format_exc()}."
                        )
                        # This is an unfortunate situation. We have no way to distinguish a genuine failure from
                        # one caused by malicious input. In my own testing I was able to make the Actor timeout by
                        # using a bad URI. As such, we have to penalize the miner here. If we didn't they could
                        # pass malicious input for chunks they don't have.
                        return ValidationResult(
                            is_valid=False,
                            reason="Failed to run Actor. This can happen if the URI is invalid, or APIfy is having an issue.",
                            content_size_bytes_validated=0,
                        )

                # Parse the response
                tweets = self._best_effort_parse_hf_dataset(dataset)
                actual_tweet = None

                for index, tweet in enumerate(tweets):
                    if utils.normalize_url(tweet['url']) == utils.normalize_url(entity.get('url')):
                        actual_tweet = tweet
                        break

                bt.logging.debug(actual_tweet)
                if actual_tweet is None:
                    # Only append a failed result if on final attempt.
                    if attempt == max_attempts:
                        return ValidationResult(
                            is_valid=False,
                            reason="Tweet not found or is invalid.",
                            content_size_bytes_validated=0,
                        )
                else:
                    return utils.validate_hf_retrieved_tweet(
                        actual_tweet=actual_tweet,
                        tweet_to_verify=entity
                    )

        # Since we are using the threading.semaphore we need to use it in a context outside of asyncio.
        bt.logging.trace("Acquiring semaphore for concurrent apidojo validations.")

        with ApiDojoTwitterScraper.concurrent_validates_semaphore:
            bt.logging.trace(
                "Acquired semaphore for concurrent apidojo validations."
            )
            results = await asyncio.gather(
                *[validate_hf_entity(entity) for entity in entities]
            )

        is_valid, valid_percent = utils.hf_tweet_validation(validation_results=results)
        return HFValidationResult(is_valid=is_valid, validation_percentage=valid_percent,
                                  reason=f"Validation Percentage = {valid_percent}")

    async def scrape(self, scrape_config: ScrapeConfig) -> List[DataEntity]:
        query_parts = []

        # Handle labels - separate usernames and keywords
        if scrape_config.labels:
            username_labels = []
            keyword_labels = []

            for label in scrape_config.labels:
                if label.value.startswith('@'):
                    username = label.value[1:]
                    username_labels.append(f"from:{username}")
                else:
                    keyword_labels.append(label.value)

            # Deduplicate labels to prevent rule length issues
            username_labels = list(set(username_labels))
            keyword_labels = list(set(keyword_labels))
            
            # Limit the number of labels to prevent query length issues
            MAX_LABELS_PER_TYPE = 10  # Reasonable limit to stay under 512 chars
            if len(username_labels) > MAX_LABELS_PER_TYPE:
                bt.logging.warning(f"Too many username labels ({len(username_labels)}), limiting to {MAX_LABELS_PER_TYPE}")
                username_labels = username_labels[:MAX_LABELS_PER_TYPE]
            
            if len(keyword_labels) > MAX_LABELS_PER_TYPE:
                bt.logging.warning(f"Too many keyword labels ({len(keyword_labels)}), limiting to {MAX_LABELS_PER_TYPE}")
                keyword_labels = keyword_labels[:MAX_LABELS_PER_TYPE]

            if username_labels:
                query_parts.append(f"({' OR '.join(username_labels)})")
            if keyword_labels:
                query_parts.append(f"({' OR '.join(keyword_labels)})")
        else:
            query_parts.append("e")
        
        query = " ".join(query_parts)
        
        # Validate query length and truncate if necessary
        query = self._validate_and_truncate_query(query)
        max_results = min(scrape_config.entity_limit or 100, 100)  # Twitter API max is 100 per request

        # Clamp start_time and end_time to Twitter API's allowed window (last 7 days)
        now = dt.datetime.now(dt.timezone.utc)
        max_lookback = dt.timedelta(days=7)
        min_start_time = now - max_lookback
        orig_start = scrape_config.date_range.start
        orig_end = scrape_config.date_range.end
        clamped_start = max(orig_start, min_start_time)
        bt.logging.info(f"Clamped start: {clamped_start}")
        clamped_end = min(orig_end, now)
        bt.logging.info(f"Clamped end: {clamped_end}")
        if clamped_start != orig_start or clamped_end != orig_end:
            bt.logging.warning(f"Clamping Twitter API date range: requested start={orig_start}, end={orig_end}; clamped to start={clamped_start}, end={clamped_end}")
        if clamped_start >= clamped_end:
            bt.logging.warning(
                f"Twitter API date range invalid after clamping: start={clamped_start}, end={clamped_end}. Skipping request."
            )
            return []
        start_time = clamped_start.isoformat()
        end_time = clamped_end.isoformat()

        bt.logging.success(f"Performing Twitter scrape for search terms: {query}.")
        tweets = []
        try:
            response = self.client.search_recent_tweets(
                query=query,
                start_time=start_time,
                end_time=end_time,
                max_results=max_results,
                tweet_fields=["created_at", "author_id", "text", "id", "entities"]
            )
            bt.logging.info(f"Twitter Response: {response}")
            tweets = response.data if response.data else []
        except Exception as e:
            bt.logging.error(f"Error fetching tweets: {e}")
            return []

        # Get the primary label from scrape_config
        primary_label = scrape_config.labels[0] if scrape_config.labels else None
        
        # Process tweets and create DataEntity objects with proper labels
        data_entities = []
        for tweet in tweets:
            try:
                # Create XContent from tweet
                x_content = self._xcontent_from_tweepy_tweet(tweet)
                
                # Modify hashtags to set the primary label first
                if primary_label is not None:
                    # Use the primary label from config as the first hashtag
                    x_content.tweet_hashtags = [primary_label.value] + x_content.tweet_hashtags
                elif not x_content.tweet_hashtags:
                    # If no hashtags and no config label, add a default
                    x_content.tweet_hashtags = ["#general"]
                
                # Create DataEntity - XContent.to_data_entity will use the first hashtag as label
                data_entity = XContent.to_data_entity(x_content)
                data_entities.append(data_entity)
            except Exception as e:
                bt.logging.warning(f"Failed to process tweet: {e}")
                continue
            
        bt.logging.info(f"Data entities: {data_entities}")
        bt.logging.success(
            f"Completed scrape for {query}. Scraped {len(data_entities)} items."
        )
        return data_entities

    def _best_effort_parse_dataset(self, tweets) -> List[DataEntity]:
        """Converts a list of Tweepy tweet objects into DataEntity objects."""
        if not tweets:
            return []
        results: List[DataEntity] = []
        for tweet in tweets:
            try:
                x_content = self._xcontent_from_tweepy_tweet(tweet)
                data_entity = XContent.to_data_entity(x_content)
                bt.logging.info(f"Data entity: {data_entity}")
                results.append(data_entity)
                bt.logging.info(f"Results: {results}")
            except Exception as e:
                bt.logging.warning(f"Failed to parse tweet to XContent/DataEntity: {e}")
        return results                
    def _xcontent_from_tweepy_tweet(self, tweet, includes=None):
        """
        Helper to convert a Tweepy tweet object to XContent with all available metadata.
        Optionally uses the includes dict for user/media expansion.
        """
        # Extract hashtags
        hashtags = []
        if hasattr(tweet, 'entities') and tweet.entities and 'hashtags' in tweet.entities:
            hashtags = [f"#{tag['tag']}" if isinstance(tag, dict) and 'tag' in tag else f"#{tag['text']}" for tag in tweet.entities['hashtags']]
        print("[DEBUG] Extracted hashtags:", hashtags)

        # Extract media URLs (if available)
        media_urls = []
        if includes and hasattr(tweet, 'attachments') and tweet.attachments and 'media_keys' in tweet.attachments:
            media_keys = tweet.attachments['media_keys']
            media_dict = {m['media_key']: m for m in includes.get('media', [])}
            for key in media_keys:
                media = media_dict.get(key)
                if media and 'url' in media:
                    media_urls.append(media['url'])

        # Extract user info (requires expansions='author_id' and user_fields)
        user_id = getattr(tweet, 'author_id', None)
        user_display_name = None
        user_verified = None
        username = user_id
        if includes and 'users' in includes and user_id:
            user_obj = next((u for u in includes['users'] if u['id'] == str(user_id)), None)
            if user_obj:
                user_display_name = user_obj.get('name')
                user_verified = user_obj.get('verified')
                username = user_obj.get('username')

        # Extract reply/quote/conversation info
        is_reply = hasattr(tweet, 'in_reply_to_user_id') and tweet.in_reply_to_user_id is not None
        is_quote = False
        if hasattr(tweet, 'referenced_tweets') and tweet.referenced_tweets:
            is_quote = any(ref.type == 'quoted' for ref in tweet.referenced_tweets)
        conversation_id = getattr(tweet, 'conversation_id', None)
        in_reply_to_user_id = getattr(tweet, 'in_reply_to_user_id', None)

        return XContent(
            username=username,
            text=tweet.text,
            url=f"https://twitter.com/i/web/status/{tweet.id}",
            timestamp=tweet.created_at,
            tweet_hashtags=hashtags,
            media=media_urls if media_urls else None,
            user_id=str(user_id) if user_id is not None else None,
            user_display_name=user_display_name,
            user_verified=user_verified,
            tweet_id=str(tweet.id) if tweet.id is not None else None,
            is_reply=is_reply,
            is_quote=is_quote,
            conversation_id=str(conversation_id) if conversation_id is not None else None,
            in_reply_to_user_id=str(in_reply_to_user_id) if in_reply_to_user_id is not None else None,
        )            

    def _best_effort_parse_hf_dataset(self, dataset: List[dict]) -> List[DataEntity]:
        """
        Parses a HuggingFace dataset (list of dicts) into DataEntity objects using XContent.
        Extracts all available metadata fields. Logs and skips any entries that cannot be parsed.
        """
        if not dataset or dataset == [{"zero_result": True}]:
            return []
        results: List[DataEntity] = []
        for data in dataset:
            try:
                # Required fields
                text = utils.sanitize_scraped_tweet(data.get('text', ''))
                url = data.get('url')
                created_at = data.get('createdAt') or data.get('timestamp')
                if not (text and url and created_at):
                    continue
                # Parse datetime
                try:
                    if isinstance(created_at, str):
                        # Try multiple formats
                        try:
                            timestamp = dt.datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y")
                        except Exception:
                            timestamp = dt.datetime.fromisoformat(created_at)
                    else:
                        timestamp = created_at
                except Exception as e:
                    bt.logging.warning(f"Failed to parse datetime: {e}")
                    continue
                # Hashtags
                hashtags = data.get('tweet_hashtags') or utils.extract_hashtags(text)
                # Media
                media_urls = []
                if 'media' in data and isinstance(data['media'], list):
                    for media_item in data['media']:
                        if isinstance(media_item, dict) and 'media_url_https' in media_item:
                            media_urls.append(media_item['media_url_https'])
                        elif isinstance(media_item, dict) and 'url' in media_item:
                            media_urls.append(media_item['url'])
                        elif isinstance(media_item, str):
                            media_urls.append(media_item)
                # Other metadata
                xcontent = XContent(
                    username=data.get('username') or utils.extract_user(url),
                    text=text,
                    url=url,
                    timestamp=timestamp,
                    tweet_hashtags=hashtags,
                    media=media_urls if media_urls else None,
                    user_id=data.get('user_id'),
                    user_display_name=data.get('user_display_name'),
                    user_verified=data.get('user_verified'),
                    tweet_id=str(data.get('tweet_id') or data.get('id') or ''),
                    is_reply=data.get('is_reply'),
                    is_quote=data.get('is_quote'),
                    conversation_id=data.get('conversation_id'),
                    in_reply_to_user_id=data.get('in_reply_to_user_id'),
                )
                data_entity = XContent.to_data_entity(xcontent)
                results.append(data_entity)
            except Exception as e:
                bt.logging.warning(f"Failed to parse HF dataset entry to XContent/DataEntity: {e}")

        return results

    def _validate_and_truncate_query(self, query: str) -> str:
        """
        Validates and truncates the Twitter search query to stay within the 512 character limit.
        
        Args:
            query: The original search query
            
        Returns:
            A valid query string within Twitter's limits
        """
        MAX_QUERY_LENGTH = 512
        
        if len(query) <= MAX_QUERY_LENGTH:
            return query
        
        bt.logging.warning(f"Query length ({len(query)}) exceeds Twitter's limit ({MAX_QUERY_LENGTH}). Truncating...")
        
        # If query is too long, try to keep the most important parts
        # Priority: usernames first, then keywords
        query_parts = query.split()
        
        # Try to keep usernames (from:...) first
        username_parts = [part for part in query_parts if part.startswith('(from:')]
        keyword_parts = [part for part in query_parts if not part.startswith('(from:')]
        
        # Build truncated query
        truncated_parts = []
        current_length = 0
        
        # Add usernames first
        for part in username_parts:
            if current_length + len(part) + 1 <= MAX_QUERY_LENGTH:
                truncated_parts.append(part)
                current_length += len(part) + 1
            else:
                break
        
        # Add keywords if space allows
        for part in keyword_parts:
            if current_length + len(part) + 1 <= MAX_QUERY_LENGTH:
                truncated_parts.append(part)
                current_length += len(part) + 1
            else:
                break
        
        truncated_query = " ".join(truncated_parts)
        
        # If still too long, take just the first few parts
        if len(truncated_query) > MAX_QUERY_LENGTH:
            # Emergency fallback: take first few parts that fit
            emergency_parts = []
            emergency_length = 0
            for part in query_parts:
                if emergency_length + len(part) + 1 <= MAX_QUERY_LENGTH:
                    emergency_parts.append(part)
                    emergency_length += len(part) + 1
                else:
                    break
            truncated_query = " ".join(emergency_parts)
        
        bt.logging.info(f"Truncated query: {truncated_query} (length: {len(truncated_query)})")
        return truncated_query


async def test_scrape():
    scraper = ApiDojoTwitterScraper()

    entities = await scraper.scrape(
        ScrapeConfig(
            entity_limit=100,
            date_range=DateRange(
                start=dt.datetime(2024, 5, 27, 0, 0, 0, tzinfo=dt.timezone.utc),
                end=dt.datetime(2024, 5, 27, 9, 0, 0, tzinfo=dt.timezone.utc),
            ),
            labels=[DataLabel(value="#bittgergnergerojngoierjgensor")],
        )
    )

    return entities


async def test_validate():
    scraper = ApiDojoTwitterScraper()

    true_entities = [
        DataEntity(
            uri="https://x.com/0xedeon/status/1790788053960667309",
            datetime=dt.datetime(2024, 5, 15, 16, 55, 17, tzinfo=dt.timezone.utc),
            source=DataSource.X,
            label=DataLabel(value="#cryptocurrency"),
            content='{"username": "@0xedeon", "text": "Deux frères ont manipulé les protocoles Ethereum pour voler 25M $ selon le Département de la Justice 🕵️‍♂️💰 #Cryptocurrency #JusticeDept", "url": "https://x.com/0xedeon/status/1790788053960667309", "timestamp": "2024-05-15T16:55:00+00:00", "tweet_hashtags": ["#Cryptocurrency", "#JusticeDept"]}',
            content_size_bytes=391
        ),
        DataEntity(
            uri="https://x.com/100Xpotential/status/1790785842967101530",
            datetime=dt.datetime(2024, 5, 15, 16, 46, 30, tzinfo=dt.timezone.utc),
            source=DataSource.X,
            label=DataLabel(value="#catcoin"),
            content='{"username": "@100Xpotential", "text": "As i said green candles incoming 🚀🫡👇👇\\n\\nAround 15% price surge in #CatCoin 📊💸🚀🚀\\n\\n𝐂𝐨𝐦𝐦𝐞𝐧𝐭 |  𝐋𝐢𝐤𝐞 |  𝐑𝐞𝐭𝐰𝐞𝐞𝐭 |  𝐅𝐨𝐥𝐥𝐨𝐰\\n\\n#Binance #Bitcoin #PiNetwork #Blockchain #NFT #BabyDoge #Solana #PEPE #Crypto #1000x #cryptocurrency #Catcoin #100x", "url": "https://x.com/100Xpotential/status/1790785842967101530", "timestamp": "2024-05-15T16:46:00+00:00", "tweet_hashtags": ["#CatCoin", "#Binance", "#Bitcoin", "#PiNetwork", "#Blockchain", "#NFT", "#BabyDoge", "#Solana", "#PEPE", "#Crypto", "#1000x", "#cryptocurrency", "#Catcoin", "#100x"]}',
            content_size_bytes=933
        ),
        DataEntity(
            uri="https://x.com/20nineCapitaL/status/1789488160688541878",
            datetime=dt.datetime(2024, 5, 12, 2, 49, 59, tzinfo=dt.timezone.utc),
            source=DataSource.X,
            label=DataLabel(value="#bitcoin"),
            content='{"username": "@20nineCapitaL", "text": "Yup! We agreed to. \\n\\n@MetaMaskSupport #Bitcoin #Investors #DigitalAssets #EthereumETF #Airdrops", "url": "https://x.com/20nineCapitaL/status/1789488160688541878", "timestamp": "2024-05-12T02:49:00+00:00", "tweet_hashtags": ["#Bitcoin", "#Investors", "#DigitalAssets", "#EthereumETF", "#Airdrops"]}',
            content_size_bytes=345
        ),
        DataEntity(
            uri="https://x.com/AAAlviarez/status/1790787185047658838",
            datetime=dt.datetime(2024, 5, 15, 16, 51, 50, tzinfo=dt.timezone.utc),
            source=DataSource.X,
            label=DataLabel(value="#web3‌‌"),
            content='{"username": "@AAAlviarez", "text": "1/3🧵\\n\\nOnce a month dozens of #web3‌‌  users show our support to one of the projects that is doing an excellent job in services and #cryptocurrency adoption.\\n\\nDo you know what Leo Power Up Day is all about?", "url": "https://x.com/AAAlviarez/status/1790787185047658838", "timestamp": "2024-05-15T16:51:00+00:00", "tweet_hashtags": ["#web3‌‌", "#cryptocurrency"]}',
            content_size_bytes=439
        ),
        DataEntity(
            uri="https://x.com/AGariaparra/status/1789488091453091936",
            datetime=dt.datetime(2024, 5, 12, 2, 49, 42, tzinfo=dt.timezone.utc),
            source=DataSource.X,
            label=DataLabel(value="#bitcoin"),
            content='{"username": "@AGariaparra", "text": "J.P Morgan, Wells Fargo hold #Bitcoin now: Why are they interested in BTC? - AMBCrypto", "url": "https://x.com/AGariaparra/status/1789488091453091936", "timestamp": "2024-05-12T02:49:00+00:00", "tweet_hashtags": ["#Bitcoin"]}',
            content_size_bytes=269
        ),
        DataEntity(
            uri="https://x.com/AGariaparra/status/1789488427546939525",
            datetime=dt.datetime(2024, 5, 12, 2, 51, 2, tzinfo=dt.timezone.utc),
            source=DataSource.X,
            label=DataLabel(value="#bitcoin"),
            content='{"username": "@AGariaparra", "text": "We Asked ChatGPT if #Bitcoin Will Enter a Massive Bull Run in 2024", "url": "https://x.com/AGariaparra/status/1789488427546939525", "timestamp": "2024-05-12T02:51:00+00:00", "tweet_hashtags": ["#Bitcoin"]}',
            content_size_bytes=249
        ),
        DataEntity(
            uri="https://x.com/AMikulanecs/status/1784324497895522673",
            datetime=dt.datetime(2024, 4, 27, 20, 51, 26, tzinfo=dt.timezone.utc),
            source=DataSource.X,
            label=DataLabel(value="#felix"),
            content='{"username": "@AMikulanecs", "text": "$FELIX The new Dog with OG Vibes... \\nWe have a clear vision for success.\\nNew Dog $FELIX \\n➡️Follow @FelixInuETH \\n➡️Join➡️Visit#memecoins #BTC #MemeCoinSeason #Bullrun2024 #Ethereum #altcoin #Crypto #meme #SOL #BaseChain #Binance", "url": "https://x.com/AMikulanecs/status/1784324497895522673", "timestamp": "2024-04-27T20:51:00+00:00", "tweet_hashtags": ["#FELIX", "#FELIX", "#memecoins", "#BTC", "#MemeCoinSeason", "#Bullrun2024", "#Ethereum", "#altcoin", "#Crypto", "#meme", "#SOL", "#BaseChain", "#Binance"]}',
            content_size_bytes=588
        ),
        DataEntity(
            uri="https://x.com/AdamEShelton/status/1789490040751411475",
            datetime=dt.datetime(2024, 5, 12, 2, 57, 27, tzinfo=dt.timezone.utc),
            source=DataSource.X,
            label=DataLabel(value="#bitcoin"),
            content='{"username": "@AdamEShelton", "text": "#bitcoin  love", "url": "https://x.com/AdamEShelton/status/1789490040751411475", "timestamp": "2024-05-12T02:57:00+00:00", "tweet_hashtags": ["#bitcoin"]}',
            content_size_bytes=199
        ),
        DataEntity(
            uri="https://x.com/AfroWestor/status/1789488798406975580",
            datetime=dt.datetime(2024, 5, 12, 2, 52, 31, tzinfo=dt.timezone.utc),
            source=DataSource.X,
            label=DataLabel(value="#bitcoin"),
            content='{"username": "@AfroWestor", "text": "Given is for Prince and princess form inheritances  to kingdom. \\n\\nWe the #BITCOIN family we Gain profits for ever. \\n\\nSo if you embrace #BTC that means you have a Kingdom to pass on for ever.", "url": "https://x.com/AfroWestor/status/1789488798406975580", "timestamp": "2024-05-12T02:52:00+00:00", "tweet_hashtags": ["#BITCOIN", "#BTC"]}',
            content_size_bytes=383
        ),
        DataEntity(
            uri="https://x.com/AlexEmidio7/status/1789488453979189327",
            datetime=dt.datetime(2024, 5, 12, 2, 51, 9, tzinfo=dt.timezone.utc),
            source=DataSource.X,
            label=DataLabel(value="#bitcoin"),
            content='{"username": "@AlexEmidio7", "text": "Bip47 V3 V4 #Bitcoin", "url": "https://x.com/AlexEmidio7/status/1789488453979189327", "timestamp": "2024-05-12T02:51:00+00:00", "tweet_hashtags": ["#Bitcoin"]}',
            content_size_bytes=203
        ),
    ]
    results = await scraper.validate(entities=true_entities)
    for result in results:
        print(result)


async def test_multi_thread_validate():
    scraper = ApiDojoTwitterScraper()

    true_entities = [
        DataEntity(
            uri="https://x.com/bittensor_alert/status/1748585332935622672",
            datetime=dt.datetime(2024, 1, 20, 5, 56, tzinfo=dt.timezone.utc),
            source=DataSource.X,
            label=DataLabel(value="#Bittensor"),
            content='{"username":"@bittensor_alert","text":"🚨 #Bittensor Alert: 500 $TAO ($122,655) deposited into #MEXC","url":"https://twitter.com/bittensor_alert/status/1748585332935622672","timestamp":"2024-01-20T5:56:00Z","tweet_hashtags":["#Bittensor", "#TAO", "#MEXC"]}',
            content_size_bytes=318,
        ),
        DataEntity(
            uri="https://x.com/HadsonNery/status/1752011223330124021",
            datetime=dt.datetime(2024, 1, 29, 16, 50, tzinfo=dt.timezone.utc),
            source=DataSource.X,
            label=DataLabel(value="#faleitoleve"),
            content='{"username":"@HadsonNery","text":"Se ele fosse brabo mesmo e eu estaria aqui defendendo ele, pq ele não foi direto no Davi já que a intenção dele era fazer o Davi comprar o barulho dela 🤷🏻\u200d♂️ MC fofoqueiro foi macetado pela CUNHÃ #faleitoleve","url":"https://twitter.com/HadsonNery/status/1752011223330124021","timestamp":"2024-01-29T16:50:00Z","tweet_hashtags":["#faleitoleve"]}',
            content_size_bytes=492,
        ),
    ]

    def sync_validate(entities: list[DataEntity]) -> None:
        """Synchronous version of eval_miner."""
        asyncio.run(scraper.validate(entities))

    threads = [
        threading.Thread(target=sync_validate, args=(true_entities,)) for _ in range(5)
    ]

    for thread in threads:
        thread.start()

    for t in threads:
        t.join(120)


if __name__ == "__main__":
    bt.logging.set_trace(True)
    asyncio.run(test_multi_thread_validate())
    asyncio.run(test_scrape())
    asyncio.run(test_validate())