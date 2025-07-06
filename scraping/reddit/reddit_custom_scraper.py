import time
from common import constants, utils
from common.date_range import DateRange
from scraping.reddit import model
from scraping.scraper import ScrapeConfig, Scraper, ValidationResult, HFValidationResult
import bittensor as bt
from common.data import DataEntity, DataLabel, DataSource
from typing import List, Dict, Optional
import asyncpraw
from scraping.reddit.utils import (
    is_valid_reddit_url,
    validate_reddit_content,
    get_time_input,
    get_custom_sort_input,
    normalize_label,
    normalize_permalink,
)
from scraping.reddit.model import RedditContent, RedditDataType
import traceback
import datetime as dt
import asyncio
import random
import os
import json
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class RedditCredential:
    """Represents a single Reddit API credential set."""
    client_id: str
    client_secret: str
    username: str
    password: str
    user_agent: str
    last_used: float = 0.0
    request_count: int = 0
    is_active: bool = True


class RedditCredentialManager:
    """Manages multiple Reddit API credentials for load balancing and rate limit avoidance."""
    
    def __init__(self, worker_id: str = "default"):
        self.credentials: List[RedditCredential] = []
        self.current_index = 0
        self.rate_limit_window = 60  # 60 seconds window
        self.max_requests_per_window = 50  # Conservative limit per credential
        self.worker_id = worker_id
        self._load_credentials()
    
    def _load_credentials(self):
        """Load Reddit credentials from environment variables."""
        # Load primary credentials
        primary_cred = RedditCredential(
            client_id=os.getenv("REDDIT_CLIENT_ID", ""),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET", ""),
            username=os.getenv("REDDIT_USERNAME", ""),
            password=os.getenv("REDDIT_PASSWORD", ""),
            user_agent=f"User-Agent: python: {os.getenv('REDDIT_USERNAME', '')}"
        )
        
        if primary_cred.client_id and primary_cred.client_secret:
            self.credentials.append(primary_cred)
        
        # Load secondary credentials (REDDIT_CLIENT_ID_2, REDDIT_CLIENT_SECRET_2, etc.)
        for i in range(2, 6):  # Support up to 5 credential sets
            client_id = os.getenv(f"REDDIT_CLIENT_ID_{i}")
            client_secret = os.getenv(f"REDDIT_CLIENT_SECRET_{i}")
            username = os.getenv(f"REDDIT_USERNAME_{i}")
            password = os.getenv(f"REDDIT_PASSWORD_{i}")
            
            if client_id and client_secret and username and password:
                cred = RedditCredential(
                    client_id=client_id,
                    client_secret=client_secret,
                    username=username,
                    password=password,
                    user_agent=f"User-Agent: python: {username}"
                )
                self.credentials.append(cred)
        
        bt.logging.info(f"Worker {self.worker_id}: Loaded {len(self.credentials)} Reddit API credentials")
        
        if not self.credentials:
            bt.logging.warning(f"Worker {self.worker_id}: No Reddit credentials found! Please set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, etc.")
        else:
            bt.logging.info(f"Worker {self.worker_id}: Available credentials: {[cred.username for cred in self.credentials]}")
            # Test each credential to ensure they're valid
            for i, cred in enumerate(self.credentials):
                bt.logging.info(f"Worker {self.worker_id}: Credential {i+1}: {cred.username} (client_id: {cred.client_id[:8]}...)")
    
    def get_available_credential(self) -> Optional[RedditCredential]:
        """Get an available credential that hasn't exceeded rate limits."""
        current_time = time.time()
        
        # Clean up old request counts
        for cred in self.credentials:
            if current_time - cred.last_used > self.rate_limit_window:
                cred.request_count = 0
        
        # Find available credentials
        available_creds = [
            cred for cred in self.credentials 
            if cred.is_active and cred.request_count < self.max_requests_per_window
        ]
        
        if not available_creds:
            bt.logging.warning(f"Worker {self.worker_id}: All Reddit credentials are rate limited, waiting for reset...")
            return None
        
        # Use worker-specific selection strategy to distribute load
        # Ensure both workers get credentials even with limited availability
        if len(available_creds) == 1:
            # If only one credential, both workers share it
            worker_creds = available_creds
            bt.logging.info(f"Worker {self.worker_id}: Sharing single credential {available_creds[0].username}")
        elif len(available_creds) == 2:
            # If two credentials, assign one to each worker
            if self.worker_id == "REDDIT_API_1":
                worker_creds = [available_creds[0]]
            elif self.worker_id == "REDDIT_API_2":
                worker_creds = [available_creds[1]]
            else:
                worker_creds = available_creds
            bt.logging.info(f"Worker {self.worker_id}: Using dedicated credential {worker_creds[0].username}")
        elif len(available_creds) >= 3:
            # If 3+ credentials, use half-and-half distribution
            mid_point = len(available_creds) // 2
            if self.worker_id == "REDDIT_API_1":
                worker_creds = available_creds[:mid_point]
            elif self.worker_id == "REDDIT_API_2":
                worker_creds = available_creds[mid_point:]
            else:
                worker_creds = available_creds
            bt.logging.info(f"Worker {self.worker_id}: Using {len(worker_creds)} credentials from pool of {len(available_creds)}")
        else:
            # Fallback: use all available credentials
            worker_creds = available_creds
            bt.logging.warning(f"Worker {self.worker_id}: No dedicated credentials available, falling back to all credentials")
        
        # If this worker has no credentials, try to get any available credential
        if not worker_creds:
            bt.logging.warning(f"Worker {self.worker_id}: No dedicated credentials available, trying to use any available credential")
            worker_creds = available_creds
        
        if not worker_creds:
            bt.logging.error(f"Worker {self.worker_id}: No available credentials for this worker")
            return None
        
        # Round-robin selection among worker-specific credentials
        selected_cred = worker_creds[self.current_index % len(worker_creds)]
        self.current_index += 1
        
        # Update usage
        selected_cred.last_used = current_time
        selected_cred.request_count += 1
        
        bt.logging.info(f"Worker {self.worker_id}: Selected credential: {selected_cred.username} (client_id: {selected_cred.client_id[:8]}...)")
        return selected_cred
    
    def mark_credential_failed(self, credential: RedditCredential):
        """Mark a credential as failed (temporarily disable it)."""
        credential.is_active = False
        bt.logging.warning(f"Worker {self.worker_id}: Marked credential for {credential.username} as failed")
        
        # Re-enable after 5 minutes
        asyncio.create_task(self._reenable_credential(credential, 300))
    
    async def _reenable_credential(self, credential: RedditCredential, delay_seconds: int):
        """Re-enable a credential after a delay."""
        await asyncio.sleep(delay_seconds)
        credential.is_active = True
        credential.request_count = 0
        bt.logging.info(f"Worker {self.worker_id}: Re-enabled credential for {credential.username}")


class RedditCustomScraper(Scraper):
    """
    Scrapes Reddit data using multiple personal reddit accounts for load balancing.
    """

    def __init__(self, worker_id: str = "default"):
        self.credential_manager = RedditCredentialManager(worker_id=worker_id)
        self.worker_id = worker_id
        self._current_credential = None
        bt.logging.info(f"RedditCustomScraper initialized with credential manager for worker {worker_id}")

    async def _get_reddit_client(self) -> Optional[asyncpraw.Reddit]:
        """Get a Reddit client using an available credential with retry logic."""
        max_retries = 3
        retry_delay = 1  # Start with 1 second delay
        
        for attempt in range(max_retries):
            credential = self.credential_manager.get_available_credential()
            
            if not credential:
                if attempt < max_retries - 1:
                    bt.logging.warning(f"Worker {self.worker_id}: No credentials available, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    bt.logging.error(f"Worker {self.worker_id}: No credentials available after {max_retries} attempts")
                    return None
            
            # Track the current credential for error handling
            self._current_credential = credential
            
            try:
                reddit = asyncpraw.Reddit(
                    client_id=credential.client_id,
                    client_secret=credential.client_secret,
                    username=credential.username,
                    password=credential.password,
                    user_agent=credential.user_agent,
                )
                return reddit
            except Exception as e:
                bt.logging.error(f"Worker {self.worker_id}: Failed to create Reddit client for {credential.username}: {e}")
                self.credential_manager.mark_credential_failed(credential)
                
                if attempt < max_retries - 1:
                    bt.logging.warning(f"Worker {self.worker_id}: Retrying with different credential in {retry_delay}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    bt.logging.error(f"Worker {self.worker_id}: Failed to create Reddit client after {max_retries} attempts")
                    return None

    async def validate(self, entities: List[DataEntity]) -> List[ValidationResult]:
        """
        Validate a list of DataEntity objects.

        * Fails automatically if a submission is NSFW (over_18=True).
        * For comments, it checks the parent submission (and subreddit) NSFW flag.
        """
        if not entities:
            return []

        results: List[ValidationResult] = []

        for entity in entities:
            # 1) Basic URI sanity check
            if not is_valid_reddit_url(entity.uri):
                results.append(
                    ValidationResult(
                        is_valid=False,
                        reason="Invalid URI.",
                        content_size_bytes_validated=entity.content_size_bytes,
                    )
                )
                continue

            # 2) Decode RedditContent blob
            try:
                ent_content = RedditContent.from_data_entity(entity)
            except Exception:
                results.append(
                    ValidationResult(
                        is_valid=False,
                        reason="Failed to decode data entity.",
                        content_size_bytes_validated=entity.content_size_bytes,
                    )
                )
                continue

            # 3) Fetch live data from Reddit
            try:
                reddit = await self._get_reddit_client()
                if not reddit:
                    results.append(
                        ValidationResult(
                            is_valid=False,
                            reason="No available Reddit credentials.",
                            content_size_bytes_validated=entity.content_size_bytes,
                        )
                    )
                    continue

                async with reddit:
                    # ---- A) POST branch ----
                    if ent_content.data_type == RedditDataType.POST:
                        submission = await reddit.submission(url=ent_content.url)
                        await submission.load()                       # ensure attrs

                        # Check NSFW only after the filter date
                        if (dt.datetime.now(tz=dt.timezone.utc) >= constants.NSFW_REDDIT_FILTER_DATE and 
                            submission.over_18):                        # NSFW post
                            results.append(
                                ValidationResult(
                                    is_valid=False,
                                    reason="Submission is NSFW (over_18).",
                                    content_size_bytes_validated=entity.content_size_bytes,
                                )
                            )
                            continue

                        live_content = self._best_effort_parse_submission(submission)

                    # ---- B) COMMENT branch ----
                    else:
                        comment = await reddit.comment(url=ent_content.url)
                        await comment.load()

                        parent = comment.submission
                        await parent.load()                           # full parent
                        subreddit = comment.subreddit
                        await subreddit.load()                        # full subreddit

                        # Check NSFW only after the filter date
                        if (dt.datetime.now(tz=dt.timezone.utc) >= constants.NSFW_REDDIT_FILTER_DATE and 
                            (parent.over_18 or subreddit.over18)):
                            results.append(
                                ValidationResult(
                                    is_valid=False,
                                    reason="Parent submission or subreddit is NSFW (over_18).",
                                    content_size_bytes_validated=entity.content_size_bytes,
                                )
                            )
                            continue

                        live_content = self._best_effort_parse_comment(comment)

            except Exception as e:
                bt.logging.error(f"Failed to retrieve content for {entity.uri}: {e}")
                results.append(
                    ValidationResult(
                        is_valid=False,
                        reason="Failed to retrieve submission/comment from Reddit.",
                        content_size_bytes_validated=entity.content_size_bytes,
                    )
                )
                continue

            # 4) Live content object exists?
            if not live_content:
                results.append(
                    ValidationResult(
                        is_valid=False,
                        reason="Reddit content not found or invalid.",
                        content_size_bytes_validated=entity.content_size_bytes,
                    )
                )
                continue

            # 5) Field-by-field validation
            results.append(
                validate_reddit_content(
                    actual_content=live_content,
                    entity_to_validate=entity,
                )
            )

        return results

    async def validate_hf(self, entities) -> HFValidationResult:
        """Validate the correctness of HFEntities by URL, focusing on username, date (hour), and text."""
        if not entities:
            return HFValidationResult(is_valid=True, validation_percentage=100.0, reason="No entities to validate")

        validation_results = []

        for entity in entities:
            if not is_valid_reddit_url(entity.get('url')):
                validation_results.append(False)
                continue

            content = None
            try:
                reddit = await self._get_reddit_client()
                if not reddit:
                    validation_results.append(False)
                    continue

                async with reddit:
                    if entity.get('dataType') == RedditDataType.COMMENT:
                        comment = await reddit.comment(url=entity.get('url'))
                        content = self._best_effort_parse_comment(comment)
                    else:
                        submission = await reddit.submission(url=entity.get('url'))
                        content = self._best_effort_parse_submission(submission)
            except Exception as e:
                bt.logging.error(
                    f"Failed to validate entity ({entity.get('url')}): {traceback.format_exc()}."
                )
                validation_results.append(False)
                continue

            if not content:
                validation_results.append(False)
                continue

            validation_result = self._validate_hf_reddit_content(content, entity)
            validation_results.append(validation_result)

        valid_percentage = sum(validation_results) / len(validation_results) * 100

        # Check if at least 60% of the data is valid
        is_valid = valid_percentage >= 60
        return HFValidationResult(is_valid=is_valid, validation_percentage=valid_percentage, reason=f"Validation Percentage = {valid_percentage}")

    def _validate_hf_reddit_content(self, actual_content: RedditContent, entity_to_validate: dict) -> bool:
        """Validate the Reddit content against the entity to validate, focusing on username, date (hour), and text."""

        # Compare date (year, month, day)
        entity_datetime_str = entity_to_validate.get('datetime')
        if not entity_datetime_str:
            return False
        entity_datetime = dt.datetime.fromisoformat(entity_datetime_str)
        actual_datetime = actual_content.created_at

        if isinstance(actual_datetime, str):
            actual_datetime = dt.datetime.fromisoformat(actual_datetime)

        if (entity_datetime.year != actual_datetime.year or
                entity_datetime.month != actual_datetime.month or
                entity_datetime.day != actual_datetime.day):
            bt.logging.info(f'HF validation failed for {entity_to_validate} due to date mismatch')
            return False

        # Compare text content
        if actual_content.data_type == RedditDataType.POST:
            # For posts, combine title and body
            actual_text = f"{actual_content.title}\n\n{actual_content.body}".strip()
        else:
            actual_text = actual_content.body.strip()

        if actual_text != entity_to_validate.get('text', '').strip():
            bt.logging.info(f'HF validation failed for {entity_to_validate} due to text mismatch')
            return False

        return True

    async def scrape(self, scrape_config: ScrapeConfig) -> List[DataEntity]:
        """Scrapes a batch of reddit posts/comments according to the scrape config."""
        bt.logging.trace(
            f"Reddit custom scraper peforming scrape with config: {scrape_config}."
        )

        assert (
            not scrape_config.labels or len(scrape_config.labels) <= 1
        ), "Can only scrape 1 subreddit at a time."

        # Strip the r/ from the config or use 'all' if no label is provided.
        subreddit_name = (
            normalize_label(scrape_config.labels[0]) if scrape_config.labels else "all"
        )

        bt.logging.trace(
             f"Worker {self.worker_id}: Running custom Reddit scraper with search: {subreddit_name}."
        )

        # Randomize between fetching submissions and comments to reduce api calls.
        fetch_submissions = bool(random.getrandbits(1))

        # Get the search terms for the reddit query.
        search_limit = scrape_config.entity_limit
        search_sort = get_custom_sort_input(scrape_config.date_range.end)
        search_time = get_time_input(scrape_config.date_range.end)

        # In either case we parse the response into a list of RedditContents.
        contents = None
        try:
            reddit = await self._get_reddit_client()
            if not reddit:
                bt.logging.error(f"Worker {self.worker_id}: No available Reddit credentials for scraping")
                return []

            async with reddit:
                subreddit = await reddit.subreddit(subreddit_name)

                if fetch_submissions:
                    submissions = None
                    match search_sort:
                        case "new":
                            submissions = subreddit.new(limit=search_limit)
                        case "top":
                            submissions = subreddit.top(
                                limit=search_limit, time_filter=search_time
                            )
                        case "hot":
                            submissions = subreddit.hot(limit=search_limit)

                    contents = [
                        self._best_effort_parse_submission(submission)
                        async for submission in submissions
                    ]
                else:
                    comments = subreddit.comments(limit=search_limit)

                    contents = [
                        self._best_effort_parse_comment(comment)
                        async for comment in comments
                    ]
        except Exception as e:
            # Check if it's a 403 error (authentication/authorization issue)
            if "403" in str(e) or "Forbidden" in str(e):
                bt.logging.error(
                    f"Worker {self.worker_id}: Authentication failed for subreddit {subreddit_name}. "
                    f"This may indicate invalid credentials or account suspension. Error: {e}"
                )
                # Mark the current credential as failed
                if hasattr(self, '_current_credential') and self._current_credential:
                    self.credential_manager.mark_credential_failed(self._current_credential)
            else:
                bt.logging.error(
                    f"Worker {self.worker_id}: Failed to scrape reddit using subreddit {subreddit_name}, limit {search_limit}, time {search_time}, sort {search_sort}: {str(e)}"
                )
                bt.logging.error(f"Worker {self.worker_id}: Full traceback: {traceback.format_exc()}")
            # TODO: Raise a specific exception, in case the scheduler wants to have some logic for retries.
            return []

        # Return the parsed results, ignoring data that can't be parsed.
        parsed_contents = [content for content in contents if content is not None]

        bt.logging.success(
            f"Worker {self.worker_id}: Completed scrape for subreddit {subreddit_name}. Scraped {len(parsed_contents)} items."
        )

        data_entities = []
        for content in parsed_contents:
            data_entities.append(RedditContent.to_data_entity(content=content))

        return data_entities

    def _best_effort_parse_submission(
        self, submission: asyncpraw.models.Submission
    ) -> RedditContent:
        """Performs a best effort parsing of a Reddit submission into a RedditContent

        Any errors are logged and ignored."""
        content = None

        try:
            # Skip NSFW content
            if getattr(submission, 'over_18', False):
                bt.logging.trace(f"Skipping NSFW submission: {submission.permalink}")
                return None
                
            user = submission.author.name if submission.author else model.DELETED_USER
            content = RedditContent(
                id=submission.name,
                url="https://www.reddit.com"
                + normalize_permalink(submission.permalink),
                username=user,
                communityName=submission.subreddit_name_prefixed,
                body=submission.selftext,
                createdAt=dt.datetime.utcfromtimestamp(submission.created_utc).replace(
                    tzinfo=dt.timezone.utc
                ),
                dataType=RedditDataType.POST,
                # Post only fields
                title=submission.title,
                # Comment only fields
                parentId=None,
            )
        except Exception:
            bt.logging.trace(
                f"Failed to decode RedditContent from Reddit Submission: {traceback.format_exc()}."
            )

        return content

    def _best_effort_parse_comment(
        self, comment: asyncpraw.models.Comment
    ) -> RedditContent:
        """Performs a best effort parsing of a Reddit comment into a RedditContent

        Any errors are logged and ignored."""
        content = None

        try:
            # Skip comments from NSFW submissions or subreddits
            if (getattr(comment.submission, 'over_18', False) or 
                getattr(comment.subreddit, 'over18', False)):
                bt.logging.trace(f"Skipping comment from NSFW submission/subreddit: {comment.permalink}")
                return None
                
            user = comment.author.name if comment.author else model.DELETED_USER
            content = RedditContent(
                id=comment.name,
                url="https://www.reddit.com" + normalize_permalink(comment.permalink),
                username=user,
                communityName=comment.subreddit_name_prefixed,
                body=comment.body,
                createdAt=dt.datetime.utcfromtimestamp(comment.created_utc).replace(
                    tzinfo=dt.timezone.utc
                ),
                dataType=RedditDataType.COMMENT,
                # Post only fields
                title=None,
                # Comment only fields
                parentId=comment.parent_id,
            )
        except Exception:
            bt.logging.trace(
                f"Failed to decode RedditContent from Reddit Submission: {traceback.format_exc()}."
            )

        return content


async def test_scrape():
    scraper = RedditCustomScraper()

    entities = await scraper.scrape(
        ScrapeConfig(
            entity_limit=3,
            date_range=DateRange(
                start=dt.datetime.now(tz=dt.timezone.utc) - dt.timedelta(days=2),
                end=dt.datetime.now(tz=dt.timezone.utc) - dt.timedelta(days=2),
            ),
            labels=[DataLabel(value="r/bittensor_")],
        )
    )

    # Scrape some older content without a label.
    start = dt.datetime.now(tz=dt.timezone.utc) - dt.timedelta(days=2)
    entities = await scraper.scrape(
        ScrapeConfig(
            entity_limit=3,
            date_range=DateRange(
                start=start,
                end=start + dt.timedelta(hours=1),
            ),
        )
    )


async def test_validate():
    scraper = RedditCustomScraper()

    # This test covers a top level comment, a submission, and a nested comment with both the correct parent id and the submission id in order.
    # Previous versions of the custom scraper incorrectly got the submission id as the parent id for nested comments.
    true_entities = [
        DataEntity(
            uri="https://www.reddit.com/r/bittensor_/comments/18bf67l/how_do_you_add_tao_to_metamask/kc3vd3n/",
            datetime=dt.datetime(2023, 12, 5, 16, 29, 27, tzinfo=dt.timezone.utc),
            source=DataSource.REDDIT,
            label=DataLabel(value="r/bittensor_"),
            content=b'{"id": "t1_kc3vd3n", "url": "https://www.reddit.com/r/bittensor_/comments/18bf67l/how_do_you_add_tao_to_metamask/kc3vd3n/", "username": "one-bad-dude", "communityName": "r/bittensor_", "body": "Its not an EVM chain or ERC-20 token. Its a subnet/substrate of Polkadot ecosystem. So you need the polkadot.js wallet.", "createdAt": "2023-12-05T16:29:27+00:00", "dataType": "comment", "title": null, "parentId": "t3_18bf67l"}',
            content_size_bytes=476,
        ),
        DataEntity(
            uri="https://www.reddit.com/r/bittensor_/comments/18bf67l/how_do_you_add_tao_to_metamask/",
            datetime=dt.datetime(2023, 12, 5, 15, 59, 13, tzinfo=dt.timezone.utc),
            source=DataSource.REDDIT,
            label=DataLabel(value="r/bittensor_"),
            content=b'{"id": "t3_18bf67l", "url": "https://www.reddit.com/r/bittensor_/comments/18bf67l/how_do_you_add_tao_to_metamask/", "username": "KOOLBREEZE144", "communityName": "r/bittensor_", "body": "Hey all!!\\n\\nHow do we add TAO to MetaMask? Online gives me these network configurations and still doesn\\u2019t work? \\n\\nHow are you all storing TAO? I wanna purchase on MEXC, but holding off until I can store it!  \\ud83d\\ude11 \\n\\nThanks in advance!!!\\n\\n=====\\n\\nhere is a manual way.\\nNetwork Name\\nTao Network\\n\\nRPC URL\\nhttp://rpc.testnet.tao.network\\n\\nChain ID\\n558\\n\\nCurrency Symbol\\nTAO", "createdAt": "2023-12-05T15:59:13+00:00", "dataType": "post", "title": "How do you add TAO to MetaMask?", "parentId": null}',
            content_size_bytes=775,
        ),
        DataEntity(
            uri="https://www.reddit.com/r/bittensor_/comments/18bf67l/how_do_you_add_tao_to_metamask/kc3w8lk/",
            datetime=dt.datetime(2023, 12, 5, 16, 35, 16, tzinfo=dt.timezone.utc),
            source=DataSource.REDDIT,
            label=DataLabel(value="r/bittensor_"),
            content=b'{"id": "t1_kc3w8lk", "url": "https://www.reddit.com/r/bittensor_/comments/18bf67l/how_do_you_add_tao_to_metamask/kc3w8lk/", "username": "KOOLBREEZE144", "communityName": "r/bittensor_", "body": "Thanks for responding. Do you recommend a wallet or YT video on setting this up? What do you use?", "createdAt": "2023-12-05T16:35:16+00:00", "dataType": "comment", "parentId": "t1_kc3vd3n"}',
            content_size_bytes=392,
        ),
        DataEntity(
            uri="https://www.reddit.com/r/bittensor_/comments/18bf67l/how_do_you_add_tao_to_metamask/kc3w8lk/",
            datetime=dt.datetime(2023, 12, 5, 16, 35, 16, tzinfo=dt.timezone.utc),
            source=DataSource.REDDIT,
            label=DataLabel(value="r/bittensor_"),
            content=b'{"id": "t1_kc3w8lk", "url": "https://www.reddit.com/r/bittensor_/comments/18bf67l/how_do_you_add_tao_to_metamask/kc3w8lk/", "username": "KOOLBREEZE144", "communityName": "r/bittensor_", "body": "Thanks for responding. Do you recommend a wallet or YT video on setting this up? What do you use?", "createdAt": "2023-12-05T16:35:16+00:00", "dataType": "comment", "parentId": "t3_18bf67l"}',
            content_size_bytes=392,
        ),
    ]
    results = await scraper.validate(entities=true_entities)
    print(f"Expecting Pass. Validation results: {results}")

    # Now modify the entities to make them invalid and check validation fails.
    good_entity = true_entities[1]
    good_comment_entity = true_entities[2]
    bad_entities = [
        # Change url.
        good_entity.copy(
            update={
                "uri": "https://www.reddit.com/r/bittensor_/comments/18bf67l/how_do_you_add_tao_to_metamask-abc123/"
            }
        ),
        # Change title.
        good_entity.copy(
            update={
                "content": b'{"id": "t3_18bf67l", "url": "https://www.reddit.com/r/bittensor_/comments/18bf67l/how_do_you_add_tao_to_metamask/", "username": "KOOLBREEZE144", "communityName": "r/bittensor_", "body": "Hey all!!\\n\\nHow do we add TAO to MetaMask? Online gives me these network configurations and still doesn\\u2019t work? \\n\\nHow are you all storing TAO? I wanna purchase on MEXC, but holding off until I can store it!  \\ud83d\\ude11 \\n\\nThanks in advance!!!\\n\\n=====\\n\\nhere is a manual way.\\nNetwork Name\\nTao Network\\n\\nRPC URL\\nhttp://rpc.testnet.tao.network\\n\\nChain ID\\n558\\n\\nCurrency Symbol\\nTAO", "createdAt": "2023-12-05T15:59:13+00:00", "dataType": "post", "title": "How do you add TAO to MetaMask??!!?", "parent_id": null}',
            }
        ),
        # Change created_at.
        good_entity.copy(
            update={"datetime": good_entity.datetime + dt.timedelta(seconds=1)}
        ),
        # Change label.
        good_entity.copy(update={"label": DataLabel(value="bittensor_")}),
        # Change comment parent id.
        good_comment_entity.copy(
            update={
                "content": b'{"id": "t1_kc3w8lk", "url": "https://www.reddit.com/r/bittensor_/comments/18bf67l/how_do_you_add_tao_to_metamask/kc3w8lk/", "username": "KOOLBREEZE144", "communityName": "r/bittensor_", "body": "Thanks for responding. Do you recommend a wallet or YT video on setting this up? What do you use?", "createdAt": "2023-12-05T16:35:16+00:00", "dataType": "comment", "parentId": "extra-long-parent-id"}'
            }
        ),
        # Change submission parent id.
        good_entity.copy(
            update={
                "content": b'{"id": "t3_18bf67l", "url": "https://www.reddit.com/r/bittensor_/comments/18bf67l/how_do_you_add_tao_to_metamask/", "username": "KOOLBREEZE144", "communityName": "r/bittensor_", "body": "Hey all!!\\n\\nHow do we add TAO to MetaMask? Online gives me these network configurations and still doesn\\u2019t work? \\n\\nHow are you all storing TAO? I wanna purchase on MEXC, but holding off until I can store it!  \\ud83d\\ude11 \\n\\nThanks in advance!!!\\n\\n=====\\n\\nhere is a manual way.\\nNetwork Name\\nTao Network\\n\\nRPC URL\\nhttp://rpc.testnet.tao.network\\n\\nChain ID\\n558\\n\\nCurrency Symbol\\nTAO", "createdAt": "2023-12-05T15:59:13+00:00", "dataType": "post", "title": "How do you add TAO to MetaMask?", "parentId": "extra-long-parent-id"}'
            }
        ),
    ]

    for entity in bad_entities:
        results = await scraper.validate(entities=[entity])
        print(f"Expecting a failed validation. Result={results}")


async def test_u_deleted():
    """Verifies that the RedditCustomScraper can handle deleted users."""
    comment = DataEntity(
        uri="https://www.reddit.com/r/AskReddit/comments/ablzuq/people_who_havent_pooped_in_2019_yet_why_are_you/ed1j7is/",
        datetime=dt.datetime(2019, 1, 1, 22, 59, 9, tzinfo=dt.timezone.utc),
        source=1,
        label=DataLabel(value="r/askreddit"),
        content=b'{"id": "t1_ed1j7is", "url": "https://www.reddit.com/r/AskReddit/comments/ablzuq/people_who_havent_pooped_in_2019_yet_why_are_you/ed1j7is/", "username": "[deleted]", "communityName": "r/AskReddit", "body": "Aw man what a terrible way to spend NYE! I hope you feel better soon bud!", "createdAt": "2019-01-01T22:59:09+00:00", "dataType": "comment", "title": null, "parentId": "t1_ed1dqvy"}',
        content_size_bytes=387,
    )

    scraper = RedditCustomScraper()
    result = await scraper.validate(entities=[comment])
    print(f"Expecting a passed validation: {result}")


if __name__ == "__main__":
    asyncio.run(test_scrape())
    asyncio.run(test_validate())
    asyncio.run(test_u_deleted())
