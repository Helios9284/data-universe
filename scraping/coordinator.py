import asyncio
import functools
import random
import threading
import traceback
import hashlib
import bittensor as bt
import datetime as dt
from typing import Dict, List, Optional
import numpy
from pydantic import Field, PositiveInt, ConfigDict
from common.date_range import DateRange
from common.data import DataEntity, DataLabel, DataSource, StrictBaseModel, TimeBucket
from scraping.provider import ScraperProvider
from scraping.scraper import ScrapeConfig, ScraperId
from storage.miner.miner_storage import MinerStorage
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
import os
import psutil

# Global cache for desirability data (using validator's system)
_desirability_cache = None
_desirability_cache_time = None
_desirability_cache_duration = dt.timedelta(minutes=30)  # Cache for 30 minutes


class LabelScrapingConfig(StrictBaseModel):
    """Describes what labels to scrape."""

    label_choices: Optional[List[DataLabel]] = Field(
        description="""The collection of labels to choose from when performing a scrape.
        On a given scrape, 1 label will be chosen at random from this list.

        If the list is None, the scraper will scrape "all".
        """
    )

    max_age_hint_minutes: int = Field(
        description="""The maximum age of data that this scrape should fetch. A random TimeBucket (currently hour block),
        will be chosen within the time frame (now - max_age_hint_minutes, now), using a probality distribution aligned
        with how validators score data freshness.

        Note: not all data sources provide date filters, so this property should be thought of as a hint to the scraper, not a rule.
        """,
    )

    max_data_entities: Optional[PositiveInt] = Field(
        default=None,
        description="The maximum number of items to fetch in a single scrape for this label. If None, the scraper will fetch as many items possible.",
    )


class ScraperConfig(StrictBaseModel):
    """Describes what to scrape for a Scraper."""

    cadence_seconds: PositiveInt = Field(
        description="Configures how often to scrape with this scraper, measured in seconds."
    )

    labels_to_scrape: List[LabelScrapingConfig] = Field(
        description="""Describes the type of data to scrape with this scraper.

        The scraper will perform one scrape per entry in this list every 'cadence_seconds'.
        """
    )


class CoordinatorConfig(StrictBaseModel):
    """Informs the Coordinator how to schedule scrapes."""

    scraper_configs: Dict[ScraperId, ScraperConfig] = Field(
        description="The configs for each scraper."
    )


async def _choose_scrape_configs(
        scraper_id: ScraperId, config: CoordinatorConfig, now: dt.datetime
) -> List[ScrapeConfig]:
    """For the given scraper, returns a list of scrapes (defined by ScrapeConfig) to be run."""
    assert (
            scraper_id in config.scraper_configs
    ), f"Scraper Id {scraper_id} not in config"

    # Ensure now has timezone information
    if now.tzinfo is None:
        now = now.replace(tzinfo=dt.timezone.utc)

    scraper_config = config.scraper_configs[scraper_id]
    results = []

    for label_config in scraper_config.labels_to_scrape:
        # Get dynamic labels based on validator preferences
        labels_to_scrape = await _get_dynamic_labels(scraper_id, label_config, now)

        # Get max age from config or use default
        max_age_minutes = label_config.max_age_hint_minutes

        # For YouTube transcript scraper, use a wider date range
        if scraper_id == ScraperId.YOUTUBE_CUSTOM_TRANSCRIPT:
            # Calculate the start time using max_age_minutes
            start_time = now - dt.timedelta(minutes=max_age_minutes)

            # Ensure start_time has timezone information
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=dt.timezone.utc)

            date_range = DateRange(start=start_time, end=now)

            bt.logging.info(f"Created special date range for YouTube: {date_range.start} to {date_range.end}")
            bt.logging.info(f"Date range span: {(date_range.end - date_range.start).total_seconds() / 3600} hours")

            results.append(
                ScrapeConfig(
                    entity_limit=label_config.max_data_entities,
                    date_range=date_range,
                    labels=labels_to_scrape,
                )
            )
        else:
            # For other scrapers, use the normal time bucket approach
            current_bucket = TimeBucket.from_datetime(now)
            oldest_bucket = TimeBucket.from_datetime(
                now - dt.timedelta(minutes=max_age_minutes)
            )

            chosen_bucket = current_bucket
            # If we have more than 1 bucket to choose from, choose a bucket in the range
            if oldest_bucket.id < current_bucket.id:
                # Use a triangular distribution for bucket selection
                chosen_id = int(numpy.random.default_rng().triangular(
                    left=oldest_bucket.id, mode=current_bucket.id, right=current_bucket.id
                ))

                chosen_bucket = TimeBucket(id=chosen_id)

            date_range = TimeBucket.to_date_range(chosen_bucket)

            # Ensure date_range has timezone info
            if date_range.start.tzinfo is None:
                date_range = DateRange(
                    start=date_range.start.replace(tzinfo=dt.timezone.utc),
                    end=date_range.end.replace(tzinfo=dt.timezone.utc)
                )

            results.append(
                ScrapeConfig(
                    entity_limit=label_config.max_data_entities,
                    date_range=date_range,
                    labels=labels_to_scrape,
                )
            )

    return results


async def _get_dynamic_labels(scraper_id: ScraperId, label_config, now: dt.datetime) -> List[DataLabel]:
    """Get dynamic labels based on validator preferences using the same system as validators."""
    
    # Get desirability lookup from validator system
    desirability_lookup = await _get_cached_desirability_lookup()
    
    if not desirability_lookup:
        bt.logging.warning("No desirability data available, using smart fallback keywords")
        return _get_smart_fallback_labels(scraper_id, label_config)
    
    # Extract labels from validator preferences (same system as validators use)
    dynamic_labels = []
    try:
        # Get the appropriate data source for this scraper
        data_source = _get_data_source_for_scraper(scraper_id)
        if data_source and data_source in desirability_lookup.distribution:
            source_desirability = desirability_lookup.distribution[data_source]
            if source_desirability.job_matcher and source_desirability.job_matcher.jobs:
                # Use the same job matching system as validators
                for job in source_desirability.job_matcher.jobs:
                    if job.label:
                        # Convert to DataLabel format
                        label = DataLabel(value=job.label)
                        # Weight by job_weight (same as validator scoring)
                        dynamic_labels.extend([label] * max(1, int(job.job_weight * 10)))
                        
                        # Also add keyword if present
                        if job.keyword:
                            keyword_label = DataLabel(value=job.keyword)
                            dynamic_labels.extend([keyword_label] * max(1, int(job.job_weight * 8)))
                
                bt.logging.info(f"Extracted {len(dynamic_labels)} labels from validator preferences for {data_source.name}")
    except Exception as e:
        bt.logging.warning(f"Error extracting validator labels: {e}")
    
    # If no dynamic labels found, use smart fallback
    if not dynamic_labels:
        bt.logging.warning(f"No dynamic labels found for {scraper_id}, using smart fallback")
        return _get_smart_fallback_labels(scraper_id, label_config)
    
    # Shuffle to avoid bias
    import random
    random.shuffle(dynamic_labels)
    
    # For Reddit, only return 1 label (scraper limitation)
    if scraper_id == ScraperId.REDDIT_CUSTOM:
        if dynamic_labels:
            selected_label = dynamic_labels[0]  # Take the first one after shuffling
            bt.logging.info(f"Reddit limitation: Using 1 label '{selected_label.value}' from {len(dynamic_labels)} available")
            return [selected_label]
        else:
            return []
    
    # For other scrapers, limit to reasonable number of labels
    max_labels = 50
    if len(dynamic_labels) > max_labels:
        dynamic_labels = dynamic_labels[:max_labels]
    
    bt.logging.info(f"Generated {len(dynamic_labels)} dynamic labels for {scraper_id} using validator system")
    return dynamic_labels


async def _get_cached_desirability_lookup():
    """Get desirability lookup with caching to avoid frequent API calls."""
    global _desirability_cache, _desirability_cache_time, _desirability_cache_duration
    
    now = dt.datetime.utcnow()
    
    # Check if cache is valid
    if (_desirability_cache is not None and 
        _desirability_cache_time is not None and 
        now - _desirability_cache_time < _desirability_cache_duration):
        bt.logging.trace("Using cached desirability data")
        return _desirability_cache
    
    # Fetch fresh data using the same system as validators
    try:
        from dynamic_desirability.desirability_retrieval import run_retrieval
        from neurons.config import create_config, NeuronType
        
        # Create a minimal config for desirability retrieval (same as validator)
        config = create_config(NeuronType.MINER)
        # Use the async version directly instead of sync_run_retrieval
        desirability_lookup = await run_retrieval(config)
        
        if desirability_lookup:
            # Update cache
            _desirability_cache = desirability_lookup
            _desirability_cache_time = now
            bt.logging.info("Updated desirability cache with fresh data from validator system")
            return desirability_lookup
        
    except Exception as e:
        bt.logging.warning(f"Failed to get desirability data from validator system: {e}")
    
    return None


def _get_static_labels(scraper_id: ScraperId, label_config) -> List[DataLabel]:
    """Get static labels from the original configuration."""
    if label_config.label_choices:
        return label_config.label_choices
    return []


def _get_smart_fallback_labels(scraper_id: ScraperId, label_config) -> List[DataLabel]:
    """Get fallback labels from scraping config when no validator preferences are available."""
    
    # Use static labels from config file
    static_labels = _get_static_labels(scraper_id, label_config)
    if static_labels:
        bt.logging.info(f"Using static labels from scraping config for {scraper_id}")
        
        # For Reddit, only return 1 label (scraper limitation)
        if scraper_id == ScraperId.REDDIT_CUSTOM:
            if static_labels:
                selected_label = static_labels[0]  # Take the first one
                bt.logging.info(f"Reddit fallback: Using 1 label '{selected_label.value}' from config")
                return [selected_label]
            else:
                return []
        
        return static_labels
    
    # If no static labels in config, return empty list
    bt.logging.warning(f"No static labels found in config for {scraper_id}, no fallback available")
    return []


def _get_data_source_for_scraper(scraper_id: ScraperId):
    """Map scraper IDs to data sources."""
    from common.data import DataSource
    
    source_mapping = {
        ScraperId.REDDIT_CUSTOM: DataSource.REDDIT,
        ScraperId.X_APIDOJO: DataSource.X,
        ScraperId.YOUTUBE_CUSTOM_TRANSCRIPT: DataSource.YOUTUBE,
    }
    
    return source_mapping.get(scraper_id)



class ParallelScraperCoordinator:
    """Enhanced coordinator that supports parallel processing with dedicated CPU allocation."""

    class Tracker:
        """Tracks scrape runs for the coordinator."""

        def __init__(self, config: CoordinatorConfig, now: dt.datetime):
            self.cadence_by_scraper_id = {
                scraper_id: dt.timedelta(seconds=cfg.cadence_seconds)
                for scraper_id, cfg in config.scraper_configs.items()
            }

            # Initialize the last scrape time as now, to protect against frequent scraping during Miner crash loops.
            self.last_scrape_time_per_scraper_id: Dict[ScraperId, dt.datetime] = {
                scraper_id: now for scraper_id in config.scraper_configs.keys()
            }

        def get_scraper_ids_ready_to_scrape(self, now: dt.datetime) -> List[ScraperId]:
            """Returns a list of ScraperIds which are due to run."""
            results = []
            for scraper_id, cadence in self.cadence_by_scraper_id.items():
                last_scrape_time = self.last_scrape_time_per_scraper_id.get(
                    scraper_id, None
                )
                if last_scrape_time is None or now - last_scrape_time >= cadence:
                    results.append(scraper_id)
            return results

        def on_scrape_scheduled(self, scraper_id: ScraperId, now: dt.datetime):
            """Notifies the tracker that a scrape has been scheduled."""
            self.last_scrape_time_per_scraper_id[scraper_id] = now

    def __init__(
        self,
        scraper_provider: ScraperProvider,
        miner_storage: MinerStorage,
        config: CoordinatorConfig,
        enable_parallel_processing: bool = True,
        cpu_allocation: Optional[Dict[str, int]] = None,
    ):
        self.provider = scraper_provider
        self.storage = miner_storage
        self.config = config
        self.enable_parallel_processing = enable_parallel_processing
        
        # CPU allocation for different scrapers
        self.cpu_allocation = cpu_allocation or {
            "X.apidojo": 1,           # Twitter gets 1 CPU
            "Reddit.custom": 2,       # Reddit gets 2 CPUs (for 2 APIs)
            "YouTube.custom.transcript": 1  # YouTube gets 1 CPU
        }
        
        # Calculate total workers based on CPU allocation
        self.total_workers = sum(self.cpu_allocation.values())
        self.max_workers = min(self.total_workers, multiprocessing.cpu_count())
        
        # Create separate queues for different scrapers
        self.twitter_queue = asyncio.Queue()
        self.reddit_queue_1 = asyncio.Queue()  # For Reddit API 1
        self.reddit_queue_2 = asyncio.Queue()  # For Reddit API 2
        self.youtube_queue = asyncio.Queue()
        
        # Thread pools for CPU-intensive tasks
        self.twitter_pool = ThreadPoolExecutor(max_workers=self.cpu_allocation.get("X.apidojo", 1), 
                                             thread_name_prefix="twitter-worker")
        self.reddit_pool_1 = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reddit-worker-1")
        self.reddit_pool_2 = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reddit-worker-2")
        self.youtube_pool = ThreadPoolExecutor(max_workers=self.cpu_allocation.get("YouTube.custom.transcript", 1), 
                                             thread_name_prefix="youtube-worker")

        self.tracker = ParallelScraperCoordinator.Tracker(self.config, dt.datetime.utcnow())
        self.is_running = False
        self.workers = []

    def run_in_background_thread(self):
        """
        Runs the Coordinator on a background thread. The coordinator will run until the process dies.
        """
        assert not self.is_running, "ParallelScrapingCoordinator already running"

        bt.logging.info(f"Starting ParallelScrapingCoordinator with {self.max_workers} workers in a background thread.")
        bt.logging.info(f"CPU Allocation: {self.cpu_allocation}")

        self.is_running = True
        self.thread = threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        """Blocking call to run the Coordinator, indefinitely."""
        asyncio.run(self._start())

    def stop(self):
        bt.logging.info("Stopping the ParallelScrapingCoordinator.")
        self.is_running = False
        
        # Shutdown thread pools
        self.twitter_pool.shutdown(wait=True)
        self.reddit_pool_1.shutdown(wait=True)
        self.reddit_pool_2.shutdown(wait=True)
        self.youtube_pool.shutdown(wait=True)

    async def _start(self):
        """Start parallel processing with dedicated workers for each scraper type."""
        
        # Start dedicated workers for each scraper type
        workers = []
        
        # Twitter worker
        if self.cpu_allocation.get("X.apidojo", 0) > 0:
            twitter_worker = asyncio.create_task(
                self._twitter_worker("twitter-worker")
            )
            workers.append(twitter_worker)
        
        # Reddit workers (2 separate workers for 2 APIs)
        if self.cpu_allocation.get("Reddit.custom", 0) > 0:
            bt.logging.info("Starting Reddit workers: reddit-worker-1 and reddit-worker-2")
            reddit_worker_1 = asyncio.create_task(
                self._reddit_worker("reddit-worker-1", self.reddit_queue_1, "REDDIT_API_1")
            )
            reddit_worker_2 = asyncio.create_task(
                self._reddit_worker("reddit-worker-2", self.reddit_queue_2, "REDDIT_API_2")
            )
            workers.extend([reddit_worker_1, reddit_worker_2])
        
        # YouTube worker
        if self.cpu_allocation.get("YouTube.custom.transcript", 0) > 0:
            youtube_worker = asyncio.create_task(
                self._youtube_worker("youtube-worker")
            )
            workers.append(youtube_worker)

        # Main coordinator loop
        while self.is_running:
            now = dt.datetime.utcnow()
            scraper_ids_to_scrape_now = self.tracker.get_scraper_ids_ready_to_scrape(now)
            
            if not scraper_ids_to_scrape_now:
                bt.logging.trace("Nothing ready to scrape yet. Trying again in 15s.")
                await asyncio.sleep(15)
                continue

            for scraper_id in scraper_ids_to_scrape_now:
                scraper = self.provider.get(scraper_id)
                scrape_configs = await _choose_scrape_configs(scraper_id, self.config, now)

                for config in scrape_configs:
                    # Route to appropriate queue based on scraper type
                    if scraper_id == "X.apidojo":
                        bt.logging.trace(f"Adding Twitter scrape task: {config}")
                        self.twitter_queue.put_nowait(functools.partial(scraper.scrape, config))
                    
                    elif scraper_id == "Reddit.custom":
                        # For Reddit, ensure both workers get tasks by creating multiple configs
                        # Get the original scrape configs
                        original_configs = await _choose_scrape_configs(scraper_id, self.config, now)
                        
                        # Create additional configs to ensure both workers have tasks
                        if len(original_configs) == 1:
                            # If only one config, create a second one with a different label
                            first_config = original_configs[0]
                            
                            # Create a second config with a different random label
                            from common.data import DataLabel
                            import random
                            
                            # Get all available labels from the config
                            scraper_config = self.config.scraper_configs[scraper_id]
                            all_labels = []
                            for label_config in scraper_config.labels_to_scrape:
                                if label_config.label_choices:
                                    all_labels.extend(label_config.label_choices)
                            
                            # Choose a different label for the second config
                            if len(all_labels) > 1 and first_config.labels:
                                different_labels = [l for l in all_labels if l.value != first_config.labels[0].value]
                                if different_labels:
                                    second_label = random.choice(different_labels)
                                    second_config = ScrapeConfig(
                                        entity_limit=first_config.entity_limit,
                                        date_range=first_config.date_range,
                                        labels=[DataLabel(value=second_label.value)]
                                    )
                                else:
                                    # If no different labels, use the same config
                                    second_config = first_config
                            else:
                                # If no labels or only one label, use the same config
                                second_config = first_config
                            
                            bt.logging.info(f"Reddit distribution: API 1 gets label {first_config.labels[0].value if first_config.labels else 'None'}")
                            bt.logging.info(f"Reddit distribution: API 2 gets label {second_config.labels[0].value if second_config.labels else 'None'}")
                            self.reddit_queue_1.put_nowait(first_config)
                            self.reddit_queue_2.put_nowait(second_config)
                            bt.logging.info(f"Queue sizes after distribution: API 1={self.reddit_queue_1.qsize()}, API 2={self.reddit_queue_2.qsize()}")
                        else:
                            # If we have multiple configs, distribute them
                            for i, config in enumerate(original_configs):
                                if i == 0:
                                    bt.logging.trace(f"Adding Reddit scrape task to API 1: {config}")
                                    self.reddit_queue_1.put_nowait(config)
                                elif i == 1:
                                    bt.logging.trace(f"Adding Reddit scrape task to API 2: {config}")
                                    self.reddit_queue_2.put_nowait(config)
                                else:
                                    # For additional configs, alternate
                                    if self.reddit_queue_1.qsize() <= self.reddit_queue_2.qsize():
                                        bt.logging.trace(f"Adding Reddit scrape task to API 1: {config}")
                                        self.reddit_queue_1.put_nowait(config)
                                    else:
                                        bt.logging.trace(f"Adding Reddit scrape task to API 2: {config}")
                                        self.reddit_queue_2.put_nowait(config)
                    
                    elif scraper_id == "YouTube.custom.transcript":
                        bt.logging.trace(f"Adding YouTube scrape task: {config}")
                        self.youtube_queue.put_nowait(functools.partial(scraper.scrape, config))

                self.tracker.on_scrape_scheduled(scraper_id, now)

        bt.logging.info("Coordinator shutting down. Waiting for workers to finish.")
        await asyncio.gather(*workers)
        bt.logging.info("Coordinator stopped.")

    async def _twitter_worker(self, name):
        """Dedicated Twitter worker using thread pool."""
        bt.logging.info(f"Starting {name} with thread pool")
        
        while self.is_running:
            try:
                # Wait for Twitter scraping task
                scrape_fn = await asyncio.wait_for(self.twitter_queue.get(), timeout=30.0)
                
                # Execute in thread pool for CPU-intensive processing
                loop = asyncio.get_event_loop()
                data_entities = await loop.run_in_executor(
                    self.twitter_pool, 
                    lambda: asyncio.run(scrape_fn())
                )

                # Apply deduplication and store
                deduplicated_entities = self._cross_bucket_deduplication(data_entities)
                self.storage.store_data_entities(deduplicated_entities)
                
                # Note: Performance tracking removed - using validator's dynamic desirability system instead
                
                self.twitter_queue.task_done()
                bt.logging.info(f"{name}: Processed {len(deduplicated_entities)} Twitter entities")
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                bt.logging.error(f"{name} error: {traceback.format_exc()}")

    async def _reddit_worker(self, name, queue, api_identifier):
        """Dedicated Reddit worker for specific API."""
        bt.logging.info(f"Starting {name} for {api_identifier}")
        
        # Create a dedicated scraper instance for this worker to ensure proper credential isolation
        # Import the scraper class directly to create worker-specific instances
        from scraping.reddit.reddit_custom_scraper import RedditCustomScraper
        worker_scraper = RedditCustomScraper(worker_id=api_identifier)
        
        while self.is_running:
            try:
                # Wait for Reddit scraping task
                scrape_config = await asyncio.wait_for(queue.get(), timeout=30.0)
                
                # Execute in dedicated thread pool with the worker's scraper instance
                loop = asyncio.get_event_loop()
                data_entities = await loop.run_in_executor(
                    self.reddit_pool_1 if api_identifier == "REDDIT_API_1" else self.reddit_pool_2,
                    lambda: asyncio.run(worker_scraper.scrape(scrape_config))
                )

                # Apply deduplication and store
                deduplicated_entities = self._cross_bucket_deduplication(data_entities)
                self.storage.store_data_entities(deduplicated_entities)
                
                # Note: Performance tracking removed - using validator's dynamic desirability system instead
                
                queue.task_done()
                bt.logging.info(f"{name}: Processed {len(deduplicated_entities)} Reddit entities using {api_identifier}")
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                bt.logging.error(f"{name} error: {traceback.format_exc()}")

    async def _youtube_worker(self, name):
        """Dedicated YouTube worker using thread pool."""
        bt.logging.info(f"Starting {name} with thread pool")
        
        while self.is_running:
            try:
                # Wait for YouTube scraping task
                scrape_fn = await asyncio.wait_for(self.youtube_queue.get(), timeout=30.0)
                
                # Execute in thread pool for CPU-intensive processing
                loop = asyncio.get_event_loop()
                data_entities = await loop.run_in_executor(
                    self.youtube_pool, 
                    lambda: asyncio.run(scrape_fn())
                )

                # Apply deduplication and store
                deduplicated_entities = self._cross_bucket_deduplication(data_entities)
                self.storage.store_data_entities(deduplicated_entities)
                
                # Note: Performance tracking removed - using validator's dynamic desirability system instead
                
                self.youtube_queue.task_done()
                bt.logging.info(f"{name}: Processed {len(deduplicated_entities)} YouTube entities")
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                bt.logging.error(f"{name} error: {traceback.format_exc()}")

    def _cross_bucket_deduplication(self, data_entities: List[DataEntity]) -> List[DataEntity]:
        """Performs cross-bucket deduplication by checking against existing data."""
        if not data_entities:
            return []

        # Create sets to track seen content hashes and URIs within this batch
        seen_content_hashes = set()
        seen_uris = set()
        deduplicated_entities = []

        for entity in data_entities:
            # Calculate content hash
            content_hash = hashlib.sha1(entity.content).hexdigest()
            
            # Normalize URI (handle twitter.com vs x.com differences)
            normalized_uri = self._normalize_uri(entity.uri)
            
            # Check if this entity is a duplicate within the current batch
            if content_hash in seen_content_hashes or normalized_uri in seen_uris:
                bt.logging.trace(f"Cross-batch deduplication: skipping duplicate entity {entity.uri}")
                continue
            
            # Add to tracking sets
            seen_content_hashes.add(content_hash)
            seen_uris.add(normalized_uri)
            deduplicated_entities.append(entity)

        if len(deduplicated_entities) < len(data_entities):
            bt.logging.info(
                f"Cross-batch deduplication removed {len(data_entities) - len(deduplicated_entities)} entities. "
                f"Keeping {len(deduplicated_entities)} out of {len(data_entities)} entities."
            )

        return deduplicated_entities

    def _normalize_uri(self, uri: str) -> str:
        """Normalizes a URI for deduplication purposes."""
        # Handle twitter.com vs x.com differences
        if "twitter.com" in uri:
            return uri.replace("twitter.com", "x.com")
        elif "x.com" in uri:
            return uri.replace("x.com", "twitter.com")
        return uri

# Keep the original ScraperCoordinator for backward compatibility
class ScraperCoordinator:
    """Coordinates all the scrapers necessary based on the specified target ScrapingDistribution."""

    class Tracker:
        """Tracks scrape runs for the coordinator."""

        def __init__(self, config: CoordinatorConfig, now: dt.datetime):
            self.cadence_by_scraper_id = {
                scraper_id: dt.timedelta(seconds=cfg.cadence_seconds)
                for scraper_id, cfg in config.scraper_configs.items()
            }

            # Initialize the last scrape time as now, to protect against frequent scraping during Miner crash loops.
            self.last_scrape_time_per_scraper_id: Dict[ScraperId, dt.datetime] = {
                scraper_id: now for scraper_id in config.scraper_configs.keys()
            }

        def get_scraper_ids_ready_to_scrape(self, now: dt.datetime) -> List[ScraperId]:
            """Returns a list of ScraperIds which are due to run."""
            results = []
            for scraper_id, cadence in self.cadence_by_scraper_id.items():
                last_scrape_time = self.last_scrape_time_per_scraper_id.get(
                    scraper_id, None
                )
                if last_scrape_time is None or now - last_scrape_time >= cadence:
                    results.append(scraper_id)
            return results

        def on_scrape_scheduled(self, scraper_id: ScraperId, now: dt.datetime):
            """Notifies the tracker that a scrape has been scheduled."""
            self.last_scrape_time_per_scraper_id[scraper_id] = now

    def __init__(
        self,
        scraper_provider: ScraperProvider,
        miner_storage: MinerStorage,
        config: CoordinatorConfig,
    ):
        self.provider = scraper_provider
        self.storage = miner_storage
        self.config = config

        self.tracker = ScraperCoordinator.Tracker(self.config, dt.datetime.utcnow())
        self.max_workers = 5
        self.is_running = False
        self.queue = asyncio.Queue()

    def run_in_background_thread(self):
        """
        Runs the Coordinator on a background thread. The coordinator will run until the process dies.
        """
        assert not self.is_running, "ScrapingCoordinator already running"

        bt.logging.info("Starting ScrapingCoordinator in a background thread.")

        self.is_running = True
        self.thread = threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        """Blocking call to run the Coordinator, indefinitely."""
        asyncio.run(self._start())

    def stop(self):
        bt.logging.info("Stopping the ScrapingCoordinator.")
        self.is_running = False

    async def _start(self):
        workers = []
        for i in range(self.max_workers):
            worker = asyncio.create_task(
                self._worker(
                    f"worker-{i}",
                )
            )
            workers.append(worker)

        while self.is_running:
            now = dt.datetime.utcnow()
            scraper_ids_to_scrape_now = self.tracker.get_scraper_ids_ready_to_scrape(
                now
            )
            if not scraper_ids_to_scrape_now:
                bt.logging.trace("Nothing ready to scrape yet. Trying again in 15s.")
                # Nothing is due a scrape. Wait a few seconds and try again
                await asyncio.sleep(15)
                continue

            for scraper_id in scraper_ids_to_scrape_now:
                scraper = self.provider.get(scraper_id)

                scrape_configs = await _choose_scrape_configs(scraper_id, self.config, now)

                for config in scrape_configs:
                    # Use .partial here to make sure the functions arguments are copied/stored
                    # now rather than being lazily evaluated (if a lambda was used).
                    # https://pylint.readthedocs.io/en/latest/user_guide/messages/warning/cell-var-from-loop.html#cell-var-from-loop-w0640
                    bt.logging.trace(f"Adding scrape task for {scraper_id}: {config}.")
                    self.queue.put_nowait(functools.partial(scraper.scrape, config))

                self.tracker.on_scrape_scheduled(scraper_id, now)

        bt.logging.info("Coordinator shutting down. Waiting for workers to finish.")
        await asyncio.gather(*workers)
        bt.logging.info("Coordinator stopped.")

    async def _worker(self, name):
        """A worker thread"""
        while self.is_running:
            try:
                # Wait for a scraping task to be added to the queue.
                scrape_fn = await self.queue.get()

                # Perform the scrape
                data_entities = await scrape_fn()

                # Apply cross-bucket deduplication before storage
                deduplicated_entities = self._cross_bucket_deduplication(data_entities)

                self.storage.store_data_entities(deduplicated_entities)
                self.queue.task_done()
            except Exception as e:
                bt.logging.error("Worker " + name + ": " + traceback.format_exc())

    def _cross_bucket_deduplication(self, data_entities: List[DataEntity]) -> List[DataEntity]:
        """Performs cross-bucket deduplication by checking against existing data."""
        if not data_entities:
            return []

        # For now, we'll do basic deduplication within the current batch
        # Full cross-bucket deduplication would require more complex storage queries
        # that might impact performance significantly
        
        # Create sets to track seen content hashes and URIs within this batch
        seen_content_hashes = set()
        seen_uris = set()
        deduplicated_entities = []

        for entity in data_entities:
            # Calculate content hash
            content_hash = hashlib.sha1(entity.content).hexdigest()
            
            # Normalize URI (handle twitter.com vs x.com differences)
            normalized_uri = self._normalize_uri(entity.uri)
            
            # Check if this entity is a duplicate within the current batch
            if content_hash in seen_content_hashes or normalized_uri in seen_uris:
                bt.logging.trace(f"Cross-batch deduplication: skipping duplicate entity {entity.uri}")
                continue
            
            # Add to tracking sets
            seen_content_hashes.add(content_hash)
            seen_uris.add(normalized_uri)
            deduplicated_entities.append(entity)

        if len(deduplicated_entities) < len(data_entities):
            bt.logging.info(
                f"Cross-batch deduplication removed {len(data_entities) - len(deduplicated_entities)} entities. "
                f"Keeping {len(deduplicated_entities)} out of {len(data_entities)} entities."
            )

        return deduplicated_entities

    def _normalize_uri(self, uri: str) -> str:
        """Normalizes a URI for deduplication purposes."""
        # Handle twitter.com vs x.com differences
        if "twitter.com" in uri:
            return uri.replace("twitter.com", "x.com")
        elif "x.com" in uri:
            return uri.replace("x.com", "twitter.com")
        return uri
