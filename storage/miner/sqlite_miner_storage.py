from collections import defaultdict
import threading
import hashlib
import traceback
from common import constants, utils
from common.data import (
    CompressedEntityBucket,
    CompressedMinerIndex,
    DataEntity,
    DataEntityBucket,
    DataEntityBucketId,
    DataLabel,
    DataSource,
    TimeBucket,
    HuggingFaceMetadata,
)
from storage.miner.miner_storage import MinerStorage
from typing import Dict, List
import datetime as dt
import sqlite3
import contextlib
import bittensor as bt
import pandas as pd


# Use a timezone aware adapter for timestamp columns.
def tz_aware_timestamp_adapter(val):
    datepart, timepart = val.split(b" ")
    year, month, day = map(int, datepart.split(b"-"))

    if b"+" in timepart:
        timepart, tz_offset = timepart.rsplit(b"+", 1)
        if tz_offset == b"00:00":
            tzinfo = dt.timezone.utc
        else:
            hours, minutes = map(int, tz_offset.split(b":", 1))
            tzinfo = dt.timezone(dt.timedelta(hours=hours, minutes=minutes))
    elif b"-" in timepart:
        timepart, tz_offset = timepart.rsplit(b"-", 1)
        if tz_offset == b"00:00":
            tzinfo = dt.timezone.utc
        else:
            hours, minutes = map(int, tz_offset.split(b":", 1))
            tzinfo = dt.timezone(dt.timedelta(hours=-hours, minutes=-minutes))
    else:
        tzinfo = None

    timepart_full = timepart.split(b".")
    hours, minutes, seconds = map(int, timepart_full[0].split(b":"))

    if len(timepart_full) == 2:
        microseconds = int("{:0<6.6}".format(timepart_full[1].decode()))
    else:
        microseconds = 0

    val = dt.datetime(year, month, day, hours, minutes, seconds, microseconds, tzinfo)

    return val


class SqliteMinerStorage(MinerStorage):
    """Sqlite backed MinerStorage"""

    # TODO Consider CHECK expression to limit source to expected ENUM values.
    # Sqlite type converters handle the mapping from Python datetime to Timestamp.
    DATA_ENTITY_TABLE_CREATE = """CREATE TABLE IF NOT EXISTS DataEntity (
                                uri                 TEXT            PRIMARY KEY,
                                datetime            TIMESTAMP(6)    NOT NULL,
                                timeBucketId        INTEGER         NOT NULL,
                                source              INTEGER         NOT NULL,
                                label               CHAR(32)                ,
                                content             BLOB            NOT NULL,
                                contentSizeBytes    INTEGER         NOT NULL,
                                contentHash         TEXT
                                ) WITHOUT ROWID"""

    DELETE_OLD_INDEX = """DROP INDEX IF EXISTS data_entity_bucket_index"""

    DATA_ENTITY_TABLE_INDEX = """CREATE INDEX IF NOT EXISTS data_entity_bucket_index2
                                ON DataEntity (timeBucketId, source, label, contentSizeBytes)"""

    CONTENT_HASH_INDEX = """CREATE INDEX IF NOT EXISTS content_hash_index
                           ON DataEntity (contentHash)"""

    HF_METADATA_TABLE_CREATE = """CREATE TABLE IF NOT EXISTS HFMetaData (
                                uri                 TEXT            PRIMARY KEY,
                                source              INTEGER         NOT NULL,
                                updatedAt           TIMESTAMP(6)    NOT NULL,
                                encodingKey         TEXT
                                ) WITHOUT ROWID"""

    def __init__(
        self,
        database="SqliteMinerStorage.sqlite",
        max_database_size_gb_hint=250,
    ):
        sqlite3.register_converter("timestamp", tz_aware_timestamp_adapter)
        self.database = database

        # TODO Account for non-content columns when restricting total database size.
        self.database_max_content_size_bytes = utils.gb_to_bytes(
            max_database_size_gb_hint
        )

        with contextlib.closing(self._create_connection()) as connection:
            cursor = connection.cursor()

            # Create the DataEntity table (if it does not already exist).
            cursor.execute(SqliteMinerStorage.DATA_ENTITY_TABLE_CREATE)

            # Delete the old index (if it exists).
            cursor.execute(SqliteMinerStorage.DELETE_OLD_INDEX)

            # Create the Index (if it does not already exist).
            cursor.execute(SqliteMinerStorage.DATA_ENTITY_TABLE_INDEX)

            # Create the huggingface table to store HF Info
            cursor.execute(SqliteMinerStorage.HF_METADATA_TABLE_CREATE)
            # Use Write Ahead Logging to avoid blocking reads.
            cursor.execute("pragma journal_mode=wal")

        # Update the HFMetaData for miners who created this table in previous versions
        self._ensure_hf_metadata_schema()
        
        # Ensure content hash column exists for deduplication
        self._ensure_content_hash_schema()
        
        # Create the content hash index AFTER ensuring the column exists
        self._ensure_content_hash_index()
        # Lock to avoid concurrency issues on clearing space when full.
        self.clearing_space_lock = threading.Lock()

        # Lock around the refresh for the index.
        self.cached_index_refresh_lock = threading.Lock()

        # Lock around the cached get miner index.
        self.cached_index_lock = threading.Lock()
        self.cached_index_4 = None
        self.cached_index_updated = dt.datetime.min
        
        # Deduplication cleanup settings
        self.last_deduplication_cleanup = None
        self.deduplication_cleanup_interval = dt.timedelta(minutes=30)
        self.max_duplicates_allowed = 200

    def _create_connection(self):
        # Create the database if it doesn't exist, defaulting to the local directory.
        # Use PARSE_DECLTYPES to convert accessed values into the appropriate type.
        connection = sqlite3.connect(
            self.database, detect_types=sqlite3.PARSE_DECLTYPES, timeout=60.0
        )
        # Allow this connection to parse results from returned rows by column name.
        connection.row_factory = sqlite3.Row

        return connection

    def _ensure_hf_metadata_schema(self):
        with contextlib.closing(self._create_connection()) as connection:
            cursor = connection.cursor()

            # Check if the encodingKey column exists
            cursor.execute("PRAGMA table_info(HFMetaData)")
            columns = [column[1] for column in cursor.fetchall()]

            if 'encodingKey' not in columns:
                # Add the new column
                cursor.execute("ALTER TABLE HFMetaData ADD COLUMN encodingKey TEXT")
                bt.logging.info("Added encodingKey column to HFMetaData table")

            connection.commit()

    def _ensure_content_hash_schema(self):
        """Ensures the content hash column exists for deduplication."""
        with contextlib.closing(self._create_connection()) as connection:
            cursor = connection.cursor()

            # Check if the contentHash column exists
            cursor.execute("PRAGMA table_info(DataEntity)")
            columns = [column[1] for column in cursor.fetchall()]

            if 'contentHash' not in columns:
                # Add the new column
                cursor.execute("ALTER TABLE DataEntity ADD COLUMN contentHash TEXT")
                bt.logging.info("Added contentHash column to DataEntity table")
                
                # Update existing rows with content hashes
                # Use uri as primary key since table is WITHOUT ROWID
                cursor.execute("SELECT uri, content FROM DataEntity WHERE contentHash IS NULL")
                rows = cursor.fetchall()
                
                for row in rows:
                    content_hash = hashlib.sha1(row[1]).hexdigest()
                    cursor.execute("UPDATE DataEntity SET contentHash = ? WHERE uri = ?", (content_hash, row[0]))
                
                bt.logging.info(f"Updated {len(rows)} existing rows with content hashes")

            connection.commit()

    def _ensure_content_hash_index(self):
        """Ensures the content hash index exists for efficient deduplication queries."""
        with contextlib.closing(self._create_connection()) as connection:
            cursor = connection.cursor()
            
            # Check if the index already exists
            cursor.execute("PRAGMA index_list(DataEntity)")
            existing_indexes = [index[1] for index in cursor.fetchall()]
            
            if 'content_hash_index' not in existing_indexes:
                # Create the index
                cursor.execute(SqliteMinerStorage.CONTENT_HASH_INDEX)
                bt.logging.info("Created content hash index for deduplication")
            
            connection.commit()

    def store_data_entities(self, data_entities: List[DataEntity]):
        """Stores any number of DataEntities, making space if necessary and applying deduplication."""

        # Apply deduplication before storage
        deduplicated_entities = self._deduplicate_data_entities(data_entities)
        
        added_content_size = 0
        for data_entity in deduplicated_entities:
            added_content_size += data_entity.content_size_bytes

        # If the total size of the store is larger than our maximum configured stored content size then ecept.
        if added_content_size > self.database_max_content_size_bytes:
            raise ValueError(
                "Content size to store: "
                + str(added_content_size)
                + " exceeds configured max: "
                + str(self.database_max_content_size_bytes)
            )

        with contextlib.closing(self._create_connection()) as connection:
            # Ensure only one thread is clearing space when necessary.
            with self.clearing_space_lock:
                # If we would exceed our maximum configured stored content size then clear space.
                cursor = connection.cursor()
                cursor.execute("SELECT SUM(contentSizeBytes) FROM DataEntity")

                # If there are no rows we convert the None result to 0
                result = cursor.fetchone()
                current_content_size = result[0] if result[0] else 0

                if (
                    current_content_size + added_content_size
                    > self.database_max_content_size_bytes
                ):
                    content_bytes_to_clear = (
                        self.database_max_content_size_bytes // 10
                        if self.database_max_content_size_bytes // 10
                        > added_content_size
                        else added_content_size
                    )
                    self.clear_content_from_oldest(content_bytes_to_clear)

            # Parse every DataEntity into an list of value lists for inserting.
            values = []

            for data_entity in deduplicated_entities:
                label = (
                    "NULL" if (data_entity.label is None) else data_entity.label.value
                )
                time_bucket_id = TimeBucket.from_datetime(data_entity.datetime).id
                content_hash = hashlib.sha1(data_entity.content).hexdigest()
                values.append(
                    [
                        data_entity.uri,
                        data_entity.datetime,
                        time_bucket_id,
                        data_entity.source,
                        label,
                        data_entity.content,
                        data_entity.content_size_bytes,
                        content_hash,
                    ]
                )

            # Insert overwriting duplicate keys (in case of updated content).
            cursor.executemany("REPLACE INTO DataEntity VALUES (?,?,?,?,?,?,?,?)", values)

            # Commit the insert.
            connection.commit()
            
            # Log deduplication results
            if len(deduplicated_entities) < len(data_entities):
                bt.logging.info(
                    f"Deduplication removed {len(data_entities) - len(deduplicated_entities)} duplicate entities. "
                    f"Stored {len(deduplicated_entities)} out of {len(data_entities)} entities."
                )
            
            # Perform periodic deduplication cleanup
            self.perform_periodic_deduplication_cleanup()

    def _deduplicate_data_entities(self, data_entities: List[DataEntity]) -> List[DataEntity]:
        """Deduplicates data entities using content hash and URI normalization with database checking."""
        if not data_entities:
            return []

        # Create sets to track seen content hashes and URIs within this batch
        seen_content_hashes = set()
        seen_uris = set()
        deduplicated_entities = []

        # Get existing content hashes from database for cross-bucket deduplication
        existing_content_hashes = self._get_existing_content_hashes()

        for entity in data_entities:
            # Calculate content hash
            content_hash = hashlib.sha1(entity.content).hexdigest()
            
            # Normalize URI (handle twitter.com vs x.com differences)
            normalized_uri = self._normalize_uri(entity.uri)
            
            # Check if this entity is a duplicate within current batch
            if content_hash in seen_content_hashes or normalized_uri in seen_uris:
                bt.logging.trace(f"Batch deduplication: skipping duplicate entity {entity.uri}")
                continue
            
            # Check if this entity is a duplicate against existing database content
            if content_hash in existing_content_hashes:
                bt.logging.trace(f"Cross-bucket deduplication: skipping duplicate entity {entity.uri}")
                continue
            
            # Add to tracking sets
            seen_content_hashes.add(content_hash)
            seen_uris.add(normalized_uri)
            deduplicated_entities.append(entity)

        if len(deduplicated_entities) < len(data_entities):
            bt.logging.info(
                f"Deduplication removed {len(data_entities) - len(deduplicated_entities)} entities. "
                f"Keeping {len(deduplicated_entities)} out of {len(data_entities)} entities."
            )

        return deduplicated_entities

    def _get_existing_content_hashes(self) -> set:
        """Gets existing content hashes from database for cross-bucket deduplication."""
        try:
            with contextlib.closing(self._create_connection()) as connection:
                cursor = connection.cursor()
                
                # Check if contentHash column exists
                cursor.execute("PRAGMA table_info(DataEntity)")
                columns = [column[1] for column in cursor.fetchall()]
                
                if 'contentHash' in columns:
                    # Use contentHash column if available
                    cursor.execute("SELECT contentHash FROM DataEntity WHERE contentHash IS NOT NULL")
                    return {row[0] for row in cursor.fetchall()}
                else:
                    # Fallback: calculate hashes from content (slower)
                    cursor.execute("SELECT content FROM DataEntity")
                    return {hashlib.sha1(row[0]).hexdigest() for row in cursor.fetchall()}
                    
        except Exception as e:
            bt.logging.warning(f"Error getting existing content hashes: {e}")
            return set()

    def _normalize_uri(self, uri: str) -> str:
        """Normalizes a URI for deduplication purposes."""
        # Handle twitter.com vs x.com differences
        if "twitter.com" in uri:
            return uri.replace("twitter.com", "x.com")
        elif "x.com" in uri:
            return uri.replace("x.com", "twitter.com")
        return uri

    def store_hf_dataset_info(self, hf_metadatas: List[HuggingFaceMetadata]):
        with contextlib.closing(self._create_connection()) as connection:
            cursor = connection.cursor()
            values = []
            bt.logging.info(f"Store_hf_dataset{hf_metadata}")
            for hf_metadata in hf_metadatas:
                values.append(
                    [
                        hf_metadata.repo_name,
                        hf_metadata.source,
                        hf_metadata.updated_at,
                        getattr(hf_metadata, 'encoding_key', None)  # Use getattr to handle cases where encoding_key might not exist
                    ]
                )
                bt.logging.info9(f"Store_hf_dataset_value{values}")
            cursor.executemany(
                "REPLACE INTO HFMetaData (uri, source, updatedAt, encodingKey) VALUES (?,?,?,?)", values)

            connection.commit()

    def get_earliest_data_datetime(self, source):
        query = "SELECT MIN(datetime) as earliest_date FROM DataEntity WHERE source = ?"
        with contextlib.closing(self._create_connection()) as connection:
            cursor = connection.cursor()
            cursor.execute(query, (source,))
            result = cursor.fetchone()
            return result['earliest_date'] if result and result['earliest_date'] else None

    def should_upload_hf_data(self, unique_id: str) -> bool:
        sql_query = """
            SELECT datetime(AVG(strftime('%s', UpdatedAt)), 'unixepoch') AS AvgUpdatedAt
            FROM (
                SELECT UpdatedAt
                FROM HFMetaData
                WHERE uri LIKE ?
                ORDER BY UpdatedAt DESC
                LIMIT 2
            );
        """
        try:
            with contextlib.closing(self._create_connection()) as connection:
                cursor = connection.cursor()
                cursor.execute(sql_query, (f"%_{unique_id}",))
                result = cursor.fetchone()

                if result is None or result[0] is None:
                    return True  # No data found, should upload

                average_datetime = dt.datetime.strptime(result[0], "%Y-%m-%d %H:%M:%S")
                average_datetime = average_datetime.replace(tzinfo=dt.timezone.utc)

                current_datetime = dt.datetime.now(dt.timezone.utc)

                # Calculate time difference for 5100 blocks (61 200 seconds (~17 hours))
                time_difference = dt.timedelta(seconds=61200)
                threshold_datetime = current_datetime - time_difference

                return threshold_datetime > average_datetime
        except sqlite3.Error as e:
            bt.logging.error(f"An error occurred: {e}")
            return False

    def get_hf_metadata(self, unique_id: str) -> List[HuggingFaceMetadata]:
        sql_query = """
            SELECT uri, source, updatedAt, 
                   CASE WHEN encodingKey IS NULL THEN '' ELSE encodingKey END as encodingKey
            FROM HFMetaData
            WHERE uri LIKE ?
            ORDER BY updatedAt DESC
            LIMIT 2;
        """

        with contextlib.closing(self._create_connection()) as connection:
            cursor = connection.cursor()
            cursor.execute(sql_query, (f"%_{unique_id}",))
            hf_metadatas = []

            for row in cursor:
                hf_metadata = HuggingFaceMetadata(
                    repo_name=row['uri'],
                    source=row['source'],
                    updated_at=row['updatedAt'],
                    encoding_key=row['encodingKey'] if row['encodingKey'] != '' else None
                )
                hf_metadatas.append(hf_metadata)

        return hf_metadatas

    def list_data_entities_in_data_entity_bucket(
        self, data_entity_bucket_id: DataEntityBucketId
    ) -> List[DataEntity]:
        """Lists from storage all DataEntities matching the provided DataEntityBucketId."""
        # Get rows that match the DataEntityBucketId.
        label = (
            "NULL"
            if (data_entity_bucket_id.label is None)
            else data_entity_bucket_id.label.value
        )

        with contextlib.closing(self._create_connection()) as connection:
            cursor = connection.cursor()
            cursor.execute(
                """SELECT * FROM DataEntity 
                        WHERE timeBucketId = ? AND source = ? AND label = ?""",
                [
                    data_entity_bucket_id.time_bucket.id,
                    data_entity_bucket_id.source,
                    label,
                ],
            )

            # Convert the rows into DataEntity objects and return them up to the configured max chuck size.
            data_entities = []

            running_size = 0

            for row in cursor:
                # If we have already reached the max DataEntityBucket size instead return early.
                if running_size >= constants.DATA_ENTITY_BUCKET_SIZE_LIMIT_BYTES:
                    return data_entities
                else:
                    # Construct the new DataEntity with all non null columns.
                    data_entity = DataEntity(
                        uri=row["uri"],
                        datetime=row["datetime"],
                        source=DataSource(row["source"]),
                        content=row["content"],
                        content_size_bytes=row["contentSizeBytes"],
                        label=DataLabel(value=row["label"]) if row["label"] != "NULL" else None
                    )

                    data_entities.append(data_entity)
                    running_size += row["contentSizeBytes"]

            # If we reach the end of the cursor then return all of the data entities for this DataEntityBucket.
            bt.logging.trace(
                f"Returning {len(data_entities)} data entities for bucket {data_entity_bucket_id}"
            )
            return data_entities

    def refresh_compressed_index(self, time_delta: dt.timedelta):
        """Refreshes the compressed MinerIndex."""
        # First check if we already have a fresh enough index, if so return immediately.
        # Since the GetMinerIndex uses a 30 minute freshness period this should be the default path with the
        # Refresh thread using a 20 minute freshness period and calling this method every 21 minutes.
        with self.cached_index_lock:
            if dt.datetime.now() - self.cached_index_updated <= time_delta:
                bt.logging.trace(
                    f"Skipping updating cached index. It is already fresher than {time_delta}."
                )
                return
            else:
                bt.logging.info(
                    f"Cached index out of {time_delta} freshness period. Refreshing cached index."
                )

        # Else we take the refresh lock and check again within the lock.
        # This handles cases where multiple threads are waiting on refresh at the same time.
        with self.cached_index_refresh_lock:
            with self.cached_index_lock:
                if dt.datetime.now() - self.cached_index_updated <= time_delta:
                    bt.logging.trace(
                        "After waiting on refresh lock the index was already refreshed."
                    )
                    return

            with contextlib.closing(self._create_connection()) as connection:
                cursor = connection.cursor()

                oldest_time_bucket_id = TimeBucket.from_datetime(
                    dt.datetime.now()
                    - dt.timedelta(constants.DATA_ENTITY_BUCKET_AGE_LIMIT_DAYS)
                ).id

                # Get sum of content_size_bytes for all rows grouped by DataEntityBucket.
                cursor.execute(
                    """SELECT SUM(contentSizeBytes) AS bucketSize, timeBucketId, source, label FROM DataEntity
                            WHERE timeBucketId >= ?
                            GROUP BY timeBucketId, source, label
                            ORDER BY bucketSize DESC
                            LIMIT ?
                            """,
                    [
                        oldest_time_bucket_id,
                        constants.DATA_ENTITY_BUCKET_COUNT_LIMIT_PER_MINER_INDEX_PROTOCOL_4,
                    ],  # Always get the max for caching and truncate to each necessary size.
                )

                buckets_by_source_by_label = defaultdict(dict)

                for row in cursor:
                    # Ensure the miner does not attempt to report more than the max DataEntityBucket size.
                    size = (
                        constants.DATA_ENTITY_BUCKET_SIZE_LIMIT_BYTES
                        if row["bucketSize"]
                        >= constants.DATA_ENTITY_BUCKET_SIZE_LIMIT_BYTES
                        else row["bucketSize"]
                    )

                    label = row["label"] if row["label"] != "NULL" else None

                    bucket = buckets_by_source_by_label[DataSource(row["source"])].get(
                        label, CompressedEntityBucket(label=label)
                    )
                    bucket.sizes_bytes.append(size)
                    bucket.time_bucket_ids.append(row["timeBucketId"])
                    buckets_by_source_by_label[DataSource(row["source"])][
                        label
                    ] = bucket

                # Convert the buckets_by_source_by_label into a list of lists of CompressedEntityBucket and return
                bt.logging.trace("Creating protocol 4 cached index.")
                with self.cached_index_lock:
                    self.cached_index_4 = CompressedMinerIndex(
                        sources={
                            source: list(labels_to_buckets.values())
                            for source, labels_to_buckets in buckets_by_source_by_label.items()
                        }
                    )
                    self.cached_index_updated = dt.datetime.now()
                    bt.logging.success(
                        f"Created cached index of {CompressedMinerIndex.size_bytes(self.cached_index_4)} bytes "
                        + f"across {CompressedMinerIndex.bucket_count(self.cached_index_4)} buckets."
                    )

    def list_contents_in_data_entity_buckets(
        self, data_entity_bucket_ids: List[DataEntityBucketId]
    ) -> Dict[DataEntityBucketId, List[bytes]]:
        """Lists contents for each requested DataEntityBucketId.
        Args:
            data_entity_bucket_ids (List[DataEntityBucketId]): Which buckets to get contents for.
        Returns:
            Dict[DataEntityBucketId, List[bytes]]: Map of each bucket id to contained contents.
        """
        # If no bucket ids or too many bucket ids are provided return an empty dict.
        if (
            len(data_entity_bucket_ids) == 0
            or len(data_entity_bucket_ids) > constants.BULK_BUCKETS_COUNT_LIMIT
        ):
            return defaultdict(list)

        # Get rows that match the DataEntityBucketIds.
        # Use a list of alternating ids and labels to match the upcoming sql query.
        time_bucket_ids_and_labels = list()
        for bucket_id in data_entity_bucket_ids:
            time_bucket_ids_and_labels.append(bucket_id.time_bucket.id)
            # Note that only twitter has NULL label and that all twitter labels are prefixed with #.
            # Therefore we do not need to distinguish labels by source.
            label = "NULL" if (bucket_id.label is None) else bucket_id.label.value
            time_bucket_ids_and_labels.append(label)

        with contextlib.closing(self._create_connection()) as connection:
            cursor = connection.cursor()
            cursor.execute(
                f"""SELECT timeBucketId, source, label, content, contentSizeBytes FROM DataEntity
                    WHERE timeBucketId = ? AND label = ?
                    {"OR timeBucketId = ? AND label = ?" * (len(data_entity_bucket_ids) - 1)}
                    LIMIT ?
                 """,
                list(time_bucket_ids_and_labels)
                + [constants.BULK_CONTENTS_COUNT_LIMIT],
            )

            # Get the contents from each row and return them up to the configured max size.
            buckets_ids_to_contents = defaultdict(list)
            running_size = 0

            for row in cursor:
                if running_size < constants.BULK_CONTENTS_SIZE_LIMIT_BYTES:
                    data_entity_bucket_id = DataEntityBucketId(
                        time_bucket=TimeBucket(id=row["timeBucketId"]),
                        source=DataSource(row["source"]),
                        label=DataLabel(value=row["label"]) if row["label"] != "NULL" else None
                    )
                    buckets_ids_to_contents[data_entity_bucket_id].append(
                        row["content"]
                    )
                    running_size += row["contentSizeBytes"]
                else:
                    # Return early since we hit the size limit.
                    break

            return buckets_ids_to_contents

    def get_compressed_index(
        self,
        bucket_count_limit=constants.DATA_ENTITY_BUCKET_COUNT_LIMIT_PER_MINER_INDEX_PROTOCOL_4,
    ) -> CompressedMinerIndex:
        """Gets the compressed MinerIndex, which is a summary of all of the DataEntities that this MinerStorage is currently serving."""

        # Force refresh index if 10 minutes beyond refersh period. Expected to be refreshed earlier by refresh loop.
        self.refresh_compressed_index(
            time_delta=(constants.MINER_CACHE_FRESHNESS + dt.timedelta(minutes=10))
        )

        with self.cached_index_lock:
            # Only protocol 4 is supported at this time.
            return self.cached_index_4

    def clear_content_from_oldest(self, content_bytes_to_clear: int):
        """Deletes entries starting from the oldest until we have cleared the specified amount of content."""

        bt.logging.debug(f"Database full. Clearing {content_bytes_to_clear} bytes.")

        with contextlib.closing(self._create_connection()) as connection:
            cursor = connection.cursor()

            # TODO Investigate way to select last X bytes worth of entries in a single query.
            # Get the contentSizeBytes of each row by timestamp desc.
            cursor.execute(
                "SELECT contentSizeBytes, datetime FROM DataEntity ORDER BY datetime ASC"
            )

            running_bytes = 0
            earliest_datetime_to_clear = dt.datetime.min
            # Iterate over rows until we have found bytes to clear or we reach the end and fail.
            for row in cursor:
                running_bytes += row["contentSizeBytes"]
                earliest_datetime_to_clear = row["datetime"]
                # Once we have enough content to clear then we do so.
                if running_bytes >= content_bytes_to_clear:
                    cursor.execute(
                        "DELETE FROM DataEntity WHERE datetime <= ?",
                        [earliest_datetime_to_clear],
                    )
                    connection.commit()

    def list_data_entity_buckets(self) -> List[DataEntityBucket]:
        """Lists all DataEntityBuckets for all the DataEntities that this MinerStorage is currently serving."""

        with contextlib.closing(self._create_connection()) as connection:
            cursor = connection.cursor()
            oldest_time_bucket_id = TimeBucket.from_datetime(
                dt.datetime.now()
                - dt.timedelta(constants.DATA_ENTITY_BUCKET_AGE_LIMIT_DAYS)
            ).id
            # Get sum of content_size_bytes for all rows grouped by DataEntityBucket.
            cursor.execute(
                """SELECT SUM(contentSizeBytes) AS bucketSize, timeBucketId, source, label FROM DataEntity
                        WHERE timeBucketId >= ?
                        GROUP BY timeBucketId, source, label
                        ORDER BY bucketSize DESC
                        LIMIT ?
                        """,
                [
                    oldest_time_bucket_id,
                    constants.DATA_ENTITY_BUCKET_COUNT_LIMIT_PER_MINER_INDEX,
                ],
            )

            data_entity_buckets = []

            for row in cursor:
                # Ensure the miner does not attempt to report more than the max DataEntityBucket size.
                size = (
                    constants.DATA_ENTITY_BUCKET_SIZE_LIMIT_BYTES
                    if row["bucketSize"]
                    >= constants.DATA_ENTITY_BUCKET_SIZE_LIMIT_BYTES
                    else row["bucketSize"]
                )

                # Construct the new DataEntityBucket with all non null columns.
                data_entity_bucket_id = DataEntityBucketId(
                    time_bucket=TimeBucket(id=row["timeBucketId"]),
                    source=DataSource(row["source"]),
                    label=(
                        DataLabel(value=row["label"])
                        if row["label"] != "NULL"
                        else None
                    ),
                )

                data_entity_bucket = DataEntityBucket(
                    id=data_entity_bucket_id, size_bytes=size
                )

                data_entity_buckets.append(data_entity_bucket)

            # If we reach the end of the cursor then return all of the data entity buckets.
            return data_entity_buckets

    def get_deduplication_stats(self) -> Dict[str, int]:
        """Returns deduplication statistics for monitoring purposes."""
        stats = {
            "total_entities": 0,
            "unique_uris": 0,
            "unique_content_hashes": 0,
            "duplicate_uris_removed": 0,
            "duplicate_content_removed": 0,
            "cross_bucket_duplicates_found": 0
        }
        
        try:
            with contextlib.closing(self._create_connection()) as connection:
                cursor = connection.cursor()
                
                # Get total entities
                cursor.execute("SELECT COUNT(*) FROM DataEntity")
                stats["total_entities"] = cursor.fetchone()[0]
                
                # Get unique URIs
                cursor.execute("SELECT COUNT(DISTINCT uri) FROM DataEntity")
                stats["unique_uris"] = cursor.fetchone()[0]
                
                # Check if contentHash column exists
                cursor.execute("PRAGMA table_info(DataEntity)")
                columns = [column[1] for column in cursor.fetchall()]
                
                if 'contentHash' in columns:
                    # Use contentHash column for faster calculation
                    cursor.execute("SELECT COUNT(DISTINCT contentHash) FROM DataEntity WHERE contentHash IS NOT NULL")
                    stats["unique_content_hashes"] = cursor.fetchone()[0]
                    
                    # Count content duplicates using contentHash
                    cursor.execute("""
                        SELECT COUNT(*) - COUNT(DISTINCT contentHash) 
                        FROM DataEntity 
                        WHERE contentHash IS NOT NULL
                    """)
                    stats["duplicate_content_removed"] = cursor.fetchone()[0]
                else:
                    # Fallback: calculate content hashes manually
                    cursor.execute("SELECT content FROM DataEntity")
                    content_hashes = set()
                    for row in cursor:
                        content_hash = hashlib.sha1(row[0]).hexdigest()
                        content_hashes.add(content_hash)
                    stats["unique_content_hashes"] = len(content_hashes)
                    stats["duplicate_content_removed"] = stats["total_entities"] - stats["unique_content_hashes"]
                
                # Calculate URI duplicates (should be 0 due to PRIMARY KEY constraint)
                stats["duplicate_uris_removed"] = stats["total_entities"] - stats["unique_uris"]
                
        except Exception as e:
            bt.logging.warning(f"Error getting deduplication stats: {e}")
            
        return stats

    def perform_periodic_deduplication_cleanup(self):
        """Performs periodic deduplication cleanup to ensure no more than max_duplicates_allowed duplicates exist."""
        now = dt.datetime.now()
        
        # Check if it's time to perform cleanup
        if (self.last_deduplication_cleanup is None or 
            now - self.last_deduplication_cleanup >= self.deduplication_cleanup_interval):
            
            try:
                with contextlib.closing(self._create_connection()) as connection:
                    cursor = connection.cursor()
                    
                    # Get current duplicate statistics
                    stats = self.get_deduplication_stats()
                    total_duplicates = stats["duplicate_uris_removed"] + stats["duplicate_content_removed"]
                    
                    bt.logging.info(f"Periodic deduplication check: {total_duplicates} total duplicates found")
                    
                    # If duplicates exceed the limit, perform cleanup
                    if total_duplicates > self.max_duplicates_allowed:
                        bt.logging.warning(f"Duplicate count ({total_duplicates}) exceeds limit ({self.max_duplicates_allowed}). Starting cleanup...")
                        
                        # Remove duplicate URIs (keep the most recent)
                        cursor.execute("""
                            DELETE FROM DataEntity 
                            WHERE uri IN (
                                SELECT uri FROM (
                                    SELECT uri, 
                                           ROW_NUMBER() OVER (PARTITION BY uri ORDER BY datetime DESC) as rn
                                    FROM DataEntity
                                ) 
                                WHERE rn > 1
                            )
                        """)
                        
                        # Remove duplicate content (keep the most recent)
                        # Use uri as primary key since table is WITHOUT ROWID
                        cursor.execute("""
                            DELETE FROM DataEntity 
                            WHERE uri IN (
                                SELECT uri FROM (
                                    SELECT uri, 
                                           ROW_NUMBER() OVER (
                                               PARTITION BY contentHash 
                                               ORDER BY datetime DESC
                                           ) as rn
                                    FROM DataEntity
                                    WHERE contentHash IS NOT NULL
                                ) 
                                WHERE rn > 1
                            )
                        """)
                        
                        connection.commit()
                        
                        # Get updated statistics
                        updated_stats = self.get_deduplication_stats()
                        updated_duplicates = updated_stats["duplicate_uris_removed"] + updated_stats["duplicate_content_removed"]
                        
                        bt.logging.success(
                            f"Deduplication cleanup completed. "
                            f"Removed {total_duplicates - updated_duplicates} duplicates. "
                            f"Remaining duplicates: {updated_duplicates}"
                        )
                    else:
                        bt.logging.trace(f"Duplicate count ({total_duplicates}) is within limit ({self.max_duplicates_allowed})")
                
                # Update last cleanup time
                self.last_deduplication_cleanup = now
                
            except Exception as e:
                bt.logging.error(f"Error during periodic deduplication cleanup: {e}")
                bt.logging.debug(traceback.format_exc())

    def get_recent_entities_for_keywords(self, hours: int = 24, limit: int = 1000) -> List[DataEntity]:
        """Get recent data entities for keyword discovery."""
        with contextlib.closing(self._create_connection()) as connection:
            cursor = connection.cursor()
            
            # Calculate the cutoff time
            cutoff_time = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=hours)
            
            # Query recent entities ordered by datetime (most recent first)
            cursor.execute("""
                SELECT uri, datetime, timeBucketId, source, label, content, contentSizeBytes
                FROM DataEntity 
                WHERE datetime >= ? 
                ORDER BY datetime DESC 
                LIMIT ?
            """, (cutoff_time, limit))
            
            entities = []
            for row in cursor.fetchall():
                entity = DataEntity(
                    uri=row['uri'],
                    datetime=row['datetime'],
                    source=DataSource(row['source']),
                    label=DataLabel(value=row['label']) if row['label'] else None,
                    content=row['content'],
                    content_size_bytes=row['contentSizeBytes']
                )
                entities.append(entity)
            
            return entities
