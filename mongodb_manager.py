# import os
# import logging
# import asyncio
# from datetime import datetime
# from typing import Optional, Dict, Any, List
# from motor.motor_asyncio import AsyncIOMotorClient
# from pymongo.errors import ConnectionFailure, DuplicateKeyError
# from bson import ObjectId

# logger = logging.getLogger(__name__)

# class MongoDBManager:
#     """Async MongoDB manager for video processing records"""
    
#     def __init__(self, mongodb_uri: str, database_name: str = "autoedit"):
#         """
#         Initialize MongoDB manager
        
#         Args:
#             mongodb_uri (str): MongoDB connection URI
#             database_name (str): Database name
#         """
#         self.mongodb_uri = mongodb_uri
#         self.database_name = database_name
#         self.client = None
#         self.db = None
#         self.collection = None
    
#     async def initialize(self):
#         """Initialize MongoDB connection"""
#         try:
#             # Create async client
#             self.client = AsyncIOMotorClient(self.mongodb_uri)
            
#             # Test connection
#             await self.client.admin.command('ismaster')
            
#             # Get database and collection
#             self.db = self.client[self.database_name]
#             self.collection = self.db.processed_videos
            
#             # Create indexes for better performance
#             await self.collection.create_index([("user_id", 1)])
#             await self.collection.create_index([("session_id", 1)])
#             await self.collection.create_index([("created_at", -1)])
#             await self.collection.create_index([("user_id", 1), ("created_at", -1)])
            
#             logger.info(f"MongoDB connection established to database: {self.database_name}")
            
#         except ConnectionFailure as e:
#             logger.error(f"Failed to connect to MongoDB: {e}")
#             raise
#         except Exception as e:
#             logger.error(f"Error initializing MongoDB: {e}")
#             raise
    
#     async def store_video_record(self, 
#                                 user_id: str, 
#                                 session_id: str, 
#                                 gdrive_data: Dict[str, Any],
#                                 processing_data: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Store video processing record in MongoDB
        
#         Args:
#             user_id (str): User identifier
#             session_id (str): Processing session ID
#             gdrive_data (dict): Google Drive upload result
#             processing_data (dict): Video processing results
            
#         Returns:
#             dict: Stored record with MongoDB _id
#         """
#         if not self.collection:
#             await self.initialize()
        
#         try:
#             # Prepare document
#             document = {
#                 "user_id": user_id,
#                 "session_id": session_id,
#                 "created_at": datetime.utcnow(),
#                 "status": "completed",
                
#                 # Google Drive information
#                 "gdrive": {
#                     "file_id": gdrive_data.get("file_id"),
#                     "file_name": gdrive_data.get("file_name"),
#                     "web_view_link": gdrive_data.get("web_view_link"),
#                     "download_link": gdrive_data.get("download_link"),
#                     "size": gdrive_data.get("size"),
#                     "folder_id": gdrive_data.get("folder_id"),
#                     "upload_time": datetime.utcnow()
#                 },
                
#                 # Processing information
#                 "processing": {
#                     "input_video": processing_data.get("input_video"),
#                     "original_output_path": processing_data.get("output_video"),
#                     "transcription": processing_data.get("transcription", {}),
#                     "optimization": processing_data.get("optimization", {}),
#                     "cost_summary": processing_data.get("cost_summary", {}),
#                     "processing_time": processing_data.get("processing_time")
#                 },
                
#                 # Metadata
#                 "metadata": {
#                     "version": "1.0",
#                     "processor_type": "async_video_processor"
#                 }
#             }
            
#             # Insert document
#             result = await self.collection.insert_one(document)
            
#             # Return the stored document with _id
#             stored_doc = await self.collection.find_one({"_id": result.inserted_id})
#             stored_doc["_id"] = str(stored_doc["_id"])  # Convert ObjectId to string
            
#             logger.info(f"Stored video record for user {user_id}, session {session_id}")
#             logger.info(f"MongoDB document ID: {stored_doc['_id']}")
            
#             return {
#                 "success": True,
#                 "document_id": stored_doc["_id"],
#                 "record": stored_doc
#             }
            
#         except DuplicateKeyError:
#             logger.error(f"Duplicate record for session {session_id}")
#             return {
#                 "success": False,
#                 "error": "Duplicate session record"
#             }
#         except Exception as e:
#             logger.error(f"Error storing video record: {e}")
#             return {
#                 "success": False,
#                 "error": str(e)
#             }
    
#     async def get_user_videos(self, user_id: str, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
#         """
#         Get all videos processed for a specific user
        
#         Args:
#             user_id (str): User identifier
#             limit (int): Maximum number of records to return
#             offset (int): Number of records to skip
            
#         Returns:
#             list: List of video records
#         """
#         if not self.collection:
#             await self.initialize()
        
#         try:
#             cursor = self.collection.find(
#                 {"user_id": user_id}
#             ).sort("created_at", -1).skip(offset).limit(limit)
            
#             videos = []
#             async for doc in cursor:
#                 doc["_id"] = str(doc["_id"])
#                 videos.append(doc)
            
#             logger.info(f"Retrieved {len(videos)} videos for user {user_id}")
#             return videos
            
#         except Exception as e:
#             logger.error(f"Error retrieving user videos: {e}")
#             return []
    
#     async def get_video_by_session(self, session_id: str) -> Optional[Dict[str, Any]]:
#         """
#         Get video record by session ID
        
#         Args:
#             session_id (str): Processing session ID
            
#         Returns:
#             dict or None: Video record if found
#         """
#         if not self.collection:
#             await self.initialize()
        
#         try:
#             doc = await self.collection.find_one({"session_id": session_id})
#             if doc:
#                 doc["_id"] = str(doc["_id"])
#                 logger.info(f"Found video record for session {session_id}")
#                 return doc
#             else:
#                 logger.info(f"No video record found for session {session_id}")
#                 return None
                
#         except Exception as e:
#             logger.error(f"Error retrieving video by session: {e}")
#             return None
    
#     async def update_video_status(self, session_id: str, status: str, additional_data: Dict[str, Any] = None) -> bool:
#         """
#         Update video processing status
        
#         Args:
#             session_id (str): Processing session ID
#             status (str): New status
#             additional_data (dict): Additional data to update
            
#         Returns:
#             bool: True if successful
#         """
#         if not self.collection:
#             await self.initialize()
        
#         try:
#             update_doc = {
#                 "status": status,
#                 "updated_at": datetime.utcnow()
#             }
            
#             if additional_data:
#                 update_doc.update(additional_data)
            
#             result = await self.collection.update_one(
#                 {"session_id": session_id},
#                 {"$set": update_doc}
#             )
            
#             if result.modified_count > 0:
#                 logger.info(f"Updated status for session {session_id} to {status}")
#                 return True
#             else:
#                 logger.warning(f"No document updated for session {session_id}")
#                 return False
                
#         except Exception as e:
#             logger.error(f"Error updating video status: {e}")
#             return False
    
#     async def delete_video_record(self, session_id: str) -> bool:
#         """
#         Delete video record by session ID
        
#         Args:
#             session_id (str): Processing session ID
            
#         Returns:
#             bool: True if successful
#         """
#         if not self.collection:
#             await self.initialize()
        
#         try:
#             result = await self.collection.delete_one({"session_id": session_id})
            
#             if result.deleted_count > 0:
#                 logger.info(f"Deleted video record for session {session_id}")
#                 return True
#             else:
#                 logger.warning(f"No document deleted for session {session_id}")
#                 return False
                
#         except Exception as e:
#             logger.error(f"Error deleting video record: {e}")
#             return False
    
#     async def get_processing_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
#         """
#         Get processing statistics
        
#         Args:
#             user_id (str, optional): Get stats for specific user
            
#         Returns:
#             dict: Processing statistics
#         """
#         if not self.collection:
#             await self.initialize()
        
#         try:
#             match_stage = {}
#             if user_id:
#                 match_stage["user_id"] = user_id
            
#             pipeline = [
#                 {"$match": match_stage},
#                 {
#                     "$group": {
#                         "_id": None,
#                         "total_videos": {"$sum": 1},
#                         "total_size": {"$sum": "$gdrive.size"},
#                         "total_cost": {"$sum": "$processing.cost_summary.estimated_cost_usd"},
#                         "avg_cost": {"$avg": "$processing.cost_summary.estimated_cost_usd"},
#                         "latest_processing": {"$max": "$created_at"},
#                         "earliest_processing": {"$min": "$created_at"}
#                     }
#                 }
#             ]
            
#             result = await self.collection.aggregate(pipeline).to_list(1)
            
#             if result:
#                 stats = result[0]
#                 stats.pop("_id", None)
#                 return stats
#             else:
#                 return {
#                     "total_videos": 0,
#                     "total_size": 0,
#                     "total_cost": 0.0,
#                     "avg_cost": 0.0,
#                     "latest_processing": None,
#                     "earliest_processing": None
#                 }
                
#         except Exception as e:
#             logger.error(f"Error getting processing stats: {e}")
#             return {}
    
#     async def close(self):
#         """Close MongoDB connection"""
#         if self.client:
#             self.client.close()
#             logger.info("MongoDB connection closed")
    
#     async def __aenter__(self):
#         """Async context manager entry"""
#         await self.initialize()
#         return self
    
#     async def __aexit__(self, exc_type, exc_val, exc_tb):
#         """Async context manager exit"""
#         await self.close()
# mongodb_manager.py - FIXED VERSION
import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure, DuplicateKeyError
from bson import ObjectId
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

class MongoDBManager:
    """Async MongoDB manager for video processing records"""
    
    def __init__(self, mongodb_uri: str = None, database_name: str = "emoai2"):
        """
        Initialize MongoDB manager
        
        Args:
            mongodb_uri (str): MongoDB connection URI
            database_name (str): Database name
        """
        self.mongodb_uri = mongodb_uri or os.getenv('MONGODB_URI')
        
        # Build URI from components if not provided
        if not self.mongodb_uri:
            username = quote_plus(os.getenv('MONGO_USERNAME', ''))
            password = quote_plus(os.getenv('MONGO_PASSWORD', ''))
            cluster = os.getenv('MONGO_CLUSTER', '')
            if username and password and cluster:
                self.mongodb_uri = f"mongodb+srv://{username}:{password}@{cluster}/?retryWrites=true&w=majority"
            else:
                raise ValueError("MongoDB URI not provided and cannot be built from environment variables")
        
        self.database_name = database_name
        self.client = None
        self.db = None
        self.collection = None  # Initialize as None
        self._initialized = False
    
    async def initialize(self):
        """Initialize MongoDB connection"""
        if self._initialized:
            return
            
        try:
            # Create async client with proper settings for Atlas
            self.client = AsyncIOMotorClient(
                self.mongodb_uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                socketTimeoutMS=10000,
                maxPoolSize=10,
                minPoolSize=1
            )
            
            # Test connection
            await self.client.admin.command('ping')
            logger.info("Successfully connected to MongoDB Atlas")
            
            # Get database and collection
            self.db = self.client[self.database_name]
            self.collection = self.db.processed_videos  # Set the collection
            
            # Create indexes for better performance
            await self.create_indexes()
            
            self._initialized = True
            logger.info(f"MongoDB initialized with database: {self.database_name}")
            
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
        except Exception as e:
            logger.error(f"Error initializing MongoDB: {e}")
            raise
    
    async def create_indexes(self):
        """Create indexes for optimal performance"""
        try:
            # Index on user_id for user queries
            await self.collection.create_index("user_id")
            
            # Index on session_id for unique lookups
            await self.collection.create_index("session_id", unique=True)
            
            # Compound index for user + timestamp queries
            await self.collection.create_index([
                ("user_id", 1),
                ("created_at", -1)
            ])
            
            logger.info("MongoDB indexes created successfully")
        except Exception as e:
            logger.warning(f"Index creation issue (may already exist): {e}")
    
    async def ensure_initialized(self):
        """Ensure MongoDB is initialized before any operation"""
        if not self._initialized or self.collection is None:
            await self.initialize()
    
    async def store_video_record(self, 
                                user_id: str, 
                                session_id: str, 
                                gdrive_data: Dict[str, Any],
                                processing_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store video processing record in MongoDB
        """
        await self.ensure_initialized()
        
        try:
            # Prepare document
            document = {
                "user_id": user_id,
                "session_id": session_id,
                "created_at": datetime.utcnow(),
                "status": "completed",
                
                "gdrive": {
                    "file_id": gdrive_data.get("file_id"),
                    "file_name": gdrive_data.get("file_name"),
                    "web_view_link": gdrive_data.get("web_view_link"),
                    "download_link": gdrive_data.get("download_link"),
                    "size": gdrive_data.get("size"),
                    "folder_id": gdrive_data.get("folder_id"),
                    "upload_time": datetime.utcnow()
                },
                
                "processing": {
                    "input_video": processing_data.get("input_video"),
                    "original_output_path": processing_data.get("output_video"),
                    "transcription": processing_data.get("transcription", {}),
                    "optimization": processing_data.get("optimization", {}),
                    "cost_summary": processing_data.get("cost_summary", {}),
                    "processing_time": processing_data.get("processing_time")
                },
                
                "metadata": {
                    "version": "1.0",
                    "processor_type": "async_video_processor"
                }
            }
            
            result = await self.collection.insert_one(document)
            
            logger.info(f"Stored video record for user {user_id}, session {session_id}")
            logger.info(f"MongoDB document ID: {result.inserted_id}")
            
            return {
                "success": True,
                "document_id": str(result.inserted_id)
            }
            
        except DuplicateKeyError:
            logger.error(f"Duplicate record for session {session_id}")
            return {"success": False, "error": "Duplicate session record"}
        except Exception as e:
            logger.error(f"Error storing video record: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_user_videos(self, user_id: str, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Get all videos processed for a specific user"""
        await self.ensure_initialized()
        
        try:
            cursor = self.collection.find(
                {"user_id": user_id}
            ).sort("created_at", -1).skip(offset).limit(limit)
            
            videos = []
            async for doc in cursor:
                doc["_id"] = str(doc["_id"])
                videos.append(doc)
            
            logger.info(f"Retrieved {len(videos)} videos for user {user_id}")
            return videos
            
        except Exception as e:
            logger.error(f"Error retrieving user videos: {e}")
            return []
    
    async def get_video_by_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get video record by session ID"""
        await self.ensure_initialized()
        
        try:
            doc = await self.collection.find_one({"session_id": session_id})
            if doc:
                doc["_id"] = str(doc["_id"])
                logger.info(f"Found video record for session {session_id}")
                return doc
            else:
                logger.info(f"No video record found for session {session_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving video by session: {e}")
            return None
    
    async def update_video_status(self, session_id: str, status: str, additional_data: Dict[str, Any] = None) -> bool:
        """Update video processing status"""
        await self.ensure_initialized()
        
        try:
            update_doc = {"status": status, "updated_at": datetime.utcnow()}
            if additional_data:
                update_doc.update(additional_data)
            
            result = await self.collection.update_one(
                {"session_id": session_id},
                {"$set": update_doc}
            )
            
            if result.modified_count > 0:
                logger.info(f"Updated status for session {session_id} to {status}")
                return True
            else:
                logger.warning(f"No document updated for session {session_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating video status: {e}")
            return False
    
    async def delete_video_record(self, session_id: str) -> bool:
        """Delete video record by session ID"""
        await self.ensure_initialized()
        
        try:
            result = await self.collection.delete_one({"session_id": session_id})
            
            if result.deleted_count > 0:
                logger.info(f"Deleted video record for session {session_id}")
                return True
            else:
                logger.warning(f"No document deleted for session {session_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting video record: {e}")
            return False
    
    async def get_processing_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get processing statistics"""
        await self.ensure_initialized()
        
        try:
            match_stage = {}
            if user_id:
                match_stage["user_id"] = user_id
            
            pipeline = [
                {"$match": match_stage},
                {
                    "$group": {
                        "_id": None,
                        "total_videos": {"$sum": 1},
                        "total_size": {"$sum": "$gdrive.size"},
                        "total_cost": {"$sum": "$processing.cost_summary.estimated_cost_usd"},
                        "avg_cost": {"$avg": "$processing.cost_summary.estimated_cost_usd"},
                        "latest_processing": {"$max": "$created_at"},
                        "earliest_processing": {"$min": "$created_at"}
                    }
                }
            ]
            
            result = await self.collection.aggregate(pipeline).to_list(1)
            
            if result:
                stats = result[0]
                stats.pop("_id", None)
                return stats
            else:
                return {
                    "total_videos": 0,
                    "total_size": 0,
                    "total_cost": 0.0,
                    "avg_cost": 0.0,
                    "latest_processing": None,
                    "earliest_processing": None
                }
                
        except Exception as e:
            logger.error(f"Error getting processing stats: {e}")
            return {}
    
    async def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            self._initialized = False
            self.collection = None
            logger.info("MongoDB connection closed")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
# import os
# import logging
# import asyncio
# from datetime import datetime
# from typing import Optional, Dict, Any, List
# from motor.motor_asyncio import AsyncIOMotorClient
# from pymongo.errors import ConnectionFailure, DuplicateKeyError
# from bson import ObjectId
# from urllib.parse import quote_plus
# import motor

# logger = logging.getLogger(__name__)

# class MongoDBManager:
#     """Async MongoDB manager for video processing records"""
    
#     def __init__(self, mongodb_uri: str, database_name: str = "emoai2"):
#         """
#         Initialize MongoDB manager
        
#         Args:
#             mongodb_uri (str): MongoDB connection URI
#             database_name (str): Database name
#         """
#         self.mongodb_uri = mongodb_uri or os.getenv('MONGODB_URI')
        
#         # Alternative: Build URI from components
#         if not self.mongodb_uri:
#             username = quote_plus(os.getenv('MONGO_USERNAME'))
#             password = quote_plus(os.getenv('MONGO_PASSWORD'))
#             cluster = os.getenv('MONGO_CLUSTER')
#             self.mongodb_uri = f"mongodb+srv://{username}:{password}@{cluster}/video_processor?retryWrites=true&w=majority"
        
#         self.database_name = database_name
#         self.client = None
#         self.db = None
    
#     async def initialize(self):
#         """Initialize MongoDB connection"""
#         try:
#             # Create async client
#             # self.client = AsyncIOMotorClient(self.mongodb_uri)
            
#             # # Test connection
#             # await self.client.admin.command('ismaster')
            
#             # # Get database and collection
#             # self.db = self.client[self.database_name]
#             # self.collection = self.db.processed_videos
            
#             # # Create indexes for better performance 
#             # await self.collection.create_index([("user_id", 1)])
#             # await self.collection.create_index([("session_id", 1)], unique=True) # Added unique constraint
#             # await self.collection.create_index([("created_at", -1)])
#             # await self.collection.create_index([("user_id", 1), ("created_at", -1)])
#             # Use motor for async operations
#             self.client = motor.motor_asyncio.AsyncIOMotorClient(
#                 self.mongodb_uri,
#                 serverSelectionTimeoutMS=5000,
#                 connectTimeoutMS=10000,
#                 socketTimeoutMS=10000,
#                 maxPoolSize=10,
#                 minPoolSize=1
#             )
            
#             # Test connection
#             await self.client.admin.command('ping')
#             logger.info("Successfully connected to MongoDB Atlas")
            
#             self.db = self.client[self.database_name]
            
#             # Create indexes for better performance
#             await self.create_indexes()            
#             logger.info(f"MongoDB connection established to database: {self.database_name}")
            
#         except ConnectionFailure as e:
#             logger.error(f"Failed to connect to MongoDB: {e}")
#             raise
#         except Exception as e:
#             logger.error(f"Error initializing MongoDB: {e}")
#             raise
#     async def create_indexes(self):
#         """Create indexes for optimal performance"""
#         try:
#             # Index on user_id for user queries
#             await self.db.videos.create_index("user_id")
            
#             # Index on session_id for unique lookups
#             await self.db.videos.create_index("session_id", unique=True)
            
#             # Compound index for user + timestamp queries
#             await self.db.videos.create_index([
#                 ("user_id", 1),
#                 ("created_at", -1)
#             ])
            
#             # TTL index to auto-delete old records (optional)
#             # await self.db.videos.create_index(
#             #     "created_at",
#             #     expireAfterSeconds=30*24*60*60  # 30 days
#             # )
            
#             logger.info("MongoDB indexes created successfully")
#         except Exception as e:
#             logger.warning(f"Index creation issue: {e}")    
#     async def store_video_record(self, 
#                                  user_id: str, 
#                                  session_id: str, 
#                                  gdrive_data: Dict[str, Any],
#                                  processing_data: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Store video processing record in MongoDB
        
#         Args:
#             user_id (str): User identifier
#             session_id (str): Processing session ID
#             gdrive_data (dict): Google Drive upload result
#             processing_data (dict): Video processing results
            
#         Returns:
#             dict: Result of the storage operation
#         """
#         # CORRECTED CHECK
#         if self.collection is None:
#             await self.initialize()
        
#         try:
#             # Prepare document
#             document = {
#                 "user_id": user_id,
#                 "session_id": session_id,
#                 "created_at": datetime.utcnow(),
#                 "status": "completed",
                
#                 "gdrive": {
#                     "file_id": gdrive_data.get("file_id"),
#                     "file_name": gdrive_data.get("file_name"),
#                     "web_view_link": gdrive_data.get("web_view_link"),
#                     "download_link": gdrive_data.get("download_link"),
#                     "size": gdrive_data.get("size"),
#                     "folder_id": gdrive_data.get("folder_id"),
#                     "upload_time": datetime.utcnow()
#                 },
                
#                 "processing": {
#                     "input_video": processing_data.get("input_video"),
#                     "original_output_path": processing_data.get("output_video"),
#                     "transcription": processing_data.get("transcription", {}),
#                     "optimization": processing_data.get("optimization", {}),
#                     "cost_summary": processing_data.get("cost_summary", {}),
#                     "processing_time": processing_data.get("processing_time")
#                 },
                
#                 "metadata": {
#                     "version": "1.0",
#                     "processor_type": "async_video_processor"
#                 }
#             }
            
#             result = await self.collection.insert_one(document)
            
#             logger.info(f"Stored video record for user {user_id}, session {session_id}")
#             logger.info(f"MongoDB document ID: {result.inserted_id}")
            
#             return {
#                 "success": True,
#                 "document_id": str(result.inserted_id)
#             }
            
#         except DuplicateKeyError:
#             logger.error(f"Duplicate record for session {session_id}")
#             return { "success": False, "error": "Duplicate session record" }
#         except Exception as e:
#             logger.error(f"Error storing video record: {e}")
#             return { "success": False, "error": str(e) }
    
#     async def get_user_videos(self, user_id: str, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
#         """
#         Get all videos processed for a specific user
#         """
#         # CORRECTED CHECK
#         if self.collection is None:
#             await self.initialize()
        
#         try:
#             cursor = self.collection.find(
#                 {"user_id": user_id}
#             ).sort("created_at", -1).skip(offset).limit(limit)
            
#             videos = []
#             async for doc in cursor:
#                 doc["_id"] = str(doc["_id"])
#                 videos.append(doc)
            
#             logger.info(f"Retrieved {len(videos)} videos for user {user_id}")
#             return videos
            
#         except Exception as e:
#             logger.error(f"Error retrieving user videos: {e}")
#             return []
    
#     async def get_video_by_session(self, session_id: str) -> Optional[Dict[str, Any]]:
#         """
#         Get video record by session ID
#         """
#         # CORRECTED CHECK
#         if self.collection is None:
#             await self.initialize()
        
#         try:
#             doc = await self.collection.find_one({"session_id": session_id})
#             if doc:
#                 doc["_id"] = str(doc["_id"])
#                 logger.info(f"Found video record for session {session_id}")
#                 return doc
#             else:
#                 logger.info(f"No video record found for session {session_id}")
#                 return None
                
#         except Exception as e:
#             logger.error(f"Error retrieving video by session: {e}")
#             return None
    
#     async def update_video_status(self, session_id: str, status: str, additional_data: Dict[str, Any] = None) -> bool:
#         """
#         Update video processing status
#         """
#         # CORRECTED CHECK
#         if self.collection is None:
#             await self.initialize()
        
#         try:
#             update_doc = { "status": status, "updated_at": datetime.utcnow() }
#             if additional_data:
#                 update_doc.update(additional_data)
            
#             result = await self.collection.update_one(
#                 {"session_id": session_id},
#                 {"$set": update_doc}
#             )
            
#             if result.modified_count > 0:
#                 logger.info(f"Updated status for session {session_id} to {status}")
#                 return True
#             else:
#                 logger.warning(f"No document updated for session {session_id}")
#                 return False
                
#         except Exception as e:
#             logger.error(f"Error updating video status: {e}")
#             return False
    
#     async def delete_video_record(self, session_id: str) -> bool:
#         """
#         Delete video record by session ID
#         """
#         # CORRECTED CHECK
#         if self.collection is None:
#             await self.initialize()
        
#         try:
#             result = await self.collection.delete_one({"session_id": session_id})
            
#             if result.deleted_count > 0:
#                 logger.info(f"Deleted video record for session {session_id}")
#                 return True
#             else:
#                 logger.warning(f"No document deleted for session {session_id}")
#                 return False
                
#         except Exception as e:
#             logger.error(f"Error deleting video record: {e}")
#             return False
    
#     async def get_processing_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
#         """
#         Get processing statistics
#         """
#         # CORRECTED CHECK
#         if self.collection is None:
#             await self.initialize()
        
#         try:
#             match_stage = {}
#             if user_id:
#                 match_stage["user_id"] = user_id
            
#             pipeline = [
#                 {"$match": match_stage},
#                 {
#                     "$group": {
#                         "_id": None,
#                         "total_videos": {"$sum": 1},
#                         "total_size": {"$sum": "$gdrive.size"},
#                         "total_cost": {"$sum": "$processing.cost_summary.estimated_cost_usd"},
#                         "avg_cost": {"$avg": "$processing.cost_summary.estimated_cost_usd"},
#                         "latest_processing": {"$max": "$created_at"},
#                         "earliest_processing": {"$min": "$created_at"}
#                     }
#                 }
#             ]
            
#             result = await self.collection.aggregate(pipeline).to_list(1)
            
#             if result:
#                 stats = result[0]
#                 stats.pop("_id", None)
#                 return stats
#             else:
#                 return {
#                     "total_videos": 0, "total_size": 0, "total_cost": 0.0,
#                     "avg_cost": 0.0, "latest_processing": None, "earliest_processing": None
#                 }
                
#         except Exception as e:
#             logger.error(f"Error getting processing stats: {e}")
#             return {}
    
#     async def close(self):
#         """Close MongoDB connection"""
#         if self.client:
#             self.client.close()
#             logger.info("MongoDB connection closed")
    
#     async def __aenter__(self):
#         """Async context manager entry"""
#         await self.initialize()
#         return self
    
#     async def __aexit__(self, exc_type, exc_val, exc_tb):
#         """Async context manager exit"""
#         await self.close()