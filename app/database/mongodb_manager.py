# app/database/mongodb_manager.py
import motor.motor_asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class MongoDBManager:
    """Async MongoDB manager for video processing tasks"""

    # Add this to your MongoDBManager.__init__ method
    def __init__(self, mongodb_uri: str, database_name: str = "emoai2"):
        self.mongodb_uri = mongodb_uri
        self.database_name = database_name
        self.client = None
        self.db = None
        self.tasks_collection = None
        self.videos_collection = None
        self.stats_collection = None
        self.users_collection = None  # Add this line
        
        logger.info(f"MongoDB manager initialized for database: {database_name}")
    
    # Add this to your initialize method
    async def initialize(self):
        """Initialize MongoDB connection and create indexes"""
        try:
            self.client = motor.motor_asyncio.AsyncIOMotorClient(self.mongodb_uri)
            self.db = self.client[self.database_name]
            
            # Initialize collections
            self.tasks_collection = self.db.processing_tasks
            self.videos_collection = self.db.processed_videos
            self.stats_collection = self.db.processing_stats
            self.users_collection = self.db.users  # Add this line
            
            # Test connection
            await self.client.admin.command('ping')
            logger.info("MongoDB connection established successfully")
            
            # Create indexes for better performance
            await self._create_indexes()
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    # Add this to your _create_indexes method
    async def _create_indexes(self):
        """Create database indexes for optimal performance"""
        try:
            # Tasks collection indexes
            await self.tasks_collection.create_index("task_id", unique=True)
            await self.tasks_collection.create_index("user_id")
            await self.tasks_collection.create_index("status")
            await self.tasks_collection.create_index([("created_at", -1)])  # Descending for recent first
            await self.tasks_collection.create_index([("user_id", 1), ("created_at", -1)])
            await self.tasks_collection.create_index("celery_task_id")
            
            # Videos collection indexes  
            await self.videos_collection.create_index("user_id")
            await self.videos_collection.create_index("session_id", unique=True)
            await self.videos_collection.create_index([("created_at", -1)])
            
            # Users collection indexes  # Add this section
            await self.users_collection.create_index("email", unique=True)
            await self.users_collection.create_index("is_active")
            await self.users_collection.create_index([("created_at", -1)])
            
            logger.info("Database indexes created successfully")
            
        except Exception as e:
            logger.warning(f"Error creating indexes: {e}")

    # def __init__(self, mongodb_uri: str, database_name: str = "emoai2"):
    #     self.mongodb_uri = mongodb_uri
    #     self.database_name = database_name
    #     self.client = None
    #     self.db = None
    #     self.tasks_collection = None
    #     self.videos_collection = None
    #     self.stats_collection = None
        
    #     logger.info(f"MongoDB manager initialized for database: {database_name}")

    # async def initialize(self):
    #     """Initialize MongoDB connection and create indexes"""
    #     try:
    #         self.client = motor.motor_asyncio.AsyncIOMotorClient(self.mongodb_uri)
    #         self.db = self.client[self.database_name]
            
    #         # Initialize collections
    #         self.tasks_collection = self.db.processing_tasks
    #         self.videos_collection = self.db.processed_videos
    #         self.stats_collection = self.db.processing_stats
            
    #         # Test connection
    #         await self.client.admin.command('ping')
    #         logger.info("MongoDB connection established successfully")
            
    #         # Create indexes for better performance
    #         await self._create_indexes()
            
    #     except Exception as e:
    #         logger.error(f"Failed to connect to MongoDB: {e}")
    #         raise

    # async def _create_indexes(self):
    #     """Create database indexes for optimal performance"""
    #     try:
    #         # Tasks collection indexes
    #         await self.tasks_collection.create_index("task_id", unique=True)
    #         await self.tasks_collection.create_index("user_id")
    #         await self.tasks_collection.create_index("status")
    #         await self.tasks_collection.create_index([("created_at", -1)])  # Descending for recent first
    #         await self.tasks_collection.create_index([("user_id", 1), ("created_at", -1)])
    #         await self.tasks_collection.create_index("celery_task_id")
            
    #         # Videos collection indexes  
    #         await self.videos_collection.create_index("user_id")
    #         await self.videos_collection.create_index("session_id", unique=True)
    #         await self.videos_collection.create_index([("created_at", -1)])
            
    #         logger.info("Database indexes created successfully")
            
    #     except Exception as e:
    #         logger.warning(f"Error creating indexes: {e}")

    async def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")

    # Task Management Methods
    
    async def create_task_record(self, task_data: Dict[str, Any]) -> str:
        """Create a new task record"""
        try:
            task_data["created_at"] = datetime.utcnow()
            task_data["updated_at"] = datetime.utcnow()
            
            result = await self.tasks_collection.insert_one(task_data)
            logger.info(f"Task record created: {task_data['task_id']}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Error creating task record: {e}")
            raise

    async def update_task_status(self, task_id: str, status: str, error: str = None, result: Dict[str, Any] = None):
        """Update task status with optional error or result data"""
        try:
            update_data = {
                "status": status,
                "updated_at": datetime.utcnow()
            }
            
            if error:
                update_data["error"] = error
            
            if result:
                update_data["result"] = result
            
            await self.tasks_collection.update_one(
                {"task_id": task_id},
                {"$set": update_data}
            )
            
            logger.info(f"Task {task_id} status updated to: {status}")
            
        except Exception as e:
            logger.error(f"Error updating task status: {e}")
            raise

    async def update_task_data(self, task_id: str, data: Dict[str, Any]):
        """Update task with additional data"""
        try:
            data["updated_at"] = datetime.utcnow()
            
            await self.tasks_collection.update_one(
                {"task_id": task_id},
                {"$set": data}
            )
            
            logger.debug(f"Task {task_id} data updated")
            
        except Exception as e:
            logger.error(f"Error updating task data: {e}")
            raise

    async def get_task_by_id(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task by ID"""
        try:
            task = await self.tasks_collection.find_one({"task_id": task_id}, {"_id": 0})
            return task
            
        except Exception as e:
            logger.error(f"Error getting task {task_id}: {e}")
            return None

    async def get_user_tasks(self, user_id: str, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Get all tasks for a user with pagination"""
        try:
            cursor = self.tasks_collection.find(
                {"user_id": user_id},
                {"_id": 0}
            ).sort("created_at", -1).skip(offset).limit(limit)
            
            tasks = await cursor.to_list(length=limit)
            return tasks
            
        except Exception as e:
            logger.error(f"Error getting tasks for user {user_id}: {e}")
            return []

    async def count_user_tasks(self, user_id: str) -> int:
        """Count total tasks for a user"""
        try:
            count = await self.tasks_collection.count_documents({"user_id": user_id})
            return count
            
        except Exception as e:
            logger.error(f"Error counting tasks for user {user_id}: {e}")
            return 0

    async def get_tasks_by_status(self, status: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get tasks by status"""
        try:
            cursor = self.tasks_collection.find(
                {"status": status},
                {"_id": 0}
            ).sort("created_at", -1).limit(limit)
            
            tasks = await cursor.to_list(length=limit)
            return tasks
            
        except Exception as e:
            logger.error(f"Error getting tasks with status {status}: {e}")
            return []

    async def cleanup_old_tasks(self, days: int = 30):
        """Clean up old completed/failed tasks"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            result = await self.tasks_collection.delete_many({
                "status": {"$in": ["completed", "failed", "cancelled"]},
                "created_at": {"$lt": cutoff_date}
            })
            
            logger.info(f"Cleaned up {result.deleted_count} old tasks")
            return result.deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old tasks: {e}")
            return 0

    # Video Records Management (Legacy support)
    
    async def store_video_record(self, user_id: str, session_id: str, gdrive_data: Dict[str, Any], 
                                processing_data: Dict[str, Any]) -> Dict[str, Any]:
        """Store processed video record"""
        try:
            video_record = {
                "user_id": user_id,
                "session_id": session_id,
                "gdrive_data": gdrive_data,
                "processing_data": processing_data,
                "created_at": datetime.utcnow()
            }
            
            result = await self.videos_collection.insert_one(video_record)
            
            return {
                "success": True,
                "document_id": str(result.inserted_id)
            }
            
        except Exception as e:
            logger.error(f"Error storing video record: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def get_video_by_session(self, session_id: str) -> Dict[str, Any]:
        """Get video record by session ID"""
        try:
            video = await self.videos_collection.find_one(
                {"session_id": session_id}, 
                {"_id": 0}
            )
            return video or {}
            
        except Exception as e:
            logger.error(f"Error getting video by session {session_id}: {e}")
            return {}

    async def get_user_videos(self, user_id: str, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Get processed videos for a user"""
        try:
            cursor = self.videos_collection.find(
                {"user_id": user_id},
                {"_id": 0}
            ).sort("created_at", -1).skip(offset).limit(limit)
            
            videos = await cursor.to_list(length=limit)
            return videos
            
        except Exception as e:
            logger.error(f"Error getting videos for user {user_id}: {e}")
            return []

    # Statistics and Analytics
    
    async def get_processing_stats(self, user_id: str = None) -> Dict[str, Any]:
        """Get processing statistics"""
        try:
            match_filter = {"user_id": user_id} if user_id else {}
            
            # Aggregate statistics
            pipeline = [
                {"$match": match_filter},
                {
                    "$group": {
                        "_id": None,
                        "total_tasks": {"$sum": 1},
                        "completed_tasks": {
                            "$sum": {"$cond": [{"$eq": ["$status", "completed"]}, 1, 0]}
                        },
                        "failed_tasks": {
                            "$sum": {"$cond": [{"$eq": ["$status", "failed"]}, 1, 0]}
                        },
                        "pending_tasks": {
                            "$sum": {"$cond": [{"$in": ["$status", ["received", "transcribing", "optimizing", "generating_audio", "queued_for_video_processing", "video_processing"]]}, 1, 0]}
                        },
                        "cancelled_tasks": {
                            "$sum": {"$cond": [{"$eq": ["$status", "cancelled"]}, 1, 0]}
                        },
                        "total_cost": {"$sum": "$result.cost_summary.estimated_cost_usd"}
                    }
                }
            ]
            
            result = await self.tasks_collection.aggregate(pipeline).to_list(1)
            
            if result:
                stats = result[0]
                del stats["_id"]
                
                # Calculate success rate
                if stats["total_tasks"] > 0:
                    stats["success_rate"] = stats["completed_tasks"] / stats["total_tasks"]
                else:
                    stats["success_rate"] = 0
                
                return stats
            else:
                return {
                    "total_tasks": 0,
                    "completed_tasks": 0,
                    "failed_tasks": 0,
                    "pending_tasks": 0,
                    "cancelled_tasks": 0,
                    "success_rate": 0,
                    "total_cost": 0
                }
                
        except Exception as e:
            logger.error(f"Error getting processing stats: {e}")
            return {}

    async def get_daily_stats(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get daily processing statistics for the last N days"""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            pipeline = [
                {"$match": {"created_at": {"$gte": start_date}}},
                {
                    "$group": {
                        "_id": {
                            "$dateToString": {
                                "format": "%Y-%m-%d",
                                "date": "$created_at"
                            }
                        },
                        "total": {"$sum": 1},
                        "completed": {
                            "$sum": {"$cond": [{"$eq": ["$status", "completed"]}, 1, 0]}
                        },
                        "failed": {
                            "$sum": {"$cond": [{"$eq": ["$status", "failed"]}, 1, 0]}
                        }
                    }
                },
                {"$sort": {"_id": 1}}
            ]
            
            results = await self.tasks_collection.aggregate(pipeline).to_list(days)
            
            # Rename _id to date for cleaner output
            for result in results:
                result["date"] = result["_id"]
                del result["_id"]
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting daily stats: {e}")
            return []

    # Health and Monitoring
    
    async def health_check(self) -> Dict[str, Any]:
        """Check MongoDB health and connectivity"""
        try:
            # Test basic connectivity
            await self.client.admin.command('ping')
            
            # Get server info
            server_info = await self.client.server_info()
            
            # Count documents in main collections
            task_count = await self.tasks_collection.estimated_document_count()
            video_count = await self.videos_collection.estimated_document_count()
            
            return {
                "status": "healthy",
                "server_version": server_info.get("version", "unknown"),
                "task_count": task_count,
                "video_count": video_count,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"MongoDB health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    # Context manager support
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()