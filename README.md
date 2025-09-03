# AutoEdit

A FastAPI-based video processing application that provides automated video editing capabilities through a REST API.

## Features

- **Video Processing API**: RESTful endpoint for processing video files
- **Docker Support**: Containerized deployment for easy setup and deployment
- **Environment Configuration**: Flexible configuration through environment variables

## Prerequisites

- Docker and Docker Compose installed on your system
- Video files for testing the API

## Quick Start

### 1. Environment Setup

First, configure your environment variables. Create a `.env` file in the project root and add all the required configuration values:

```bash
# Example .env file structure
# Add your specific configuration values here
API_KEY=your_api_key_here
DATABASE_URL=your_database_url
# ... other required environment variables
```

### 2. Build and Run with Docker

Build the Docker image without using cache to ensure fresh dependencies:

```bash
docker compose -f docker-compose.yml build --no-cache
```

Start the application:

```bash
docker compose -f docker-compose.yml up
```

If facing issues with Docker command:

Check entrypoint.sh - In code editor check bottom right for "CRLF" -> change it to "LF" and then save the file!
and rerun the docker command! - fix for line endings in windows

then run these commands:

```bash
docker compose down

docker compose build --no-cache

docker compose up
```

The API will be available at the configured port (typically `http://localhost:8000`).

### 3. Test the API

Test the video processing endpoint by sending a POST request to `/process-video`:

- **Endpoint**: `POST /process-video`
- **Input**: Video file and user_id parameter
- **User ID**: Use any random string as the user_id for testing purposes

#### Example API Test

```bash
curl -X POST "http://localhost:8000/process-video" \
  -H "Content-Type: multipart/form-data" \
  -F "video=@your_video_file.mp4" \
  -F "user_id=test_user_123"
```
#### Testing via Postman
POST http://localhost:8000/process-video?user_id=test_user_123
Body -> form-data
key = "file"
value = "select the input video file" ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v')

GET http://localhost:8000/user/test_user_123/tasks
To check the task status... Once completed this will return the google drive link for the output video!

## API Documentation

Once the application is running, you can access the interactive API documentation at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Development

For development purposes, you can also run the application locally without Docker by installing the Python dependencies and running the FastAPI server directly.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

[Add your license information here]

## Support

For issues and questions, please create an issue in the repository or contact the development team.