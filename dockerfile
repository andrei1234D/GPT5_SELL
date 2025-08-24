# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose ports used by Flask (Cloud Run requires a web port)
EXPOSE 8080

# Run your bot (same command as in .replit)
CMD ["python", "bot/discord_listener.py"]

