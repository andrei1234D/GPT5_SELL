from flask import Flask
import os

app = Flask(__name__)

@app.route('/')
def home():
    return "✅ Bot is running!"

def keep_alive():
    port = int(os.environ.get("PORT", 8080))  # use Replit’s assigned port if available
    app.run(host="0.0.0.0", port=port)

if __name__ == "__main__":
    keep_alive()
