import os

from flask import Flask, jsonify

app = Flask(__name__)
DATABASE_URL = os.environ.get("DATABASE_URL")
REDIS_URL = os.environ.get("REDIS_URL")
SECRET_KEY = os.environ.get("SECRET_KEY")


@app.get("/")
def index():
    return jsonify(status="ok")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5050")))
