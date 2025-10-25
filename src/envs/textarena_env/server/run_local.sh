TEXTARENA_ENV_ID="Wordle-v0" TEXTARENA_NUM_PLAYERS=2

# Run the server
exec uvicorn envs.textarena_env.server.app:app --host 0.0.0.0 --port 8000


