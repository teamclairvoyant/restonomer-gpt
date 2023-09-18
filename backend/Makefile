.PHONY: start
start:
	uvicorn chatbot-api:app --reload --port 8080

.PHONY: format
format:
	black .
	isort .