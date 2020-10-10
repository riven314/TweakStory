APP_NAME=tweak-story
DOCKER_CMD=$(shell which docker || which podman || echo "docker")
PORT=8501

build: 
	$(DOCKER_CMD) build -t $(APP_NAME) .

start:
	$(DOCKER_CMD) run -d -p=$(PORT):$(PORT) --rm --name=$(APP_NAME) $(APP_NAME)
	@echo $(APP_NAME) running at localhost:$(PORT)

run: build start

stop:
	$(DOCKER_CMD) stop $(APP_NAME) || true
