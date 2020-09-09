APP_NAME=tweak-story
DOCKER_CMD=$(shell which docker || which podman || echo "docker")
PORT=8501
MODEL_CHECKPOINT_NAME=BEST_checkpoint_flickr8k_1_cap_per_img_1_min_word_freq.pth
WORD_MAP_CHECKPOINT_NAME=WORDMAP_flickr8k_1_cap_per_img_1_min_word_freq.json
CONFIG_NAME=config.json
CHECKPOINT_LOCATION=http://tweakstory.stefanmesken.info/checkpoints

.get_checkpoint/%:
	test -f ckpts/$* || wget $(CHECKPOINT_LOCATION)/$* -P ckpts/

get_checkpoints: .get_checkpoint/$(CONFIG_NAME)\
	             .get_checkpoint/$(WORD_MAP_CHECKPOINT_NAME)\
				 .get_checkpoint/$(MODEL_CHECKPOINT_NAME)

build: get_checkpoints
	$(DOCKER_CMD) build -t $(APP_NAME) .

run: build
	$(DOCKER_CMD) run -d -p=$(PORT):$(PORT) --rm --name=$(APP_NAME) $(APP_NAME)
	@echo $(APP_NAME) running at localhost:$(PORT)

stop:
	$(DOCKER_CMD) stop $(APP_NAME) || true
