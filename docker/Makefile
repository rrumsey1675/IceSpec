######################################################################
# This file is read-only. To edit, change the README.org file.
######################################################################

registry_port:=5050
registry:=hsi.um.maine.edu
project_path:=/hsi/hsi-tools
build_tag:=$(shell date +'%Y%m%d')
tag:=latest
build_name:=$(registry):$(registry_port)$(project_path):$(build_tag)
run_name:=$(registry):$(registry_port)$(project_path):$(tag)
jupyter_port:=8888

PHONY=build upload serve bash

build:
	docker build -t $(build_name) .
	docker tag $(build_name) $(run_name)

upload:
	docker push $(build_name)
	docker push $(run_name)

serve:
	docker run -p $(jupyter_port):8888 --rm -v "$(PWD)":/home/jovyan/ \
                   $(run_name)

bash:
	docker run -p $(jupyter_port):8888 --rm -v "$(PWD)":/home/jovyan/ \
                   -it --cpus="4" $(run_name) bash
