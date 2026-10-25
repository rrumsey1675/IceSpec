######################################################################
# This file is read-only. To edit, change the README.org file.
######################################################################

FROM jupyter/datascience-notebook:ubuntu-22.04

USER root
COPY install.sh .
RUN ./install.sh && rm ./install.sh

USER jovyan
COPY jovyan-install.sh .
RUN ./jovyan-install.sh && rm ./jovyan-install.sh
