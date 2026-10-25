#!/usr/bin/env bash
######################################################################
# This file is read-only. To edit, change the README.org file.
######################################################################

export DEBIAN_FRONTEND=noninteractive

apt-get update

apt-get install -y --no-install-recommends apt-utils 2>&1 | grep -v "debconf: delaying package configuration, since apt-utils is not installed"

apt-get -y upgrade
