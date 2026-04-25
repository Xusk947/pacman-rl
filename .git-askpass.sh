#!/bin/sh
case "$1" in
  *Username*) echo "x-access-token" ;;
  *) echo "${GITHUB_TOKEN:-}" ;;
esac
