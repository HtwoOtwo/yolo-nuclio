#!/bin/bash

func_config="$1"
func_root=$(dirname "$func_config")

docker build -t yolo .
nuctl deploy --project-name cvat --path "$func_root" --file "$func_config" --platform local