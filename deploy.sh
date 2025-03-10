#!/bin/bash

func_config="$1"
func_root=$(dirname "$func_config")

docker build -t yolo .
nuctl create project yolo # create project yolo
nuctl deploy --project-name yolo --path "$func_root" --file "$func_config" --platform local